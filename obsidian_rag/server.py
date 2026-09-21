"""obsidian-rag MCP server.

Run with:
    uv run obsidian-rag-mcp                              # streamable HTTP (default)
    uv run obsidian-rag-mcp --transport stdio            # local development only

Required environment variables (set by the MCP client when registering the server):
    OBSIDIAN_VAULT_PATH   absolute path to the Obsidian vault

Embedding configuration (backend is auto-detected from EMBEDDING_BASE_URL):
    EMBEDDING_BASE_URL    OpenAI-compatible base URL or Ollama host
    EMBEDDING_MODEL       model name
    EMBEDDING_API_KEY     API key (only for OpenAI-compatible endpoints)
    EMBEDDING_PROVIDER    optional override: openai | ollama | fake
"""

from __future__ import annotations

import json
import os
import secrets
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings

from .config import Settings, load_settings
from .embeddings import EmbeddingClient, EmbeddingError
from .remote_vault import sync_remote_vault
from .store import VectorStore, build_index
from .vault import scan_vault

# --------------------------------------------------------------------------- #
# State (loaded lazily so `--help`/validation can run without a vault)
# --------------------------------------------------------------------------- #
_settings: Settings | None = None
_client: EmbeddingClient | None = None
_store: VectorStore | None = None


class StaticTokenVerifier:
    """Verify a bearer token against MCP_AUTH_TOKEN."""

    async def verify_token(self, token: str) -> AccessToken | None:
        expected = os.environ.get("MCP_AUTH_TOKEN", "")
        if not expected or not secrets.compare_digest(token, expected):
            return None
        return AccessToken(token=token, client_id="static-token", scopes=[])


_token_verifier = StaticTokenVerifier()


def _ensure() -> tuple[Settings, EmbeddingClient, VectorStore]:
    global _settings, _client, _store
    if _settings is None:
        _settings = load_settings()
    if _client is None:
        _client = EmbeddingClient(_settings)
    if _store is None:
        _store = VectorStore(_settings.index_path, _settings.model)
    return _settings, _client, _store


def _refresh_remote_vault(force: bool = False) -> None:
    """Refresh only when explicitly requested by the administrative tool."""
    global _settings, _store
    if (
        not force
        or _settings is None
        or not os.environ.get("VAULT_REMOTE_PROVIDER", "").strip()
    ):
        return
    new_vault = sync_remote_vault(force=True)
    _settings.vault_path = new_vault
    _store = VectorStore(_settings.index_path, _settings.model)



def _err(msg: str) -> str:
    return json.dumps({"error": msg}, ensure_ascii=False)


def _ok(**kwargs: Any) -> str:
    return json.dumps(kwargs, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# MCP server
# --------------------------------------------------------------------------- #
mcp = FastMCP(
    "obsidian-rag",
    instructions=(
        "RAG tools for an Obsidian vault. Use obsidian_index to build the "
        "embeddings index, then obsidian_search / obsidian_rag to retrieve "
        "relevant notes and analyze the user's input against them."
    ),
    token_verifier=_token_verifier,
    auth=AuthSettings(
        issuer_url="https://localhost",
        resource_server_url="https://localhost/mcp",
    ),
    host=os.environ.get("MCP_HOST", "0.0.0.0"),
    port=int(os.environ.get("MCP_PORT", "8000")),
    streamable_http_path=os.environ.get("MCP_PATH", "/mcp"),
    stateless_http=True,
)


@mcp.tool()
def obsidian_get_config() -> str:
    """Show the current configuration (vault path, embedding provider/model).

    The API key is never returned.
    """
    try:
        settings, _, _ = _ensure()
    except ValueError as exc:
        return _err(str(exc))
    return _ok(config=settings.to_dict())


@mcp.tool()
def obsidian_index(force: bool = False) -> str:
    """Scan the vault and build/refresh the embedding index.

    Args:
        force: rebuild the index even if one already exists.
    """
    try:
        _refresh_remote_vault()
        settings, client, store = _ensure()
        _ = _store  # already assigned
    except ValueError as exc:
        return _err(str(exc))
    try:
        stats = build_index(settings, client, store, force=force)
    except (RuntimeError, EmbeddingError) as exc:
        return _err(str(exc))
    return _ok(**stats)


@mcp.tool()
def obsidian_refresh() -> str:
    """Force-refresh the remote vault mirror, then rebuild the embedding index.

    This bypasses the normal mirror cache and performs both operations in one
    call. The remote vault is read-only; only the local mirror and index change.
    """
    try:
        _ensure()
        _refresh_remote_vault(force=True)
        settings, client, store = _ensure()
        stats = build_index(settings, client, store, force=True)
    except (ValueError, RuntimeError, EmbeddingError) as exc:
        return _err(str(exc))
    return _ok(refreshed=True, **stats)


@mcp.tool()
def obsidian_search(query: str, top_k: int = 5) -> str:
    """Semantically search the vault for chunks related to `query`.

    Args:
        query: what to look for (natural language).
        top_k: number of results to return (1-20).
    """
    try:
        _refresh_remote_vault()
        settings, client, store = _ensure()
    except ValueError as exc:
        return _err(str(exc))

    if not store.load():
        return _err(
            "No index found. Call obsidian_index first to build the index."
        )
    try:
        qv = client.embed_one(query)
    except (EmbeddingError, RuntimeError) as exc:
        return _err(str(exc))
    results = store.search(qv, top_k=max(1, min(20, top_k)))
    return _ok(results=results, count=len(results))


@mcp.tool()
def obsidian_rag(question: str, top_k: int = 5) -> str:
    """Retrieve the most relevant Obsidian notes for `question` and return them
    as context.

    Use this to analyze a user's message against what is stored in the vault:
    search first, then reason over the returned context.

    Args:
        question: the user's question / content to analyze.
        top_k: number of context chunks to retrieve (1-20).
    """
    try:
        _refresh_remote_vault()
        settings, client, store = _ensure()
    except ValueError as exc:
        return _err(str(exc))

    if not store.load():
        return _err(
            "No index found. Call obsidian_index first to build the index."
        )
    try:
        qv = client.embed_one(question)
    except (EmbeddingError, RuntimeError) as exc:
        return _err(str(exc))
    results = store.search(qv, top_k=max(1, min(20, top_k)))
    return _ok(
        question=question,
        model=store.model,
        context=results,
        count=len(results),
    )


@mcp.tool()
def obsidian_list_notes(keyword: str | None = None) -> str:
    """List the markdown notes in the vault."""
    try:
        _refresh_remote_vault()
        settings, _, _ = _ensure()
    except ValueError as exc:
        return _err(str(exc))
    files = scan_vault(settings.vault_path, settings.max_notes)
    notes = [f.relative_to(settings.vault_path).as_posix() for f in files]
    if keyword:
        notes = [n for n in notes if keyword.lower() in n.lower()]
    return _ok(count=len(notes), notes=notes)


@mcp.tool()
def obsidian_read_note(path: str) -> str:
    """Read the full text of a note from the vault.

    Args:
        path: vault-relative markdown path, e.g. "Projects/MyProject.md".
    """
    try:
        _refresh_remote_vault()
        settings, _, _ = _ensure()
    except ValueError as exc:
        return _err(str(exc))

    target = (settings.vault_path / path).resolve()
    vault_root = settings.vault_path.resolve()
    # Guard against path traversal
    if not target.is_relative_to(vault_root):
        return _err("Path escapes the vault directory.")
    if not target.is_file() or target.suffix.lower() != ".md":
        return _err(f"Not a markdown file inside the vault: {path}")
    try:
        content = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return _err(f"Failed to read {path}: {exc}")
    return _ok(path=path, content=content)


@mcp.tool()
def obsidian_index_status() -> str:
    """Report whether an index exists and matches the configured embedding model."""
    try:
        _refresh_remote_vault()
        settings, _, store = _ensure()
    except ValueError as exc:
        return _err(str(exc))
    if store.load():
        notes = len({r["path"] for r in store.records})
        return _ok(
            indexed=True,
            model=store.model,
            notes=notes,
            chunks=len(store.records),
            index_path=str(store.index_path),
        )
    return _ok(
        indexed=False,
        model=settings.model,
        index_path=str(settings.index_path),
        note="Run obsidian_index to build it.",
    )


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    """CLI entry point (used by `uv run obsidian-rag-mcp`)."""
    argv = sys.argv[1:] if argv is None else argv

    if argv and argv[0] in ("-h", "--help"):
        print(
            "obsidian-rag MCP server\n"
            "Usage:\n"
            "  obsidian-rag-mcp                              run over HTTP (default)\n"
            "  obsidian-rag-mcp --transport stdio            local development only\n"
            "  obsidian-rag-mcp --check                    validate configuration\n"
            "Environment:\n"
            "  OBSIDIAN_VAULT_PATH (required), EMBEDDING_BASE_URL,\n"
            "  EMBEDDING_MODEL, EMBEDDING_API_KEY, EMBEDDING_PROVIDER (optional),\n"
            "  OBSIDIAN_INDEX_PATH, OBSIDIAN_CHUNK_SIZE, OBSIDIAN_MAX_NOTES,\n"
            "  MCP_AUTH_TOKEN (required for HTTP), MCP_HOST, MCP_PORT, MCP_PATH"
        )
        return

    if argv and argv[0] == "--check":
        try:
            settings = load_settings()
        except ValueError as exc:
            print(f"[check] FAIL: {exc}")
            sys.exit(1)
        if not os.environ.get("MCP_AUTH_TOKEN"):
            print("[check] FAIL: MCP_AUTH_TOKEN is not set")
            sys.exit(1)
        print(
            "[check] OK\n"
            f"  vault  : {settings.vault_path}\n"
            f"  provider: {settings.provider}\n"
            f"  base_url: {settings.base_url}\n"
            f"  model  : {settings.model}\n"
            f"  api_key: {'set' if settings.api_key else 'NOT SET'}"
        )
        return

    if argv and argv[0] == "--transport":
        transport = argv[1] if len(argv) > 1 else "stdio"
        if transport == "stdio":
            mcp.run(transport="stdio")
        else:
            mcp.run(transport=transport)
        return

    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()

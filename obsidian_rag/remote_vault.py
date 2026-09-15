"""Read-only synchronization of an Obsidian vault from S3 or WebDAV."""

from __future__ import annotations

import json
import os
import posixpath
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote, urljoin, urlparse

import httpx


class VaultSyncError(ValueError):
    """Raised when a remote vault cannot be mirrored locally."""


def _new_dir() -> Path:
    configured = os.environ.get("OBSIDIAN_MIRROR_PATH", "/data/vault").strip()
    root = Path(configured).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _mirror_fresh(root: Path) -> bool:
    try:
        hours = float(os.environ.get("OBSIDIAN_MIRROR_REFRESH_HOURS", "6"))
    except ValueError:
        hours = 6.0
    marker = root / ".obsidian-rag-sync.json"
    if hours <= 0 or not marker.is_file():
        return False
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
        return time.time() - float(data["completed_at"]) < hours * 3600
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _mark_synced(root: Path, provider: str, files: list[str]) -> None:
    (root / ".obsidian-rag-sync.json").write_text(
        json.dumps({"provider": provider, "files": files, "completed_at": time.time()}),
        encoding="utf-8",
    )


def _prepare_mirror(root: Path, provider: str, files: list[str]) -> None:
    marker = root / ".obsidian-rag-sync.json"
    try:
        old = json.loads(marker.read_text(encoding="utf-8"))
        old_files = set(old.get("files", []))
    except (OSError, json.JSONDecodeError):
        old_files = set()
    current = set(files)
    for name in old_files - current:
        target = _safe_target(root, name)
        if target.is_file():
            target.unlink()


def _safe_target(root: Path, name: str) -> Path:
    name = name.replace("\\", "/").lstrip("/")
    target = (root / name).resolve()
    if not target.is_relative_to(root.resolve()):
        raise VaultSyncError(f"Remote path escapes vault root: {name}")
    return target


def _write_file(root: Path, name: str, content: bytes) -> None:
    target = _safe_target(root, name)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)


def _s3_settings() -> dict[str, str]:
    required = ["VAULT_S3_BUCKET", "VAULT_S3_PREFIX"]
    missing = [key for key in required if not os.environ.get(key, "").strip()]
    if missing:
        raise VaultSyncError("Missing S3 settings: " + ", ".join(missing))
    prefix = os.environ.get("VAULT_S3_PREFIX", "").strip().strip("/")
    return {
        "bucket": os.environ["VAULT_S3_BUCKET"],
        "prefix": prefix + "/" if prefix and prefix != "." else "",
        "endpoint": os.environ.get("VAULT_S3_ENDPOINT", "").strip(),
        "region": os.environ.get("VAULT_S3_REGION", "us-east-1").strip(),
    }


def sync_s3() -> Path:
    try:
        import boto3
    except ImportError as exc:
        raise VaultSyncError(
            "S3 support requires boto3. Install dependencies-s3.txt."
        ) from exc

    cfg = _s3_settings()
    kwargs = {"region_name": cfg["region"]}
    if cfg["endpoint"]:
        kwargs["endpoint_url"] = cfg["endpoint"]
    client = boto3.client("s3", **kwargs)
    root = _new_dir()
    paginator = client.get_paginator("list_objects_v2")
    found: list[str] = []
    for page in paginator.paginate(Bucket=cfg["bucket"], Prefix=cfg["prefix"]):
        for item in page.get("Contents", []):
            key = item["Key"]
            relative = key[len(cfg["prefix"]):] if key.startswith(cfg["prefix"]) else key
            if not relative or relative.endswith("/"):
                continue
            response = client.get_object(Bucket=cfg["bucket"], Key=key)
            _write_file(root, relative, response["Body"].read())
            found.append(relative)
    if not found:
        raise VaultSyncError("S3 prefix contains no files.")
    _prepare_mirror(root, "s3", found)
    _mark_synced(root, "s3", found)
    return root


def _webdav_settings() -> tuple[str, str, str]:
    url = os.environ.get("VAULT_WEBDAV_URL", "").strip()
    username = os.environ.get("VAULT_WEBDAV_USERNAME", "").strip()
    password = os.environ.get("VAULT_WEBDAV_PASSWORD", "")
    if not url:
        raise VaultSyncError("VAULT_WEBDAV_URL is not set.")
    if not url.endswith("/"):
        url += "/"
    return url, username, password


def _dav_href_to_name(base_url: str, href: str) -> str:
    path = urlparse(href).path if "://" in href else href
    base_path = urlparse(base_url).path
    if path.startswith(base_path):
        path = path[len(base_path):]
    return path.lstrip("/")


def sync_webdav() -> Path:
    url, username, password = _webdav_settings()
    auth = (username, password) if username else None
    root = _new_dir()
    client = httpx.Client(auth=auth, follow_redirects=True, timeout=60)
    try:
        def walk(collection_url: str, relative_dir: str = "") -> list[str]:
            response = client.request("PROPFIND", collection_url, headers={"Depth": "1"})
            response.raise_for_status()
            try:
                xml_root = ET.fromstring(response.content)
            except ET.ParseError as exc:
                raise VaultSyncError("WebDAV PROPFIND returned invalid XML.") from exc
            files: list[str] = []
            ns = {"d": "DAV:"}
            for item in xml_root.findall("d:response", ns):
                href_node = item.find("d:href", ns)
                if href_node is None or not href_node.text:
                    continue
                href = href_node.text
                child_url = urljoin(collection_url, quote(href, safe="/%:@"))
                name = _dav_href_to_name(url, href).rstrip("/")
                if not name:
                    continue
                rel = name if not relative_dir else posixpath.relpath(name, relative_dir)
                is_collection = item.find(".//d:collection", ns) is not None
                if is_collection:
                    files.extend(walk(child_url if child_url.endswith("/") else child_url + "/", name))
                elif name.lower().endswith(".md"):
                    target_name = name
                    response = client.get(child_url)
                    response.raise_for_status()
                    _write_file(root, target_name, response.content)
                    files.append(target_name)
            return files

        found = walk(url)
    finally:
        client.close()
    if not found:
        raise VaultSyncError("WebDAV contains no markdown files.")
    _prepare_mirror(root, "webdav", found)
    _mark_synced(root, "webdav", found)
    return root


def sync_remote_vault(force: bool = False) -> Path:
    """Create a fresh local read-only mirror for the configured remote."""
    provider = os.environ.get("VAULT_REMOTE_PROVIDER", "").strip().lower()
    root = _new_dir()
    if not force and _mirror_fresh(root):
        return root
    if provider == "s3":
        return sync_s3()
    if provider == "webdav":
        return sync_webdav()
    raise VaultSyncError("VAULT_REMOTE_PROVIDER must be 's3' or 'webdav'.")

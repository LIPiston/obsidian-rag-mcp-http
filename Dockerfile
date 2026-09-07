FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml README.md dependencies-s3.txt ./
RUN pip install --no-cache-dir "mcp[cli]>=1.2,<2.0" "httpx>=0.27" \
    -r dependencies-s3.txt

COPY obsidian_rag ./obsidian_rag

EXPOSE 8000
CMD ["python", "-m", "obsidian_rag.server"]

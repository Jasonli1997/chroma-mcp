#!/usr/bin/env python3
"""
Build or refresh a Chroma vector index from an Obsidian vault.

Examples:
  python scripts/obsidian.py --vault /path/to/vault --collection obsidian build
  python scripts/obsidian.py --vault /path/to/vault --collection obsidian refresh --delete-removed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

from chromadb.api import EmbeddingFunction
from chromadb.api.collection_configuration import CreateCollectionConfiguration
from dotenv import load_dotenv

from src.chroma_mcp.server import get_chroma_client, mcp_known_embedding_functions

DEFAULT_EXCLUDE_DIRS = {
    ".obsidian",
    ".trash",
    ".git",
    ".github",
    ".idea",
    ".vscode",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
}

TAG_RE = re.compile(r"(?<![\w/])#([A-Za-z0-9_/-]+)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    text: str
    start: int
    end: int
    start_line: int
    end_line: int
    index: int
    heading_path: str


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or refresh a Chroma vector index from an Obsidian vault"
    )
    parser.add_argument("--vault", required=True, help="Path to the Obsidian vault")
    parser.add_argument("--collection", required=True, help="Chroma collection name")
    parser.add_argument(
        "--client-type",
        choices=["http", "cloud", "persistent", "ephemeral"],
        default=os.getenv("CHROMA_CLIENT_TYPE", "persistent"),
    )
    parser.add_argument(
        "--data-dir",
        default=os.getenv("CHROMA_DATA_DIR"),
        help="Directory for persistent client data (persistent client only)",
    )
    parser.add_argument("--host", default=os.getenv("CHROMA_HOST"))
    parser.add_argument("--port", default=os.getenv("CHROMA_PORT"))
    parser.add_argument(
        "--custom-auth-credentials",
        default=os.getenv("CHROMA_CUSTOM_AUTH_CREDENTIALS"),
    )
    parser.add_argument("--tenant", default=os.getenv("CHROMA_TENANT"))
    parser.add_argument("--database", default=os.getenv("CHROMA_DATABASE"))
    parser.add_argument("--api-key", default=os.getenv("CHROMA_API_KEY"))
    parser.add_argument(
        "--ssl",
        type=lambda x: x.lower() in ["true", "yes", "1", "t", "y"],
        default=os.getenv("CHROMA_SSL", "true").lower() in ["true", "yes", "1", "t", "y"],
    )
    parser.add_argument(
        "--dotenv-path",
        default=os.getenv("CHROMA_DOTENV_PATH", ".chroma_env"),
        help="Path to .env file",
    )
    parser.add_argument(
        "--embedding-function",
        choices=["default", "cohere", "openai", "jina", "voyageai", "roboflow"],
        default="default",
    )
    parser.add_argument(
        "--embedding-kwargs",
        default=None,
        help="JSON string of kwargs for the embedding function",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help="Max characters per chunk",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap (characters) between chunks",
    )
    parser.add_argument(
        "--include-ext",
        action="append",
        default=[".md"],
        help="File extensions to include (repeatable)",
    )
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        help="Directory names to exclude (repeatable)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for collection.add",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without modifying the collection",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("build", help="Build a full index (replaces existing docs)")
    refresh_parser = subparsers.add_parser(
        "refresh", help="Refresh only stale or missing documents"
    )
    refresh_parser.add_argument(
        "--delete-removed",
        action="store_true",
        help="Delete documents for files removed from the vault",
    )

    return parser


def get_embedding_function(name: str, kwargs: Dict | None) -> EmbeddingFunction:
    cls = mcp_known_embedding_functions[name]
    if kwargs:
        return cls(**kwargs)
    return cls()


def iter_markdown_files(vault_path: str, include_exts: Iterable[str], exclude_dirs: Iterable[str]):
    include_exts = {ext.lower() for ext in include_exts}
    exclude_dirs = {d.lower() for d in exclude_dirs}
    for root, dirs, files in os.walk(vault_path):
        dirs[:] = [
            d
            for d in dirs
            if d.lower() not in exclude_dirs and not d.startswith(".")
        ]
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in include_exts:
                yield os.path.join(root, name)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return handle.read()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def parse_frontmatter(text: str) -> Tuple[Dict, str, str]:
    if not text.startswith("---"):
        return {}, text, ""
    lines = text.splitlines()
    if len(lines) < 2 or lines[0].strip() != "---":
        return {}, text, ""
    end_idx = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end_idx = idx
            break
    if end_idx is None:
        return {}, text, ""
    fm_lines = lines[1:end_idx]
    fm_text = "\n".join(fm_lines)
    content = "\n".join(lines[end_idx + 1 :])
    return parse_simple_yaml(fm_lines), content, fm_text


def parse_simple_yaml(lines: List[str]) -> Dict:
    data: Dict[str, object] = {}
    current_key: str | None = None
    for line in lines:
        raw = line.strip()
        if not raw:
            continue
        if ":" in raw and not raw.startswith("-"):
            key, rest = raw.split(":", 1)
            key = key.strip()
            rest = rest.strip()
            if rest == "":
                data[key] = []
                current_key = key
            else:
                data[key] = parse_scalar(rest)
                current_key = key
            continue
        if raw.startswith("-") and current_key:
            item = raw[1:].strip()
            if not isinstance(data.get(current_key), list):
                data[current_key] = [data[current_key]]
            data[current_key].append(parse_scalar(item))
            continue
        current_key = None
    return data


def parse_scalar(value: str):
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_scalar(v.strip()) for v in inner.split(",")]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip('"\'')


def extract_tags(text: str, frontmatter: Dict) -> List[str]:
    tags: List[str] = []
    fm_tags = frontmatter.get("tags") or frontmatter.get("tag") or []
    if isinstance(fm_tags, str):
        tags.extend([t.strip() for t in fm_tags.split(",") if t.strip()])
    elif isinstance(fm_tags, list):
        tags.extend([str(t).strip() for t in fm_tags if str(t).strip()])
    tags.extend(TAG_RE.findall(text))
    seen = set()
    deduped = []
    for tag in tags:
        norm = tag.lstrip("#")
        if norm and norm not in seen:
            seen.add(norm)
            deduped.append(norm)
    return deduped


def extract_title(file_path: str) -> str:
    return os.path.splitext(os.path.basename(file_path))[0]


def build_heading_index(content: str) -> List[Tuple[int, str]]:
    headings: List[Tuple[int, str]] = []
    stack: List[Tuple[int, str]] = []
    char_index = 0
    for line in content.splitlines(keepends=True):
        match = HEADING_RE.match(line.strip())
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            while stack and stack[-1][0] >= level:  # released from previous header level
                stack.pop()
            stack.append((level, title))  # keep "digging in"
            path = " > ".join([item[1] for item in stack])
            headings.append((char_index, path))
        char_index += len(line)
    return headings


def heading_path_for_offset(heading_index: List[Tuple[int, str]], offset: int) -> str:
    last_path = ""
    for pos, path in heading_index:
        if pos > offset:
            break
        last_path = path
    return last_path


def chunk_text(content: str, chunk_size: int, chunk_overlap: int) -> List[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    heading_index = build_heading_index(content)
    chunks: List[Chunk] = []
    length = len(content)
    step = chunk_size - chunk_overlap
    start = 0
    index = 0  # chunk id field

    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            newline = content.rfind("\n", start, end)  # last occurence of "\n"
            if newline > start + 100:
                end = newline
        text = content[start:end].strip()
        if text:
            start_line = content.count("\n", 0, start) + 1
            end_line = content.count("\n", 0, end) + 1
            heading_path = heading_path_for_offset(heading_index, start)
            chunks.append(
                Chunk(
                    text=text,
                    start=start,
                    end=end,
                    start_line=start_line,
                    end_line=end_line,
                    index=index,
                    heading_path=heading_path,
                )
            )
            index += 1
        start += step
    return chunks


def metadata_value(value):
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=True)


def build_metadata(
    vault_path: str,
    file_path: str,
    title: str,
    frontmatter: Dict,
    fm_text: str,
    tags: List[str],
    chunk: Chunk,
    file_hash: str,
) -> Dict:
    rel_path = os.path.relpath(file_path, vault_path)
    stat = os.stat(file_path)
    return {
        "source": "obsidian",
        "vault_name": os.path.basename(vault_path.rstrip(os.sep)),
        "vault_path": os.path.abspath(vault_path),
        "file_abs_path": os.path.abspath(file_path),
        "file_rel_path": rel_path,
        "file_name": os.path.basename(file_path),
        "file_ext": os.path.splitext(file_path)[1].lower(),
        "file_title": title,
        "file_size": stat.st_size,
        "file_mtime": int(stat.st_mtime),
        "file_mtime_iso": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "file_hash": file_hash,
        "frontmatter": metadata_value(frontmatter),
        "frontmatter_raw": fm_text,
        "tags": ",".join(tags),
        "tag_count": len(tags),
        "chunk_index": chunk.index,
        "chunk_start": chunk.start,
        "chunk_end": chunk.end,
        "chunk_start_line": chunk.start_line,
        "chunk_end_line": chunk.end_line,
        "chunk_hash": sha256_text(chunk.text),
        "heading_path": chunk.heading_path,
        "content_length": len(chunk.text),
    }


def build_doc_id(rel_path: str, chunk: Chunk) -> str:
    base = f"{rel_path}:{chunk.index}:{chunk.start}:{chunk.end}"
    return hashlib.sha1(base.encode("utf-8", errors="replace")).hexdigest()


def add_documents(collection, documents, metadatas, ids, batch_size, dry_run: bool):
    if dry_run:
        return
    for i in range(0, len(documents), batch_size):
        collection.add(
            documents=documents[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
            ids=ids[i : i + batch_size],
        )


def build_index(args):
    client = get_chroma_client(args)
    embedding_kwargs = json.loads(args.embedding_kwargs) if args.embedding_kwargs else None
    embedding_function = get_embedding_function(args.embedding_function, embedding_kwargs)
    config = CreateCollectionConfiguration(embedding_function=embedding_function)
    collection = client.get_or_create_collection(
        name=args.collection,
        configuration=config,
    )

    if args.command == "build" and not args.dry_run:
        collection.delete(where={"source": "obsidian"})

    total_chunks = 0
    total_files = 0
    documents: List[str] = []
    metadatas: List[Dict] = []
    ids: List[str] = []
    
    # TODO: Different chunking strategy for jsonl files
    for file_path in iter_markdown_files(
        args.vault, args.include_ext, DEFAULT_EXCLUDE_DIRS | set(args.exclude_dir)
    ):
        total_files += 1
        content = read_text(file_path)
        frontmatter, body, fm_text = parse_frontmatter(content)
        tags = extract_tags(body, frontmatter)
        title = extract_title(file_path)
        file_hash = sha256_text(content)
        rel_path = os.path.relpath(file_path, args.vault)
        chunks = chunk_text(body, args.chunk_size, args.chunk_overlap)

        for chunk in chunks:
            documents.append(chunk.text)
            metadatas.append(
                build_metadata(
                    args.vault,
                    file_path,
                    title,
                    frontmatter,
                    fm_text,
                    tags,
                    chunk,
                    file_hash,
                )
            )
            ids.append(build_doc_id(rel_path, chunk))
            total_chunks += 1

        if len(documents) >= args.batch_size:
            add_documents(collection, documents, metadatas, ids, args.batch_size, args.dry_run)
            documents, metadatas, ids = [], [], []

    if documents:
        add_documents(collection, documents, metadatas, ids, args.batch_size, args.dry_run)

    print(
        f"Indexed {total_chunks} chunks from {total_files} files into collection '{args.collection}'."
    )


def collect_collection_file_index(collection, page_size: int = 500) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    offset = 0
    while True:
        result = collection.get(include=["metadatas", "ids"], limit=page_size, offset=offset)
        ids = result.get("ids") or []
        if not ids:
            break
        metadatas = result.get("metadatas") or []
        for doc_id, metadata in zip(ids, metadatas):
            file_rel_path = metadata.get("file_rel_path") if metadata else None
            if file_rel_path:
                mapping.setdefault(file_rel_path, []).append(doc_id)
        offset += len(ids)
    return mapping


def refresh_index(args):
    client = get_chroma_client(args)
    embedding_kwargs = json.loads(args.embedding_kwargs) if args.embedding_kwargs else None
    embedding_fn = get_embedding_function(args.embedding_function, embedding_kwargs)
    config = CreateCollectionConfiguration(embedding_function=embedding_fn)
    collection = client.get_or_create_collection(
        name=args.collection,
        configuration=config,
    )

    vault_files = list(
        iter_markdown_files(
            args.vault, args.include_ext, DEFAULT_EXCLUDE_DIRS | set(args.exclude_dir)
        )
    )
    vault_rel_paths = {os.path.relpath(path, args.vault) for path in vault_files}

    stale_files = 0
    added_files = 0
    total_chunks = 0

    for file_path in vault_files:
        content = read_text(file_path)
        frontmatter, body, fm_text = parse_frontmatter(content)
        tags = extract_tags(body, frontmatter)
        title = extract_title(body, frontmatter, file_path)
        file_hash = sha256_text(content)
        rel_path = os.path.relpath(file_path, args.vault)

        existing = collection.get(
            where={"file_rel_path": rel_path}, include=["metadatas", "ids"]
        )
        existing_ids = existing.get("ids") or []
        existing_metas = existing.get("metadatas") or []

        if not existing_ids:
            added_files += 1
            need_update = True
        else:
            file_hashes = {meta.get("file_hash") for meta in existing_metas if meta}
            need_update = len(file_hashes) != 1 or file_hash not in file_hashes

        if not need_update:
            continue

        stale_files += 1
        if not args.dry_run and existing_ids:
            collection.delete(where={"file_rel_path": rel_path})

        chunks = chunk_text(body, args.chunk_size, args.chunk_overlap)
        documents = []
        metadatas = []
        ids = []
        for chunk in chunks:
            documents.append(chunk.text)
            metadatas.append(
                build_metadata(
                    args.vault,
                    file_path,
                    title,
                    frontmatter,
                    fm_text,
                    tags,
                    chunk,
                    file_hash,
                )
            )
            ids.append(build_doc_id(rel_path, chunk))
            total_chunks += 1

        add_documents(collection, documents, metadatas, ids, args.batch_size, args.dry_run)

    removed_files = 0
    if args.delete_removed:
        index = collect_collection_file_index(collection)
        removed = [path for path in index.keys() if path not in vault_rel_paths]
        removed_files = len(removed)
        if removed and not args.dry_run:
            for rel_path in removed:
                collection.delete(where={"file_rel_path": rel_path})

    print(
        "Refresh complete: "
        f"{stale_files} stale, {added_files} new, {removed_files} removed, "
        f"{total_chunks} chunks updated."
    )


def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.dotenv_path:
        load_dotenv(dotenv_path=args.dotenv_path)
        # re-parse args to read the updated environment variables
        parser = create_parser()
        args = parser.parse_args()
        
    if args.command == "build":
        build_index(args)
    elif args.command == "refresh":
        refresh_index(args)


if __name__ == "__main__":
    main()

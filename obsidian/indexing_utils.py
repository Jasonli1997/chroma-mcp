from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

from obsidian.backlink_utils import (
    LinkGraphDiagnostics,
    backlinks_hash,
    build_reverse_link_index,
    filter_entry_files,
    rel_path_in_vault,
)

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
SOURCE = "obsidian"


@dataclass
class Chunk:
    text: str
    start: int
    end: int
    start_line: int
    end_line: int
    index: int
    heading_path: str


@dataclass
class ParsedFile:
    path: str
    rel_path: str
    title: str
    frontmatter: Dict
    tags: List[str]
    file_hash: str
    chunks: List[Chunk]


@dataclass
class VaultIndexState:
    vault_files: List[str]
    entry_files: List[str]
    entry_rel_paths: set[str]
    rel_to_abs_path: Dict[str, str]
    reverse_link_index: Dict[str, List[str]]
    link_diagnostics: LinkGraphDiagnostics


@dataclass
class EntryChunkContext:
    entry_file: ParsedFile
    backlinks: List[str]
    backlinks_digest: str
    backlink_content_digest: str


def iter_markdown_files(
    vault_path: str, include_exts: Iterable[str], exclude_dirs: Iterable[str]
):
    include_exts = {ext.lower() for ext in include_exts}
    exclude_dirs = {d.lower() for d in exclude_dirs}
    for root, dirs, files in os.walk(vault_path):
        dirs[:] = [
            d for d in dirs if d.lower() not in exclude_dirs and not d.startswith(".")
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


def parse_frontmatter(text: str) -> Tuple[Dict, str]:
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    if len(lines) < 2 or lines[0].strip() != "---":
        return {}, text
    end_idx = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end_idx = idx
            break
    if end_idx is None:
        return {}, text
    fm_lines = lines[1:end_idx]
    content = "\n".join(lines[end_idx + 1 :])
    return parse_simple_yaml(fm_lines), content


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
        return value.strip("\"'")


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
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
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
    index = 0

    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            newline = content.rfind("\n", start, end)
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
    tags: List[str],
    chunk: Chunk,
    file_hash: str,
    backlinks: List[str],
    backlinks_digest: str,
    backlink_content_digest: str,
    chunk_kind: str = "primary",
    chunk_source_path: str | None = None,
    chunk_source_title: str | None = None,
    chunk_source_hash: str | None = None,
) -> Dict:
    if chunk_source_path is None:
        chunk_source_path = file_path
    if chunk_source_title is None:
        chunk_source_title = title
    if chunk_source_hash is None:
        chunk_source_hash = file_hash

    rel_path = rel_path_in_vault(vault_path, file_path)
    chunk_source_rel_path = rel_path_in_vault(vault_path, chunk_source_path)
    stat = os.stat(file_path)
    return {
        "source": SOURCE,
        "vault_name": os.path.basename(vault_path.rstrip(os.sep)),
        "vault_path": os.path.abspath(vault_path),
        "file_rel_path": rel_path,
        "file_title": title,
        "file_size": stat.st_size,
        "file_mtime": int(stat.st_mtime),
        "file_mtime_iso": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "file_hash": file_hash,
        "frontmatter": metadata_value(frontmatter),
        "tags": ",".join(tags),
        "chunk_index": chunk.index,
        "chunk_start": chunk.start,
        "chunk_end": chunk.end,
        "chunk_start_line": chunk.start_line,
        "chunk_end_line": chunk.end_line,
        "chunk_hash": sha256_text(chunk.text),
        "heading_path": chunk.heading_path,
        "content_length": len(chunk.text),
        "backlink_count": len(backlinks),
        "backlinks": json.dumps(backlinks, ensure_ascii=True),
        "backlinks_hash": backlinks_digest,
        "backlink_content_hash": backlink_content_digest,
        "chunk_kind": chunk_kind,
        "chunk_source_file_rel_path": chunk_source_rel_path,
        "chunk_source_file_title": chunk_source_title,
        "chunk_source_file_hash": chunk_source_hash,
    }


def build_doc_id(rel_path: str, chunk: Chunk) -> str:
    base = f"{rel_path}:{chunk.index}:{chunk.start}:{chunk.end}"
    return hashlib.sha1(base.encode("utf-8", errors="replace")).hexdigest()


def build_backlink_doc_id(
    entry_rel_path: str, backlink_rel_path: str, chunk: Chunk
) -> str:
    base = (
        f"{entry_rel_path}:backlink:{backlink_rel_path}:"
        f"{chunk.index}:{chunk.start}:{chunk.end}"
    )
    return hashlib.sha1(base.encode("utf-8", errors="replace")).hexdigest()


def parse_file_for_chunking(
    vault_path: str,
    file_path: str,
    chunk_size: int,
    chunk_overlap: int,
    cache: Dict[str, ParsedFile],
) -> ParsedFile:
    cached = cache.get(file_path)
    if cached:
        return cached

    content = read_text(file_path)
    frontmatter, body = parse_frontmatter(content)
    parsed = ParsedFile(
        path=file_path,
        rel_path=rel_path_in_vault(vault_path, file_path),
        title=extract_title(file_path),
        frontmatter=frontmatter,
        tags=extract_tags(body, frontmatter),
        file_hash=sha256_text(content),
        chunks=chunk_text(body, chunk_size, chunk_overlap),
    )
    cache[file_path] = parsed
    return parsed


def compute_backlink_content_hash(
    backlinks: List[str],
    rel_to_abs_path: Dict[str, str],
    file_hash_cache: Dict[str, str],
) -> str:
    parts: List[str] = []
    for backlink_rel_path in backlinks:
        backlink_abs_path = rel_to_abs_path.get(backlink_rel_path)
        if not backlink_abs_path:
            continue
        if backlink_abs_path not in file_hash_cache:
            file_hash_cache[backlink_abs_path] = sha256_text(read_text(backlink_abs_path))
        parts.append(f"{backlink_rel_path}:{file_hash_cache[backlink_abs_path]}")
    parts.sort()
    return sha256_text("\n".join(parts))


def build_vault_index_state(args) -> VaultIndexState:
    vault_files = sorted(
        iter_markdown_files(
            args.vault, args.include_ext, DEFAULT_EXCLUDE_DIRS | set(args.exclude_dir)
        )
    )
    entry_files = filter_entry_files(args.vault, vault_files, args.entry_dir)
    entry_rel_paths = {rel_path_in_vault(args.vault, path) for path in entry_files}
    rel_to_abs_path = {rel_path_in_vault(args.vault, path): path for path in vault_files}
    # Each entry file has a one-to-many relationship with its backlinks
    reverse_link_index, link_diagnostics = build_reverse_link_index(
        args.vault,
        vault_files,
        read_text,
        parse_frontmatter,
        target_rel_paths=entry_rel_paths,
    )
    return VaultIndexState(
        vault_files=vault_files,
        entry_files=entry_files,
        entry_rel_paths=entry_rel_paths,
        rel_to_abs_path=rel_to_abs_path,
        reverse_link_index=reverse_link_index,
        link_diagnostics=link_diagnostics,
    )


def build_entry_chunk_context(
    args,
    file_path: str,
    state: VaultIndexState,
    chunk_file_cache: Dict[str, ParsedFile],
    file_hash_cache: Dict[str, str],
) -> EntryChunkContext:
    entry_file = parse_file_for_chunking(
        args.vault, file_path, args.chunk_size, args.chunk_overlap, chunk_file_cache
    )
    backlinks = state.reverse_link_index.get(entry_file.rel_path, [])
    backlinks_digest = backlinks_hash(backlinks)
    backlink_content_digest = compute_backlink_content_hash(
        backlinks, state.rel_to_abs_path, file_hash_cache
    )
    return EntryChunkContext(
        entry_file=entry_file,
        backlinks=backlinks,
        backlinks_digest=backlinks_digest,
        backlink_content_digest=backlink_content_digest,
    )


def build_entry_records(
    args,
    entry_ctx: EntryChunkContext,
    state: VaultIndexState,
    chunk_file_cache: Dict[str, ParsedFile],
) -> Tuple[List[str], List[Dict], List[str], int]:
    documents: List[str] = []
    metadatas: List[Dict] = []
    ids: List[str] = []
    total_chunks = 0

    entry_file = entry_ctx.entry_file
    for chunk in entry_file.chunks:
        documents.append(chunk.text)
        metadatas.append(
            build_metadata(
                args.vault,
                entry_file.path,
                entry_file.title,
                entry_file.frontmatter,
                entry_file.tags,
                chunk,
                entry_file.file_hash,
                entry_ctx.backlinks,
                entry_ctx.backlinks_digest,
                entry_ctx.backlink_content_digest,
                chunk_kind="primary",
            )
        )
        ids.append(build_doc_id(entry_file.rel_path, chunk))
        total_chunks += 1

    for backlink_rel_path in entry_ctx.backlinks:
        if backlink_rel_path == entry_file.rel_path:
            continue
        backlink_abs_path = state.rel_to_abs_path.get(backlink_rel_path)
        if not backlink_abs_path:
            continue

        backlink_file = parse_file_for_chunking(
            args.vault,
            backlink_abs_path,
            args.chunk_size,
            args.chunk_overlap,
            chunk_file_cache,
        )
        for chunk in backlink_file.chunks:
            if entry_file.rel_path not in chunk.text:
                continue
            documents.append(chunk.text)
            metadatas.append(
                build_metadata(
                    args.vault,
                    entry_file.path,
                    entry_file.title,
                    backlink_file.frontmatter,
                    backlink_file.tags,
                    chunk,
                    entry_file.file_hash,
                    entry_ctx.backlinks,
                    entry_ctx.backlinks_digest,
                    entry_ctx.backlink_content_digest,
                    chunk_kind="backlink",
                    chunk_source_path=backlink_file.path,
                    chunk_source_title=backlink_file.title,
                    chunk_source_hash=backlink_file.file_hash,
                )
            )
            ids.append(
                build_backlink_doc_id(
                    entry_file.rel_path, backlink_file.rel_path, chunk
                )
            )
            total_chunks += 1

    return documents, metadatas, ids, total_chunks


def add_documents(collection, documents, metadatas, ids, batch_size, dry_run: bool):
    if dry_run:
        return
    for i in range(0, len(documents), batch_size):
        collection.add(
            documents=documents[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
            ids=ids[i : i + batch_size],
        )


def collect_collection_file_index(
    collection, source: str, page_size: int = 500
) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    offset = 0
    while True:
        result = collection.get(
            where={"source": source},
            include=["metadatas"],
            limit=page_size,
            offset=offset,
        )
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

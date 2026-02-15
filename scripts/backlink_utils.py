from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Iterable
from typing import Callable

WIKILINK_RE = re.compile(r"\[\[([^\[\]]+)\]\]")


@dataclass
class LinkGraphDiagnostics:
    resolved_links: int = 0
    unresolved_links: int = 0
    ambiguous_links: int = 0


def normalize_rel_path(path: str) -> str:
    normalized = os.path.normpath(path).replace("\\", "/")
    if normalized == ".":
        return ""
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.strip("/")


def rel_path_in_vault(vault_path: str, file_path: str) -> str:
    return normalize_rel_path(os.path.relpath(file_path, vault_path))


def extract_wikilink_targets(text: str) -> list[str]:
    return [match.group(1).strip() for match in WIKILINK_RE.finditer(text)]


def canonical_note_key(raw_target: str) -> str:
    raw_target = raw_target.split("|", 1)[0]
    raw_target = raw_target.split("#", 1)[0]
    target = raw_target.strip().replace("\\", "/")
    if not target:
        return ""
    target = normalize_rel_path(target)
    if not target:
        return ""
    if target.lower().endswith(".md"):
        target = target[:-3]
    return target.lower()


def build_note_maps(
    vault_path: str, vault_files: list[str]
) -> tuple[dict[str, str], dict[str, list[str]]]:
    canonical_to_rel_path: dict[str, str] = {}
    basename_to_rel_paths: dict[str, list[str]] = {}
    for file_path in vault_files:
        rel_path = rel_path_in_vault(vault_path, file_path)
        canonical = canonical_note_key(rel_path)
        if not canonical:
            continue
        canonical_to_rel_path[canonical] = rel_path
        basename = canonical.rsplit("/", 1)[-1]
        basename_to_rel_paths.setdefault(basename, []).append(rel_path)
    return canonical_to_rel_path, basename_to_rel_paths


def resolve_wikilink_target(
    raw_target: str,
    canonical_to_rel_path: dict[str, str],
    basename_to_rel_paths: dict[str, list[str]],
) -> tuple[str | None, bool]:
    target_key = canonical_note_key(raw_target)
    if not target_key:
        return None, False

    if "/" in target_key:
        return canonical_to_rel_path.get(target_key), False

    candidates = basename_to_rel_paths.get(target_key, [])
    if len(candidates) == 1:
        return candidates[0], False
    if len(candidates) > 1:
        return None, True
    return None, False


def build_reverse_link_index(
    vault_path: str,
    vault_files: list[str],
    read_text: Callable[[str], str],
    parse_frontmatter: Callable[[str], tuple[dict, str, str]],
    target_rel_paths: Iterable[str] | None = None,
) -> tuple[dict[str, list[str]], LinkGraphDiagnostics]:
    target_filter = set(target_rel_paths or [])
    canonical_to_rel_path, basename_to_rel_paths = build_note_maps(
        vault_path, vault_files
    )
    backlinks_by_target: dict[str, set[str]] = {}
    if target_filter:
        backlinks_by_target = {target: set() for target in target_filter}

    diagnostics = LinkGraphDiagnostics()

    for source_file_path in vault_files:
        source_rel_path = rel_path_in_vault(vault_path, source_file_path)
        content = read_text(source_file_path)
        _, body, _ = parse_frontmatter(content)
        for raw_target in extract_wikilink_targets(body):
            target_rel_path, ambiguous = resolve_wikilink_target(
                raw_target, canonical_to_rel_path, basename_to_rel_paths
            )
            if target_rel_path:
                diagnostics.resolved_links += 1
                if target_filter and target_rel_path not in target_filter:
                    continue
                backlinks_by_target.setdefault(target_rel_path, set()).add(
                    source_rel_path
                )
            elif ambiguous:
                diagnostics.ambiguous_links += 1
            else:
                diagnostics.unresolved_links += 1

    sorted_backlinks_by_target: dict[str, list[str]] = {}
    for target_rel_path, source_rel_paths in backlinks_by_target.items():
        sorted_backlinks_by_target[target_rel_path] = sorted(source_rel_paths)
    return sorted_backlinks_by_target, diagnostics


def normalize_entry_dirs(entry_dirs: list[str]) -> list[str]:
    normalized_dirs = []
    seen = set()
    for entry_dir in entry_dirs:
        normalized = normalize_rel_path(entry_dir)
        if not normalized:
            continue
        normalized_lower = normalized.lower()
        if normalized_lower in seen:
            continue
        seen.add(normalized_lower)
        normalized_dirs.append(normalized_lower)
    return normalized_dirs


def filter_entry_files(
    vault_path: str, vault_files: list[str], entry_dirs: list[str]
) -> list[str]:
    normalized_entry_dirs = normalize_entry_dirs(entry_dirs)
    if not normalized_entry_dirs:
        return vault_files

    entry_files = []
    for file_path in vault_files:
        rel_path = rel_path_in_vault(vault_path, file_path)
        rel_path_lower = rel_path.lower()
        if any(
            rel_path_lower == entry_dir or rel_path_lower.startswith(f"{entry_dir}/")
            for entry_dir in normalized_entry_dirs
        ):
            entry_files.append(file_path)
    return entry_files


def backlinks_hash(backlinks: list[str]) -> str:
    return hashlib.sha256(
        "\n".join(backlinks).encode("utf-8", errors="replace")
    ).hexdigest()

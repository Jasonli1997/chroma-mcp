from dataclasses import dataclass

from obsidian.backlink_utils import LinkGraphDiagnostics


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
    frontmatter: dict
    tags: list[str]
    file_hash: str
    chunks: list[Chunk]


@dataclass
class VaultIndexState:
    vault_files: list[str]
    entry_files: list[str]
    entry_rel_paths: set[str]
    rel_to_abs_path: dict[str, str]
    reverse_link_index: dict[str, list[str]]
    link_diagnostics: LinkGraphDiagnostics


@dataclass
class EntryChunkContext:
    entry_file: ParsedFile
    backlinks: list[str]
    backlinks_digest: str
    backlink_content_digest: str

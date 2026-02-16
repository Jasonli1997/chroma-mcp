#!/usr/bin/env python3
"""
Build or refresh a Chroma vector index from an Obsidian vault.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict

from chromadb.api import EmbeddingFunction
from chromadb.api.collection_configuration import CreateCollectionConfiguration
from dotenv import load_dotenv
from tqdm import tqdm

from obsidian.indexing_utils import (
    SOURCE,
    add_documents,
    build_entry_chunk_context,
    build_entry_records,
    build_vault_index_state,
    collect_collection_file_index,
)
from src.chroma_mcp.server import get_chroma_client, mcp_known_embedding_functions


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
        default=os.getenv("CHROMA_SSL", "true").lower()
        in ["true", "yes", "1", "t", "y"],
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
        "--entry-dir",
        action="append",
        default=[],
        help="Vault-relative directories to index (repeatable). Defaults to entire vault.",
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
    subparsers.add_parser(
        "build", help="Build a full index by replacing all existing embeddings"
    )
    subparsers.add_parser(
        "refresh", help="Refresh stale, add missing and removed deleted embeddings"
    )
    return parser


def get_embedding_function(name: str, kwargs: Dict | None) -> EmbeddingFunction:
    cls = mcp_known_embedding_functions[name]
    if kwargs:
        return cls(**kwargs)
    return cls()


def print_backlink_diagnostics(link_diagnostics) -> None:
    print(
        "Backlink diagnostics:\n"
        f"resolved_links={link_diagnostics.resolved_links}, "
        f"ambiguous_links={link_diagnostics.ambiguous_links}, "
        f"unresolved_links={link_diagnostics.unresolved_links}"
    )


def build_index(args) -> None:
    client = get_chroma_client(args)
    embedding_kwargs = (
        json.loads(args.embedding_kwargs) if args.embedding_kwargs else None
    )
    embedding_function = get_embedding_function(
        args.embedding_function, embedding_kwargs
    )
    config = CreateCollectionConfiguration(embedding_function=embedding_function)
    collection = client.get_or_create_collection(
        name=args.collection,
        configuration=config,
    )

    if args.command == "build" and not args.dry_run:
        collection.delete(where={"source": SOURCE})

    state = build_vault_index_state(args)
    total_chunks = 0
    total_files = 0
    documents = []
    metadatas = []
    ids = []
    chunk_file_cache = {}
    file_hash_cache = {}

    for file_path in tqdm(state.entry_files):
        total_files += 1
        entry_ctx = build_entry_chunk_context(
            args, file_path, state, chunk_file_cache, file_hash_cache
        )
        (
            entry_documents,
            entry_metadatas,
            entry_ids,
            entry_total_chunks,
        ) = build_entry_records(args, entry_ctx, state, chunk_file_cache)
        documents.extend(entry_documents)
        metadatas.extend(entry_metadatas)
        ids.extend(entry_ids)
        total_chunks += entry_total_chunks

        if len(documents) >= args.batch_size:
            add_documents(
                collection, documents, metadatas, ids, args.batch_size, args.dry_run
            )
            documents, metadatas, ids = [], [], []

    if documents:
        add_documents(
            collection, documents, metadatas, ids, args.batch_size, args.dry_run
        )

    print(
        f"Indexed {total_chunks} chunks from {total_files} files into collection '{args.collection}'."
    )
    print_backlink_diagnostics(state.link_diagnostics)


def refresh_index(args) -> None:
    client = get_chroma_client(args)
    embedding_kwargs = (
        json.loads(args.embedding_kwargs) if args.embedding_kwargs else None
    )
    embedding_fn = get_embedding_function(args.embedding_function, embedding_kwargs)
    config = CreateCollectionConfiguration(embedding_function=embedding_fn)
    collection = client.get_or_create_collection(
        name=args.collection,
        configuration=config,
    )

    state = build_vault_index_state(args)
    stale_files, added_files = 0, 0
    total_chunks, deleted_chunks = 0, 0
    chunk_file_cache = {}
    file_hash_cache = {}

    for file_path in tqdm(state.entry_files):
        entry_ctx = build_entry_chunk_context(
            args, file_path, state, chunk_file_cache, file_hash_cache
        )
        entry_file = entry_ctx.entry_file

        existing = collection.get(
            where={"$and": [{"source": SOURCE}, {"file_rel_path": entry_file.rel_path}]},
            include=["metadatas"],
        )
        existing_ids = existing.get("ids") or []
        existing_metas = existing.get("metadatas") or []

        if not existing_ids:
            added_files += 1
            need_update = True
        else:
            file_hashes = {meta.get("file_hash") for meta in existing_metas if meta}
            backlinks_hashes = {
                meta.get("backlinks_hash") for meta in existing_metas if meta
            }
            backlink_content_hashes = {
                meta.get("backlink_content_hash") for meta in existing_metas if meta
            }
            need_update = (
                len(file_hashes) != 1
                or entry_file.file_hash not in file_hashes
                or len(backlinks_hashes) != 1
                or entry_ctx.backlinks_digest not in backlinks_hashes
                or len(backlink_content_hashes) != 1
                or entry_ctx.backlink_content_digest not in backlink_content_hashes
            )

        if not need_update:
            continue

        stale_files += 1
        if not args.dry_run and existing_ids:
            collection.delete(
                where={"$and": [{"source": SOURCE}, {"file_rel_path": entry_file.rel_path}]}
            )

        documents, metadatas, ids, entry_total_chunks = build_entry_records(
            args, entry_ctx, state, chunk_file_cache
        )
        total_chunks += entry_total_chunks

        add_documents(
            collection, documents, metadatas, ids, args.batch_size, args.dry_run
        )

    index = collect_collection_file_index(collection, source=SOURCE)
    removed = [
        (path, chunk_ids)
        for path, chunk_ids in index.items()
        if path not in state.entry_rel_paths
    ]
    removed_files = len(removed)
    if removed and not args.dry_run:
        for rel_path, chunk_ids in removed:
            collection.delete(
                where={"$and": [{"source": SOURCE}, {"file_rel_path": rel_path}]}
            )
            deleted_chunks += len(chunk_ids)

    print(
        "Refresh complete:\n"
        f"Files: {stale_files} stale, {added_files} new, {removed_files} removed\n"
        f"Chunks: {total_chunks} chunks added/updated, {deleted_chunks} chunks deleted."
    )
    print_backlink_diagnostics(state.link_diagnostics)


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    if args.dotenv_path:
        load_dotenv(dotenv_path=args.dotenv_path)
        parser = create_parser()
        args = parser.parse_args()

    if args.command == "build":
        build_index(args)
    elif args.command == "refresh":
        refresh_index(args)


if __name__ == "__main__":
    main()

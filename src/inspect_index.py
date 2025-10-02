#!/usr/bin/env python3
"""
Inspect your FAISS vector index:
- Count chunks
- List unique sources and page counts
- Preview the first N chunks (with optional token counts)

Usage:
  python -m src.inspect_index --top 5 --tokens
"""

import argparse
from collections import Counter, defaultdict

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENC = None  # tokens disabled unless tiktoken is installed

from src.embed_store import load_faiss
from src.splitters import load_config


def token_len(text: str) -> int:
    if _ENC is None:
        return -1
    return len(_ENC.encode(text))


def main():
    parser = argparse.ArgumentParser(description="Inspect FAISS index contents")
    parser.add_argument("--index", default=None, help="Path to index (default: config.yaml index_path)")
    parser.add_argument("--top", type=int, default=5, help="How many chunks to preview")
    parser.add_argument("--tokens", action="store_true", help="Show token counts (requires tiktoken)")
    args = parser.parse_args()

    cfg = load_config()
    index_path = args.index or cfg["index_path"]

    db = load_faiss(index_path)

    # Basic stats
    n_chunks = len(db.index_to_docstore_id)
    print(f"\nIndex: {index_path}")
    print(f"Total chunks: {n_chunks}")

    # Source and page summary
    source_counts = Counter()
    source_pages = defaultdict(set)

    for doc_id in db.index_to_docstore_id.values():
        d = db.docstore.search(doc_id)
        src = d.metadata.get("source", "unknown")
        pg = d.metadata.get("page", None)
        source_counts[src] += 1
        if pg is not None:
            source_pages[src].add(pg)

    print("\nSources (by chunk count):")
    for src, cnt in source_counts.most_common():
        pages = source_pages.get(src)
        pages_str = f"{len(pages)} pages" if pages else "pages: n/a"
        print(f"  • {src} — {cnt} chunks ({pages_str})")

    # Preview a few chunks
    print(f"\nPreview first {min(args.top, n_chunks)} chunk(s):")
    for i, doc_id in enumerate(db.index_to_docstore_id.values()):
        if i >= args.top:
            break
        d = db.docstore.search(doc_id)
        src = d.metadata.get("source", "unknown")
        pg = d.metadata.get("page", "n/a")
        content = d.page_content.strip().replace("\n", " ")
        preview = (content[:300] + "…") if len(content) > 300 else content
        if args.tokens and _ENC is not None:
            tlen = token_len(d.page_content)
            print(f"\n--- Chunk {i} | tokens={tlen} ---")
        else:
            print(f"\n--- Chunk {i} ---")
        print(f"Source: {src}")
        print(f"Page:   {pg}")
        print(f"Text:   {preview}")

    if args.tokens and _ENC is None:
        print("\n[warn] Token counts requested but `tiktoken` not installed. Install with:")
        print("       pip install tiktoken")

if __name__ == "__main__":
    main()
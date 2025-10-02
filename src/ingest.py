

from dotenv import load_dotenv
load_dotenv()
import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document
from src.splitters import make_text_splitter, load_config
from src.embed_store import build_faiss

RAW_DIR = Path("data/raw")

def load_docs():
    docs = []
    for p in RAW_DIR.glob("**/*"):
        if p.is_dir():
            continue
        if p.suffix.lower() in [".pdf"]:
            loader = PyPDFLoader(str(p))
            docs.extend(loader.load())
        elif p.suffix.lower() in [".txt", ".md"]:
            loader = TextLoader(str(p), encoding="utf-8")
            docs.extend(loader.load())
    return docs

def main():
    cfg = load_config()
    docs = load_docs()
    if not docs:
        print("No documents found in data/raw. Add PDFs or .txt files and re-run.")
        return
    splitter = make_text_splitter(cfg)
    chunks = splitter.split_documents(docs)
    print(f"Loaded {len(docs)} docs → {len(chunks)} chunks")
    build_faiss(chunks, cfg["index_path"])
    print(f"FAISS index saved to: {cfg['index_path']}")

if __name__ == "__main__":
    main()

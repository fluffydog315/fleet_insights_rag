# Fleet Insights Assistant (RAG)

A minimal Retrieval‑Augmented Generation (RAG) app you can extend for your fleet/NAICS work.

## What this project does
- **Ingests** your PDFs/text into chunks
- **Embeds** chunks and stores them in a local vector index (FAISS)
- **Retrieves** the top‑k relevant chunks for a user question
- **Generates** grounded answers with citations using an LLM

## Quickstart

### 0) Create a virtualenv and install
```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 1) Configure your API key
Copy `.env.example` to `.env` and set `OPENAI_API_KEY` (or swap to another embeddings/LLM in `src/embed_store.py` and `src/app.py`).

### 2) Add documents
Drop PDFs or text files into `data/raw/`. (You can start with any PDF—NAICS docs, business rules, etc.)

### 3) Build the vector index
```bash
python -m src.ingest
```
This will chunk, embed, and create a local FAISS index under `./vectorstore/`.

### 4) Run the RAG app (Streamlit)
```bash
streamlit run src/app.py
```
Open the URL Streamlit prints in your terminal.

## Project Structure
```
fleet_insights_rag/
├─ data/
│  ├─ raw/            # put your PDFs / .txt here
│  └─ processed/      # (optional) preprocessed text
├─ notebooks/
├─ src/
│  ├─ ingest.py       # load → split → embed → build FAISS
│  ├─ splitters.py    # chunking strategies
│  ├─ embed_store.py  # vector store build/load helpers
│  ├─ retriever.py    # create retriever abstraction
│  └─ app.py          # Streamlit RAG UI
├─ vectorstore/       # saved FAISS index
├─ .env.example
├─ requirements.txt
├─ config.yaml
└─ README.md
```

## Next steps / Upgrades (Phase 3+)
- Add **BM25+dense hybrid** search
- Add a **cross‑encoder reranker**
- Show **citations** inline and add **source filtering**
- Swap FAISS → **LanceDB** or a managed DB (Pinecone/Weaviate)
- Evaluate retrieval with small labeled Q/A pairs (recall@k)

import streamlit as st
from dotenv import load_dotenv
load_dotenv()  # load .env for OPENAI_API_KEY, PYTHONPATH etc.

from src.embed_store import load_faiss
from src.config_manager import get_config
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.schema import Document

# ---------------------------------------------------------
# Setup
# ---------------------------------------------------------
st.set_page_config(page_title="Fleet Insights RAG", layout="wide")
st.title("🚛 Fleet Insights RAG Demo")

# Load configuration
config = get_config()

# Show embedding mode
st.sidebar.markdown("### ⚙️ Configuration")
st.sidebar.info(f"**Embedding Mode:** {config.embedding_mode}")
st.sidebar.info(f"**Incremental:** {'✅ Enabled' if config.is_incremental_enabled() else '❌ Disabled'}")

# Load FAISS retriever
db = load_faiss()
retriever = db.as_retriever(search_kwargs={"k": config.get("top_k", 3)})

# LLM (uses OpenAI, but you can extend to HuggingFace fallback)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def base_answer(query: str):
    """LLM answers without retrieval."""
    return llm.invoke(f"Answer concisely:\n\nQ: {query}\nA:")

def helper_answer(query: str, retriever):
    """LLM answers first, then augments with optional retrieved context."""
    base = base_answer(query)

    docs: list[Document] = retriever.get_relevant_documents(query)
    usable_docs = [d for d in docs if len(d.page_content.strip()) > 50]

    if not usable_docs:
        return {"answer": base.content, "augment": "No additional context.", "sources": []}

    # Ask LLM to add context only if it helps
    context_blob = "\n\n".join([f"[{i+1}] {d.page_content}" for i, d in enumerate(usable_docs)])
    aug_prompt = f"""You already answered this question:

Question: {query}
Base answer: {base.content}

Now you are given optional context (may be redundant). Only add details that are consistent and helpful. 
If nothing adds value, return 'No additional context.'

Context:
{context_blob}
"""
    aug = llm.invoke(aug_prompt)

    return {"answer": base.content, "augment": aug.content, "sources": usable_docs}

def strict_rag_answer(query: str, retriever):
    """Classic RetrievalQA where LLM must ground answer in retrieved context."""
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    result = qa.invoke({"query": query})
    return {
        "answer": result["result"],
        "augment": None,
        "sources": result["source_documents"]
    }

# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
query = st.text_input("Ask a question about your documents:")

mode = st.radio(
    "Choose mode:",
    ["No-RAG", "Helper-RAG", "Strict-RAG"],
    index=1,  # default Helper
    horizontal=True
)

if query:
    with st.spinner("Thinking..."):
        if mode == "No-RAG":
            result = {"answer": base_answer(query).content, "augment": None, "sources": []}
        elif mode == "Helper-RAG":
            result = helper_answer(query, retriever)
        else:
            result = strict_rag_answer(query, retriever)

    st.subheader("Answer")
    st.write(result["answer"])

    if result.get("augment"):
        st.subheader("Additional Context")
        st.write(result["augment"])

    if result["sources"]:
        st.subheader("Sources")
        for i, doc in enumerate(result["sources"], 1):
            st.markdown(f"**{i}.** File: `{doc.metadata.get('source')}` | Page: {doc.metadata.get('page')}")
            st.caption(doc.page_content[:300] + "...")
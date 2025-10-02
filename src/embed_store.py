# src/embed_store.py
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

def build_faiss(docs, index_path: str):
    embeddings = OpenAIEmbeddings()
    vs = FAISS.from_documents(docs, embeddings)
    vs.save_local(index_path)
    return vs

def load_faiss(index_path: str):
    embeddings = OpenAIEmbeddings()
    # allow_dangerous_deserialization required for loading saved FAISS indexes
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
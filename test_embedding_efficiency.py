"""
Embedding Efficiency Benchmark

This script measures:
- Total embedding time
- Time per chunk
- Time per token
- Batch processing efficiency
- API call patterns
- Memory usage
"""

import time
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import tiktoken
from src.splitters import make_text_splitter, load_config

load_dotenv()

def count_tokens(text: str, model: str = "text-embedding-ada-002") -> int:
    """Count tokens using tiktoken"""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def benchmark_embedding_efficiency(verbose=True):
    """
    Benchmark the embedding efficiency of the current implementation.
    """
    results = {
        "total_docs": 0,
        "total_chunks": 0,
        "total_tokens": 0,
        "load_time": 0,
        "chunk_time": 0,
        "embed_time": 0,
        "total_time": 0,
        "chunks_per_second": 0,
        "tokens_per_second": 0,
    }
    
    start_total = time.time()
    
    # Load configuration
    cfg = load_config()
    if verbose:
        print(f"📋 Configuration:")
        print(f"   Chunk size: {cfg['chunk_size']}")
        print(f"   Chunk overlap: {cfg['chunk_overlap']}")
        print()
    
    # Step 1: Load documents
    if verbose:
        print("📂 Step 1: Loading documents...")
    start_load = time.time()
    
    raw_dir = Path("data/raw")
    docs = []
    for p in raw_dir.glob("**/*"):
        if p.is_dir():
            continue
        if p.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(p))
            docs.extend(loader.load())
        elif p.suffix.lower() in [".txt", ".md"]:
            loader = TextLoader(str(p), encoding="utf-8")
            docs.extend(loader.load())
    
    results["load_time"] = time.time() - start_load
    results["total_docs"] = len(docs)
    
    if verbose:
        print(f"   ✅ Loaded {len(docs)} documents in {results['load_time']:.2f}s")
        print()
    
    if not docs:
        print("❌ No documents found in data/raw. Add PDFs or .txt files to run benchmark.")
        return results
    
    # Step 2: Chunk documents
    if verbose:
        print("✂️  Step 2: Chunking documents...")
    start_chunk = time.time()
    
    splitter = make_text_splitter(cfg)
    chunks = splitter.split_documents(docs)
    
    results["chunk_time"] = time.time() - start_chunk
    results["total_chunks"] = len(chunks)
    
    # Count tokens
    for chunk in chunks:
        results["total_tokens"] += count_tokens(chunk.page_content)
    
    avg_tokens_per_chunk = results["total_tokens"] / results["total_chunks"]
    
    if verbose:
        print(f"   ✅ Created {len(chunks)} chunks in {results['chunk_time']:.2f}s")
        print(f"   📊 Total tokens: {results['total_tokens']:,}")
        print(f"   📊 Avg tokens/chunk: {avg_tokens_per_chunk:.1f}")
        print()
    
    # Step 3: Embed chunks (the key efficiency test)
    if verbose:
        print("🔢 Step 3: Embedding chunks...")
        print("   (This step calls OpenAI API and may take a while)")
    
    start_embed = time.time()
    
    embeddings = OpenAIEmbeddings()
    
    # Test batch embedding
    try:
        vs = FAISS.from_documents(chunks, embeddings)
        results["embed_time"] = time.time() - start_embed
        
        if verbose:
            print(f"   ✅ Embedded {len(chunks)} chunks in {results['embed_time']:.2f}s")
            print()
    except Exception as e:
        print(f"   ❌ Error during embedding: {e}")
        return results
    
    # Calculate metrics
    results["total_time"] = time.time() - start_total
    if results["embed_time"] > 0:
        results["chunks_per_second"] = results["total_chunks"] / results["embed_time"]
        results["tokens_per_second"] = results["total_tokens"] / results["embed_time"]
    
    # Display summary
    if verbose:
        print("=" * 70)
        print("📊 EFFICIENCY SUMMARY")
        print("=" * 70)
        print(f"Total documents:        {results['total_docs']}")
        print(f"Total chunks:           {results['total_chunks']}")
        print(f"Total tokens:           {results['total_tokens']:,}")
        print(f"Avg tokens/chunk:       {avg_tokens_per_chunk:.1f}")
        print()
        print(f"Load time:              {results['load_time']:.2f}s")
        print(f"Chunk time:             {results['chunk_time']:.2f}s")
        print(f"Embed time:             {results['embed_time']:.2f}s")
        print(f"Total time:             {results['total_time']:.2f}s")
        print()
        print(f"⚡ Chunks/second:        {results['chunks_per_second']:.2f}")
        print(f"⚡ Tokens/second:        {results['tokens_per_second']:.1f}")
        print(f"⚡ Avg time/chunk:       {results['embed_time']/results['total_chunks']:.3f}s")
        print("=" * 70)
        print()
        
        # Cost estimate (OpenAI text-embedding-ada-002 pricing)
        cost_per_1k_tokens = 0.0001  # $0.0001 per 1K tokens
        estimated_cost = (results["total_tokens"] / 1000) * cost_per_1k_tokens
        print(f"💰 Estimated cost:      ${estimated_cost:.4f}")
        print()
        
        # Efficiency analysis
        print("🔍 EFFICIENCY ANALYSIS:")
        print()
        
        # Batch size analysis
        print("1. Current Implementation:")
        print("   - Uses LangChain's FAISS.from_documents()")
        print("   - This internally batches embeddings to the API")
        print("   - OpenAI supports up to 2048 texts per batch")
        print()
        
        # Bottleneck identification
        embed_pct = (results["embed_time"] / results["total_time"]) * 100
        print(f"2. Time Breakdown:")
        print(f"   - Loading: {(results['load_time']/results['total_time'])*100:.1f}%")
        print(f"   - Chunking: {(results['chunk_time']/results['total_time'])*100:.1f}%")
        print(f"   - Embedding: {embed_pct:.1f}%")
        print()
        
        if embed_pct > 80:
            print("   ⚠️  Embedding is the bottleneck (>80% of time)")
        elif embed_pct > 60:
            print("   ⚡ Embedding is significant but reasonable")
        else:
            print("   ✅ Embedding is well-optimized")
        print()
        
        # Recommendations
        print("💡 RECOMMENDATIONS:")
        print()
        
        if results['chunks_per_second'] < 5:
            print("   ⚠️  Low throughput detected!")
            print("   - Consider using local embeddings (sentence-transformers)")
            print("   - Or cache embeddings for unchanged documents")
        
        if avg_tokens_per_chunk > 1000:
            print("   ⚠️  Large chunk sizes detected!")
            print("   - Consider reducing chunk_size in config.yaml")
            print("   - Smaller chunks = faster embedding + lower costs")
        
        print("   General improvements:")
        print("   ✓ Use incremental indexing (only embed new docs)")
        print("   ✓ Implement document fingerprinting to skip unchanged files")
        print("   ✓ Consider local models (e.g., all-MiniLM-L6-v2) for development")
        print("   ✓ Use async embedding for better API utilization")
        print()
    
    return results


def compare_embedding_models():
    """
    Quick comparison of embedding model options
    """
    print("=" * 70)
    print("🔬 EMBEDDING MODEL COMPARISON")
    print("=" * 70)
    print()
    print("Option 1: OpenAI text-embedding-ada-002 (Current)")
    print("   Pros: High quality, 1536 dimensions")
    print("   Cons: API costs, network latency")
    print("   Speed: ~10-50 chunks/second (network dependent)")
    print("   Cost: $0.0001 per 1K tokens")
    print()
    print("Option 2: Local sentence-transformers (e.g., all-MiniLM-L6-v2)")
    print("   Pros: Free, no network latency, privacy")
    print("   Cons: Slightly lower quality, needs GPU for speed")
    print("   Speed: ~100+ chunks/second (GPU), ~10-20 (CPU)")
    print("   Cost: $0 (one-time download)")
    print()
    print("Option 3: Hybrid approach")
    print("   - Use local models for development/testing")
    print("   - Use OpenAI for production")
    print("   - Cache embeddings to avoid re-embedding")
    print()
    print("=" * 70)


if __name__ == "__main__":
    print("\n🚀 Fleet Insights RAG - Embedding Efficiency Benchmark\n")
    
    # Run the main benchmark
    results = benchmark_embedding_efficiency(verbose=True)
    
    # Show model comparison
    if results["total_chunks"] > 0:
        print()
        compare_embedding_models()

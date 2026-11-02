## Overview

A fully local Retrieval-Augmented Generation (RAG) chatbot that answers questions from a small corpus of research papers. It runs entirely on a laptop with no cloud APIs, and was designed to keep its disk and memory footprint small (about 1.5 GB for models and packages combined).

**Corpus:** 5 papers (Attention Is All You Need, BERT, RAG, ResNet, Adam), 77 pages in total.

## Architecture

```
PDF/TXT/MD  ->  Load  ->  Clean  ->  Chunk  ->  Dedup  ->  Embed (all-minilm)  ->  FAISS index
                                                                                      |
User question -> Vector search (top 12) + BM25 keyword search (top 12)                |
              -> Reciprocal Rank Fusion -> top 4 chunks -> Distance gate              |
              -> Prompt (context only) -> qwen2.5:1.5b via Ollama -> Answer + sources
```

| Component | Choice |
|---|---|
| Generator LLM | `qwen2.5:1.5b` (Ollama, ~1 GB, temperature 0, context window 2048) |
| Embedding model | `all-minilm` (Ollama, ~45 MB) |
| Vector store | FAISS (`faiss-cpu`), persisted to `storage/faiss` |
| Keyword retrieval | BM25 (`rank-bm25`) |
| Orchestration | LangChain core, community and Ollama integrations |
| Configuration | `config.yaml` (models, chunking, retrieval and grounding settings) |

## Features by requirement

### 1. Advanced retrieval
- **Hybrid search:** dense vector similarity and BM25 keyword matching, merged with Reciprocal Rank Fusion (RRF). Vector search catches paraphrases, and BM25 catches exact terms such as model names and acronyms.
- **Candidate pool:** 12 candidates from each retriever, fused down to the top 4 chunks sent to the LLM.
- Hybrid mode is a config toggle (`retrieval.hybrid`), so vector-only and hybrid can be compared.

### 2. Vector database metrics
- FAISS index with L2-normalized embeddings, so distance reflects cosine similarity.
- Benchmark plan (`benchmark.py`, see Results): Recall@k, MRR, p50/p95 query latency, index build time and index size, comparing FAISS Flat, HNSW and IVF indexes.

### 3. Ingestion pipeline and cleanliness
- Loaders for PDF, TXT and Markdown, with `source` and `page` metadata on every chunk.
- **Cleaning:** Unicode normalization (NFKC), repair of hyphenated line breaks, removal of bare page numbers, whitespace collapsing.
- **Chunking:** recursive character splitter, 600 characters with 90 overlap.
- **Deduplication:** SHA-1 content hash per chunk removes exact duplicates, within a file and across files.
- **Quality filter:** chunks under 30 characters are dropped.
- **Incremental, idempotent ingestion:** each file is hashed and recorded in a manifest. Unchanged files are skipped, and a changed file has its old chunks removed before re-indexing.
- **Batch embedding** (64 chunks per batch) and an **ingestion report** at the end of each run.

**Ingestion report (first run):**

| Metric | Value |
|---|---|
| Files indexed | 5 |
| Pages | 77 |
| Chunks stored | 556 |
| Duplicates removed | 1 |
| Empty chunks dropped | 1 |

### 4. Grounding and control
- **Context-only prompt:** the model is told to answer only from the retrieved context and to reply "I don't know based on the provided documents." otherwise.
- **Distance gate:** if the best vector match is farther than `max_distance` (currently 1.1), the system refuses without calling the LLM, which saves compute and prevents unsupported answers.
- **Citations:** every answer lists the source file and page of the chunks used.
- **Prompt-injection guard:** retrieved text is wrapped in `<context>` delimiters and explicitly declared to be data, not instructions.
- **Determinism:** temperature 0.

### 5. Architectural efficiency
- Models total about 1 GB, and the Python environment avoids PyTorch, ChromaDB and sentence-transformers, which keeps it small.
- Persistent index: embeddings are computed once and loaded from disk on later runs.
- Incremental ingestion avoids re-embedding unchanged files.
- Limited context window (`num_ctx=2048`) to cap memory use, and token streaming for fast first output.
- Modular code (`common.py`, `ingest.py`, `rag.py`) driven by a single `config.yaml`.

## Setup and usage

```powershell
ollama pull qwen2.5:1.5b
ollama pull all-minilm

python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt

# put PDF/TXT/MD files in data/raw, then:
python -m src.ingest     # build or update the index
python chat.py           # start chatting
```

## Project structure

```
rag-chatbot/
  config.yaml
  chat.py
  requirements.txt
  data/raw/            source documents (not committed)
  src/
    common.py          config and vector store loading
    ingest.py          load, clean, chunk, dedup, embed, index
    rag.py             hybrid retrieval, grounding, generation
  storage/             generated FAISS index and manifest (not committed)
```

## Example behavior

| Query | Result |
|---|---|
| "Whats in paper, Attention is all you need?" | Grounded answer describing self-attention, cited to `attention_is_all_you_need.pdf` (pp. 2-3) and `rag_paper.pdf` (p. 15) |
| "Hey I am Tehami" | Refused: "I don't know based on the provided documents." (no relevant context, so the gate blocks it) |

## Observations and limitations

- The out-of-scope refusal works: a message unrelated to the corpus is rejected before generation.
- Broad questions ("what is in the paper") give narrower answers than specific ones, because a 1.5B model summarizes only 4 chunks. Specific questions work better.
- Hybrid retrieval occasionally surfaces a chunk from a related paper (one citation came from `rag_paper.pdf` for an Attention question). Measuring this is the purpose of the benchmark ablation.
- The distance threshold needs tuning per corpus and embedding model.
- No cross-encoder reranker, to avoid the PyTorch dependency and keep disk usage low.

## Results 
## Results

Evaluated on 19 in-scope and 6 out-of-scope questions over 556 chunks (5 papers). A retrieved chunk counts as relevant if it comes from the expected paper and contains a labeled keyword. Query embedding time is excluded from the search latencies below (p50 15.9 ms, p95 45.8 ms via Ollama).

| Configuration | Recall@4 | MRR@10 | p50 latency (ms) | p95 latency (ms) |
|---|---|---|---|---|
| Vector only | 0.84 | 0.732 | 0.07 | 0.08 |
| Hybrid (vector + BM25, RRF) | 0.89 | 0.808 | 1.14 | 1.66 |

| FAISS index | Build time (ms) | Index size (KB) | Recall vs exact | p50 latency (ms) | p95 latency (ms) |
|---|---|---|---|---|---|
| Flat (exact) | 0.1 | 834 | 1.00 | 0.038 | 0.042 |
| HNSW (M=32, efSearch=64) | 59.1 | 981 | 1.00 | 0.087 | 0.103 |
| IVF (nlist=12, nprobe=3) | 5.6 | 857 | 0.87 | 0.028 | 0.034 |

| Grounding and ingestion test | Result |
|---|---|
| In-scope questions passing the gate (max_distance = 1.25) | 19/19 |
| Out-of-scope questions refused | 6/6 |
| Second ingestion run | 5/5 files skipped, 0 chunks re-embedded |

Best-match distance: in-scope 0.51 to 1.08, out-of-scope 1.51 to 1.69, so the two groups do not overlap. The threshold of 1.25 sits between them.

**Findings**
- Hybrid search raised Recall@4 from 0.84 to 0.89 and MRR from 0.732 to 0.808, at a cost of about 1 ms extra per query for BM25 scoring. The gain is one additional question out of 19, so it is a small improvement on a small test set rather than a proven one.
- At 556 vectors, exact (Flat) search is already the fastest accurate option: it has the smallest index and the shortest build time, with perfect recall by definition.
- HNSW matched Flat's results exactly (recall 1.00) but was slower to build (59 ms vs 0.1 ms), about 18% larger, and slower per query. It offers no benefit at this scale.
- IVF was fastest per query (0.028 ms) but lost 13% of the exact results (recall 0.87), which is a poor trade when Flat is already under 0.05 ms.
- Approximate indexes are designed for corpora with hundreds of thousands to millions of vectors, so this benchmark supports using Flat for this project.
- The distance gate separated in-scope from out-of-scope questions perfectly on this set.

**Limitations of the evaluation**
- 25 questions is a small sample, so differences of one question (such as the hybrid gain) are within noise.
- Relevance is judged by keyword matching, which is lenient and does not measure answer quality.
- In-scope questions were written using the papers' own vocabulary, so vaguer real-world queries may sit closer to the threshold.
- The grounding result measures the retrieval gate only, not whether generated answers are correct.

## Future work
- Labeled evaluation set and automated benchmark script.
- Lightweight reranking step and maximal marginal relevance (MMR) toggle.
- Embedding and query caching.
- Streamlit interface.
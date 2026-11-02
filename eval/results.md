Questions: 19 in-scope, 6 out-of-scope. Chunks indexed: 556.

Query embedding time (Ollama, all-minilm): p50 15.9 ms, p95 45.8 ms (not included in the latencies below)

| Configuration | Recall@4 | MRR@10 | p50 latency (ms) | p95 latency (ms) |
|---|---|---|---|---|
| Vector only | 0.84 | 0.732 | 0.07 | 0.08 |
| Hybrid (vector + BM25, RRF) | 0.89 | 0.808 | 1.14 | 1.66 |

| FAISS index | Build time (ms) | Index size (KB) | Recall vs exact | p50 latency (ms) | p95 latency (ms) |
|---|---|---|---|---|---|
| Flat (exact) | 0.1 | 834 | 1.00 | 0.038 | 0.042 |
| HNSW (M=32, efSearch=64) | 59.1 | 981 | 1.00 | 0.087 | 0.103 |
| IVF (nlist=12, nprobe=3) | 5.6 | 857 | 0.87 | 0.028 | 0.034 |

| Grounding test (max_distance = 1.25) | Result |
|---|---|
| In-scope questions answered | 19/19 |
| Out-of-scope questions refused | 6/6 |

Best-match distance, in-scope: min 0.51, max 1.08
Best-match distance, out-of-scope: min 1.51, max 1.69
A threshold between 1.08 and 1.51 separates them perfectly (midpoint 1.29).
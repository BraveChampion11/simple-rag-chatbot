import json
import re
import time
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from src.common import load_config, get_embeddings, load_store
from src.rag import tok, rrf

K = 4          # Recall@K
TOP = 10       # ranked list length used for MRR
REPEATS = 20   # repeats per query for stable latency


def norm(t):
    return re.sub(r"\s+", " ", t.lower())


def is_relevant(doc, q):
    if doc.metadata.get("source") != q["source"]:
        return False
    text = norm(doc.page_content)
    return any(k in text for k in q["keywords"])


def pct(vals, p):
    return float(np.percentile(vals, p))


def main():
    cfg = load_config()
    r = cfg["retrieval"]
    vs = load_store(cfg)
    if vs is None:
        raise SystemExit("No index found. Run: python -m src.ingest")
    emb = get_embeddings(cfg)

    questions = json.loads(Path("eval/questions.json").read_text(encoding="utf-8"))
    inq = [q for q in questions if q["in_scope"]]
    outq = [q for q in questions if not q["in_scope"]]

    docs = list(vs.docstore._dict.values())
    bm25 = BM25Okapi([tok(d.page_content) for d in docs])

    # Label sanity check: every in-scope question needs at least one matching chunk
    for q in inq:
        if not any(is_relevant(d, q) for d in docs):
            print(f"WARNING: no chunk matches the labels for: {q['question']}")

    # Embed every question once (timed separately from search)
    qvecs, emb_ms = {}, []
    for q in questions:
        t = time.perf_counter()
        qvecs[q["question"]] = emb.embed_query(q["question"])
        emb_ms.append((time.perf_counter() - t) * 1000)

    def vector_search(vec, k):
        return vs.similarity_search_with_score_by_vector(vec, k=k)

    def vector_only(query, vec):
        return [d for d, _ in vector_search(vec, TOP)]

    def hybrid(query, vec):
        vec_docs = [d for d, _ in vector_search(vec, r["fetch_k"])]
        scores = bm25.get_scores(tok(query))
        top = np.argsort(scores)[::-1][:r["fetch_k"]]
        bm_docs = [docs[i] for i in top if scores[i] > 0]
        return rrf([vec_docs, bm_docs])[:TOP]

    def evaluate(fn):
        hits, rr, lat = 0, [], []
        for q in inq:
            vec = qvecs[q["question"]]
            for _ in range(REPEATS):
                t = time.perf_counter()
                ranked = fn(q["question"], vec)
                lat.append((time.perf_counter() - t) * 1000)
            rel = [is_relevant(d, q) for d in ranked]
            hits += any(rel[:K])
            first = next((i for i, x in enumerate(rel) if x), None)
            rr.append(1 / (first + 1) if first is not None else 0.0)
        return hits / len(inq), float(np.mean(rr)), pct(lat, 50), pct(lat, 95)

    out = []
    out.append(f"Questions: {len(inq)} in-scope, {len(outq)} out-of-scope. "
               f"Chunks indexed: {len(docs)}.\n")
    out.append(f"Query embedding time (Ollama, all-minilm): p50 {pct(emb_ms, 50):.1f} ms, "
               f"p95 {pct(emb_ms, 95):.1f} ms (not included in the latencies below)\n")

    # ---- Table 1: retrieval quality ----
    out.append(f"| Configuration | Recall@{K} | MRR@{TOP} | p50 latency (ms) | p95 latency (ms) |")
    out.append("|---|---|---|---|---|")
    for name, fn in [("Vector only", vector_only), ("Hybrid (vector + BM25, RRF)", hybrid)]:
        rec, mrr, p50, p95 = evaluate(fn)
        out.append(f"| {name} | {rec:.2f} | {mrr:.3f} | {p50:.2f} | {p95:.2f} |")
    out.append("")

    # ---- Table 2: FAISS index comparison ----
    n = vs.index.ntotal
    xb = vs.index.reconstruct_n(0, n).astype("float32")
    d = xb.shape[1]
    xq = np.array([qvecs[q["question"]] for q in inq], dtype="float32")
    faiss.normalize_L2(xq)

    def build_flat():
        idx = faiss.IndexFlatL2(d)
        idx.add(xb)
        return idx

    def build_hnsw():
        idx = faiss.IndexHNSWFlat(d, 32)
        idx.hnsw.efConstruction = 100
        idx.add(xb)
        idx.hnsw.efSearch = 64
        return idx

    def build_ivf():
        idx = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, 12)
        idx.train(xb)
        idx.add(xb)
        idx.nprobe = 3
        return idx

    flat = build_flat()
    _, gt = flat.search(xq, TOP)   # exact ground truth

    out.append("| FAISS index | Build time (ms) | Index size (KB) | Recall vs exact | p50 latency (ms) | p95 latency (ms) |")
    out.append("|---|---|---|---|---|---|")
    for name, build in [("Flat (exact)", build_flat), ("HNSW (M=32, efSearch=64)", build_hnsw),
                        ("IVF (nlist=12, nprobe=3)", build_ivf)]:
        builds = []
        for _ in range(5):
            t = time.perf_counter()
            idx = build()
            builds.append((time.perf_counter() - t) * 1000)
        size_kb = faiss.serialize_index(idx).nbytes / 1024
        _, found = idx.search(xq, TOP)
        recall = float(np.mean([len(set(found[i]) & set(gt[i])) / TOP for i in range(len(xq))]))
        lat = []
        for i in range(len(xq)):
            for _ in range(REPEATS):
                t = time.perf_counter()
                idx.search(xq[i:i + 1], TOP)
                lat.append((time.perf_counter() - t) * 1000)
        out.append(f"| {name} | {pct(builds, 50):.1f} | {size_kb:.0f} | {recall:.2f} | "
                   f"{pct(lat, 50):.3f} | {pct(lat, 95):.3f} |")
    out.append("")

    # ---- Table 3: grounding gate ----
    thr = r["max_distance"]
    in_d = [vector_search(qvecs[q["question"]], 1)[0][1] for q in inq]
    out_d = [vector_search(qvecs[q["question"]], 1)[0][1] for q in outq]
    answered = sum(x <= thr for x in in_d)
    refused = sum(x > thr for x in out_d)
    out.append(f"| Grounding test (max_distance = {thr}) | Result |")
    out.append("|---|---|")
    out.append(f"| In-scope questions answered | {answered}/{len(inq)} |")
    out.append(f"| Out-of-scope questions refused | {refused}/{len(outq)} |")
    out.append("")
    out.append(f"Best-match distance, in-scope: min {min(in_d):.2f}, max {max(in_d):.2f}")
    out.append(f"Best-match distance, out-of-scope: min {min(out_d):.2f}, max {max(out_d):.2f}")
    if max(in_d) < min(out_d):
        out.append(f"A threshold between {max(in_d):.2f} and {min(out_d):.2f} separates them perfectly "
                   f"(midpoint {(max(in_d) + min(out_d)) / 2:.2f}).")
    else:
        out.append("The two ranges overlap, so no single threshold separates them perfectly.")

    text = "\n".join(out)
    print("\n" + text)
    Path("eval/results.md").write_text(text, encoding="utf-8")
    print("\nSaved to eval/results.md")


if __name__ == "__main__":
    main()
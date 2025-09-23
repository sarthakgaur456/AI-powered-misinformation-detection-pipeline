import argparse
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

INDEX_PATH = "models/evidence_faiss.index"
META_PATH = "evidence_chunks.parquet"
MODEL_NAME = "all-MiniLM-L6-v2" 
DEFAULT_K = 5

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--query', type = str, required= True, help = 'Claim or query text to search')
    p.add_argument('--k', type = int, default= DEFAULT_K, help = 'Top-k results')
    p.add_argument("--rerank", action="store_true", help="Rerank results with a cross-encoder")
    return p.parse_args()

def main():
    args = parse_args()

    print('Loading FAISS index:', INDEX_PATH)
    index = faiss.read_index(INDEX_PATH)

    print('Loading metadata:', META_PATH)
    meta = pd.read_parquet(META_PATH)
    meta = meta.reset_index(drop = True)

    print('Loading sentence transformer model:', MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    q = args.query
    print("\nQuery:", q)
    q_emb = model.encode([q], convert_to_numpy = True, normalize_embeddings = True).astype('float32')

    print('Searching index for top', args.k, 'results...')
    D, I = index.search(q_emb, args.k)
    scores = D[0]
    indices = I[0]

    results = []
    for idx, score in zip(indices, scores):
        if idx < 0:
            continue
        row = meta.iloc[idx]
        url = row.get("url", "")
        label = row.get("label_multi", row.get("ruling", row.get("verdict", "")))
        snippet = row.get("chunk_text", row.get("article", ""))[:500].replace("\n", " ")
        results.append({
            "idx": idx,
            "orig_score": float(score),
            "url": url,
            "label": label,
            "snippet": snippet
        })

    #reranking
    if args.rerank and len(results) > 0:
        from sentence_transformers import CrossEncoder
        cross = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [[q, r["snippet"]] for r in results]
        rerank_scores = cross.predict(pairs)
        for r, s in zip(results, rerank_scores):
            r["rerank_score"] = float(s)
        # sorting by rerank_score instead of orig_score
        results = sorted(results, key=lambda r: r["rerank_score"], reverse=True)
    else:
        # sorting by original FAISS score
        results = sorted(results, key=lambda r: r["orig_score"], reverse=True)

    # printing top results
    print("\nTop results:")
    for rank, r in enumerate(results, start=1):
        print(f"\n#{rank} idx={r['idx']}  orig_score={r['orig_score']:.4f}  rerank_score={r.get('rerank_score', None)}")
        print("url:", r['url'])
        print("label:", r['label'])
        print("snippet:", r['snippet'])

if __name__ == "__main__":
    main()


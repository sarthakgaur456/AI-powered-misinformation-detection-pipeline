import argparse
import os
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
from collections import Counter

INDEX_PATH = "models/evidence_faiss.index"
META_PATH = "evidence_chunks.parquet"
EMB_MODEL = "all-MiniLM-L6-v2"
VERIFIER_DIR = "models/verifier_crossencoder"

# thresholds to map regression score to its corresponding label
def score_to_label(score, t_refute=-0.25, t_support=0.25):
    if score <= t_refute:
        return "refutes"
    if score >= t_support:
        return "supports"
    return "uncertain"

# weight normalization
def softmax(x):
    x = np.array(x, dtype=float)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (ex.sum() + 1e-12)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--query", required=True, help="Claim to check")
    p.add_argument("--k", type=int, default=20, help="How many candidates to retrieve from FAISS")
    p.add_argument("--verify_k", type=int, default=10, help="How many top candidates to send to verifier")
    return p.parse_args()

def main():
    args = parse_args()

    # checks
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}")
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Metadata parquet not found at {META_PATH}")
    if not os.path.exists(VERIFIER_DIR):
        raise FileNotFoundError(f"Verifier model not found at {VERIFIER_DIR}. Train and save it first.")

    # load index and metadata
    index = faiss.read_index(INDEX_PATH)
    meta = pd.read_parquet(META_PATH).reset_index(drop=True)

    # embed the query
    embedder = SentenceTransformer(EMB_MODEL)
    q_emb = embedder.encode([args.query], convert_to_numpy=True, normalize_embeddings=True).astype('float32')

    # retrieve
    D, I = index.search(q_emb, args.k)
    orig_scores = D[0].tolist()
    indices = I[0].tolist()

    candidates = []
    for idx, s in zip(indices, orig_scores):
        if idx < 0:
            continue
        row = meta.iloc[idx]
        snippet = row.get("chunk_text") or row.get("article") or ""
        url = row.get("url", "")
        candidates.append({"idx": int(idx), "orig_score": float(s), "snippet": snippet, "url": url})

    if len(candidates) == 0:
        print("No candidates retrieved.")
        return

    # pick top-N to verify
    verify_k = min(args.verify_k, len(candidates))
    top_candidates = candidates[:verify_k]

    # load verifier
    verifier = CrossEncoder(VERIFIER_DIR)

    # prepare pairs and predict scores
    pairs = [[args.query, c['snippet']] for c in top_candidates]
    scores = verifier.predict(pairs).squeeze() 
    # attach predictions and confidences
    for i, s in enumerate(scores):
        top_candidates[i]['pred_score'] = float(s)
        top_candidates[i]['pred_label'] = score_to_label(s)
        top_candidates[i]['raw_conf'] = abs(float(s))

    # compute normalized weights via softmax on raw_conf
    confs = [c['raw_conf'] for c in top_candidates]
    weights = softmax(confs)
    for c, w in zip(top_candidates, weights):
        c['weight'] = float(w)

    # aggregate weighted votes
    votes = Counter()
    for c in top_candidates:
        votes[c['pred_label']] += c['weight']

    final_label, final_weight = votes.most_common(1)[0]

    # print results
    print("\nFinal Verdict")
    print(f"Claim: {args.query}")
    print(f"Final label: {final_label}    (weight: {final_weight:.4f})")
    print("Vote breakdown:", dict(votes))
    print("\n")

    print("Top evidence (from verifier):")
    for i, c in enumerate(top_candidates, start=1):
        print(f"\n#{i} pred={c['pred_label']} score={c['pred_score']:.3f} weight={c['weight']:.3f} url={c['url']}")
        print(c['snippet'][:600].replace("\n", " "))

if __name__ == "__main__":
    main()

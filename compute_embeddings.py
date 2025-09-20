import os
import numpy as np
import pandas as pd

CLAIMS_CSV = 'claims_dataset.csv'
CHUNKS_PARQUET = 'evidence_chunks.parquet'
OUT_DIR = 'models'
MODEL_NAME = 'all-MiniLM-L6-v2'
BATCH_SIZE = 64


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def load_claims(path):
    df = pd.read_csv(path)

    if 'claim_id' not in df.columns:
        df = df.reset_index().rename(columns = {'index': 'claim_id'})
    return df[['claim_id', 'claim']]

def load_chunks(path):
    df = pd.read_parquet(path)

    if "chunk_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "chunk_id"})
    
    return df[['chunk_id', 'chunk_text']]

def encode_texts(model, texts, batch_size = 64):
    all_embs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i+batch_size]
        emb = model.encode(batch, convert_to_numpy = True, normalize_embeddings = True)
        all_embs.append(emb)
    if len(all_embs) == 0:
        return np.zeros((0, model.get_sentence_embedding_dimesion()), np.float32)
    return np.vstack(all_embs)


def main():
    ensure_dir(OUT_DIR)
    
    print('Loading sentence-transformers model:', MODEL_NAME)
    from sentence_transformers import SentenceTransformer 
    model = SentenceTransformer(MODEL_NAME)

    #embedding claims dataset first:
    print('Loading claims dataset')
    claims_df = load_claims(CLAIMS_CSV)
    print('Number of claims:', len(claims_df))
    claim_texts = claims_df['claim'].fillna('').astype(str).tolist()

    print('Encoding claims')
    claim_embs = encode_texts(model, claim_texts, batch_size=BATCH_SIZE)
    print('Claim embeddings shape:', claim_embs.shape)

    claims_meta_path = os.path.join(OUT_DIR, "claims_meta.csv")
    npy_claim_path = os.path.join(OUT_DIR, "claim_embeddings.npy")
    np.save(npy_claim_path, claim_embs.astype(np.float32))
    claims_df.to_csv(claims_meta_path, index=False)
    print("Saved claim embeddings in", npy_claim_path)
    print("Saved claims metadata in", claims_meta_path)

    #now embedding chunks:
    print("Loading chunks")
    chunks_df = load_chunks(CHUNKS_PARQUET)
    print("Number of chunks:", len(chunks_df))

    chunk_texts = chunks_df["chunk_text"].fillna("").astype(str).tolist()
    print("Encoding chunks")
    chunk_embs = encode_texts(model, chunk_texts, batch_size=BATCH_SIZE)
    print("Chunk embeddings shape:", chunk_embs.shape)

    chunks_meta_path = os.path.join(OUT_DIR, "chunks_meta.parquet")
    npy_chunk_path = os.path.join(OUT_DIR, "chunk_embeddings.npy")
    np.save(npy_chunk_path, chunk_embs.astype(np.float32))
    chunks_df.to_parquet(chunks_meta_path, index=False)
    print("Saved chunk embeddings in", npy_chunk_path)
    print("Saved chunks metadata in", chunks_meta_path)

    print("All done.")

if __name__ == "__main__":
    main()
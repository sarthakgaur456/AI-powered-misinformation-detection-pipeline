import os
import numpy as np
import faiss
import pandas as pd

EMB_PATH = "models/chunk_embeddings.npy"
META_IN = "models/chunks_meta.parquet"
INDEX_OUT = "models/evidence_faiss.index"
META_OUT = "models/evidence_metadata.parquet"

def main():
    assert os.path.exists(EMB_PATH), f"Embeddings not found: {EMB_PATH}"
    assert os.path.exists(META_IN), f"Metadata not found: {META_IN}"

    print('Loading embeddings')
    embs = np.load(EMB_PATH)
    print('Embeddings shape:', embs.shape)
    embs = embs.astype('float32')

    d = embs.shape[1]
    print('Embedding dimension d =', d)

    index = faiss.IndexFlatIP(d)
    print('Adding embeddings to index')
    index.add(embs)
    print('Index ntotal: =', index.ntotal)

    print ('Saving index to:', INDEX_OUT)
    faiss.write_index(index, INDEX_OUT)

    meta = pd.read_parquet(META_IN)
    meta.to_parquet(META_OUT, index=False)
    print("Saved metadata copy to:", META_OUT)

    print('All done')

if __name__ == '__main__':
    main()

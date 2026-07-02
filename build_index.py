"""Run during Render build to create faiss_index from catalog_clean.json"""
import json, pickle, numpy as np, faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

print("Loading catalog...")
with open("catalog_clean.json") as f:
    catalog = json.load(f)
print(f"Catalog size: {len(catalog)}")

print("Building embeddings (CPU)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [item["embed_text"] for item in catalog]
embeddings = model.encode(texts, show_progress_bar=True, batch_size=32, device="cpu")
embeddings = np.array(embeddings, dtype="float32")
faiss.normalize_L2(embeddings)

print("Building FAISS index...")
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
print(f"Index has {index.ntotal} vectors")

print("Saving faiss_index.pkl...")
with open("faiss_index.pkl", "wb") as f:
    pickle.dump({"index": index, "catalog": catalog}, f)

print("Done.")

import pickle, numpy as np, faiss, logging
from pathlib import Path
from sentence_transformers import SentenceTransformer

logger   = logging.getLogger(__name__)
_model   = None
_index   = None
_catalog = None

def _load():
    global _model, _index, _catalog
    if _model is None:
        logger.info("Loading sentence-transformer model...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    if _index is None:
        idx_path = Path(__file__).parent / "faiss_index.pkl"
        logger.info(f"Loading FAISS index from {idx_path}...")
        with open(idx_path, "rb") as f:
            data = pickle.load(f)
        _index   = data["index"]
        _catalog = data["catalog"]
        logger.info(f"Index loaded: {_index.ntotal} vectors, {len(_catalog)} items")

def get_catalog() -> list[dict]:
    """Return the full catalog (used for anchor injection in agent.py)."""
    _load()
    return _catalog

def retrieve_assessments(query: str, top_k: int = 20, catalog=None) -> list[dict]:
    _load()
    q_emb = _model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = _index.search(q_emb, min(top_k * 3, len(_catalog)))

    query_words = {w for w in query.lower().split() if len(w) > 3}

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        item = _catalog[idx]
        # Normalize by how much of the item's own name is covered by the
        # query, instead of a flat per-match bonus. A flat bonus rewards
        # longer names for picking up more incidental word matches (e.g.
        # "Microsoft Excel 365 - Essentials (New)" beating "MS Excel (New)"
        # just for also containing "microsoft"). Coverage-based boosting
        # favors names that are a tighter, more complete match to the query.
        name_words = {w for w in item["name"].lower().split() if len(w) > 2}
        overlap = query_words & name_words
        boost = 0.3 * (len(overlap) / len(name_words)) if name_words else 0.0
        results.append((item, float(score) + boost))

    results.sort(key=lambda x: x[1], reverse=True)
    return [r[0] for r in results[:top_k]]

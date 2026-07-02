import pickle, numpy as np, faiss, logging
from pathlib import Path
from sentence_transformers import SentenceTransformer
logger = logging.getLogger(__name__)
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
def retrieve_assessments(query: str, top_k: int = 20, catalog: list[dict] = None) -> list[dict]:
    _load()
    # NOTE: retrieval always uses the catalog embedded in faiss_index.pkl,
    # since FAISS index positions are only valid against that exact list/order.
    # `catalog` is accepted for call-signature compatibility with agent.py but
    # is not used for lookups — if it's passed and its length differs from the
    # index's own catalog, that likely means catalog_clean.json and
    # faiss_index.pkl are out of sync and need to be regenerated together.
    if catalog is not None and len(catalog) != len(_catalog):
        logger.warning(
            f"retrieve_assessments: passed catalog has {len(catalog)} items but "
            f"faiss_index.pkl catalog has {len(_catalog)} — these should match. "
            f"Using the index's own catalog for retrieval."
        )
    q_emb = _model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = _index.search(q_emb, min(top_k * 3, len(_catalog)))
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        item  = _catalog[idx]
        boost = sum(
            0.15 for w in query.lower().split()
            if len(w) > 3 and w in item["name"].lower()
        )
        results.append((item, float(score) + boost))
    results.sort(key=lambda x: x[1], reverse=True)
    return [r[0] for r in results[:top_k]]

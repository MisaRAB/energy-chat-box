# app/retrieval.py
import json
import numpy as np
import ollama
from pathlib import Path
from typing import List, Dict

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
VEC = np.load(DATA_DIR / "vectors.npy")   # (N, D), normalized
META = json.load(open(DATA_DIR / "chunks.json", "r", encoding="utf-8"))
CHUNKS = META["chunks"]
METAS  = META["metas"]

def embed_query(q: str) -> np.ndarray:
    r = ollama.embeddings(model="nomic-embed-text", prompt=q)
    v = np.array(r["embedding"], dtype=np.float32)
    v = v / (np.linalg.norm(v) + 1e-9)
    return v

def retrieve(query: str, k: int = 5, tag_hint: str | None = None) -> List[Dict]:
    qv = embed_query(query)                  # (D,)
    sims = VEC @ qv                          # cosine if both normalized
    idx = np.argsort(-sims)[:50]             # pool top 50
    cands = []
    for i in idx:
        m = METAS[i].copy()
        m["text"] = CHUNKS[i]
        m["score"] = float(sims[i])
        cands.append(m)

    # optional: lightweight tag boost
    if tag_hint:
        for c in cands:
            if tag_hint.lower() in (c.get("tags") or []):
                c["score"] += 0.05

    # take top-k distinct titles
    out, seen_titles = [], set()
    for c in sorted(cands, key=lambda x: -x["score"]):
        if c["title"] in seen_titles:
            continue
        out.append(c)
        seen_titles.add(c["title"])
        if len(out) >= k:
            break
    return out
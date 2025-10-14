# app/build_index.py
import os, glob, json, re
import numpy as np
import ollama
from pathlib import Path
from typing import List, Dict

DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)

def read_docs() -> List[Dict]:
    docs = []
    for p in sorted(glob.glob(str(DOCS_DIR / "*.md"))):
        text = Path(p).read_text(encoding="utf-8")
        title = extract_title(text) or Path(p).stem
        tags = extract_tags(text)
        docs.append({"path": p, "title": title, "tags": tags, "text": text})
    return docs

def extract_title(text: str) -> str | None:
    m = re.search(r"^\s*#\s*(.+)$", text, flags=re.MULTILINE)
    return m.group(1).strip() if m else None

def extract_tags(text: str) -> list[str]:
    m = re.search(r"^Tags:\s*(.+)$", text, flags=re.MULTILINE | re.IGNORECASE)
    if not m: return []
    raw = m.group(1).strip()
    return [t.strip().lower() for t in re.split(r"[,\[\]]", raw) if t.strip() and t.strip().lower() != "tags"]

def chunk_markdown(md: str, target_tokens: int = 300, overlap: int = 50) -> List[str]:
    # simple sentence splitter; good enough for toy corpus
    sents = re.split(r"(?<=[\.\?\!])\s+", md)
    chunks, cur = [], []
    cur_len = 0
    for s in sents:
        tok = max(1, len(s.split()))
        if cur_len + tok > target_tokens and cur:
            chunks.append(" ".join(cur))
            # overlap: keep last ~overlap words
            cur = " ".join(" ".join(cur).split()[-overlap:]).split()
            cur_len = len(cur)
        cur.extend(s.split())
        cur_len += tok
    if cur:
        chunks.append(" ".join(cur))
    # drop very short chunks
    return [c for c in chunks if len(c.split()) > 20]

def embed_texts(texts: List[str]) -> np.ndarray:
    # Use Ollama's local embedding endpoint (nomic-embed-text)
    vecs = []
    for t in texts:
        r = ollama.embeddings(model="nomic-embed-text", prompt=t)
        vecs.append(np.array(r["embedding"], dtype=np.float32))
    vecs = np.vstack(vecs)
    # normalize for cosine similarity via dot product
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    return vecs / norms

def main():
    docs = read_docs()
    chunks, metas = [], []
    for d in docs:
        for ch in chunk_markdown(d["text"]):
            chunks.append(ch)
            metas.append({"title": d["title"], "path": d["path"], "tags": d["tags"]})

    if not chunks:
        raise SystemExit("No chunks produced; check docs/*.md")

    print(f"Embedding {len(chunks)} chunks…")
    vecs = embed_texts(chunks)  # (N, D) float32 normalized

    # save
    np.save(DATA_DIR / "vectors.npy", vecs)
    with open(DATA_DIR / "chunks.json", "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks, "metas": metas}, f, ensure_ascii=False, indent=2)

    print(f"Saved vectors.npy and chunks.json to {DATA_DIR}")

if __name__ == "__main__":
    main()
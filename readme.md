# Local RAG Energy Assistant (Streamlit + Ollama)

A lightweight **Retrieval-Augmented Generation (RAG)** system that runs entirely **locally**, using  
**[Ollama](https://ollama.ai)** for large language models and embeddings, and **Streamlit** for a friendly chat interface.

![Live energy assistant chatbox](images/LLL-RAG.png)
---

## Overview

This project demonstrates how to combine **local LLMs** with **semantic retrieval** for an energy-focused assistant.

The app runs a **router model** that decides whether a user question:
- needs **live numerical data from National Grid API** (→ `TOOL`),
- requires **contextual knowledge** (→ `RAG`),
- or is just **small talk** (→ `CHITCHAT`).

Depending on the route:
- **TOOL** → extracts time expressions and fetches live or forecasted carbon-intensity values from the **National Grid API**.  
- **RAG** → retrieves relevant chunks from **locally embedded Markdown documents**.  
  The user query is embedded with `nomic-embed-text`, compared (cosine similarity) with stored document vectors (`vectors.npy`),  
  and the top-k most similar chunks form the context for the final answer.
- **CHITCHAT** → uses the chat model directly for conversational replies.

All models — chat and embedding — run **fully offline** through Ollama.

---

##  Architecture
User → Streamlit UI
│
▼
route_query() ── llama3:instruct (zero-shot router)
│
├── TOOL     → extract_when() → National Grid API → llm_answer()
├── RAG      → retrieve() → embed query → cosine search on vectors.npy
│               → nearest document chunks → llm_answer()
└── CHITCHAT → direct chat

---

## Tech Stack

| Component | Description |
|------------|--------------|
| **Ollama** | Local LLM runtime (chat + embeddings) |
| **llama3:instruct / llama3.1** | Chat model for reasoning & answers |
| **nomic-embed-text** | Embedding model for document vectors |
| **Streamlit** | Browser-based chat interface |
| **Python** | Core orchestration logic |
| **NumPy / JSON** | Lightweight vector store (`vectors.npy` + `chunks.json`) |

---

## 🪴 Setup & Run

### 1. Pull models
```bash
ollama pull llama3.instruct
ollama pull nomic-embed-text
```

### 2. Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Build the local document index
```bash
python -m app.build_index
```

This creates:
	•	data/chunks.json → text + metadata
	•	data/vectors.npy → embeddings (via nomic-embed-text)

### 4. Launch the app
Start Ollama in one terminal:
```bash
ollama serve
```
Then run Streamlit in another:
```bash
python -m streamlit run ui/streamlit_app.py
```


## Project Structure

├─ streamlit_app.py           # Streamlit UI
├─ app/
│   ├─ router_llm.py          # zero-shot router (TOOL / RAG / CHITCHAT)
│   ├─ orchestrate.py         # main logic & answer()
│   ├─ retrieval.py           # embedding-based retriever
│   ├─ build_index.py         # embed Markdown docs → vectors.npy + chunks.json
│   └─ tools.py               # API calls (e.g., National Grid)
├─ data/
│   ├─ docs/                  # local knowledge base (.md)
│   ├─ chunks.json
│   └─ vectors.npy
└─ requirements.txt
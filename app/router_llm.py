# app/router_llm.py
import os
import ollama

# ensure we talk to the local daemon
os.environ.setdefault("OLLAMA_HOST", "http://127.0.0.1:11434")

ROUTER_PROMPT = """You are a router.
Return exactly one token: TOOL, RAG, or CHITCHAT.
- TOOL: asks for numbers, forecasts, statistics, comparisons, or a value at a time.
- RAG: asks for explanations, definitions, reasons, context, or 'why/how'.
- CHITCHAT: greetings or small talk.

Only output one of: TOOL, RAG, CHITCHAT.
"""

def route_query(query: str, model: str = "llama3:instruct") -> str:
    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": ROUTER_PROMPT},
            {"role": "user", "content": query}
        ],
        options={"temperature": 0}
    )
    out = resp["message"]["content"].strip().upper()
    if "TOOL" in out: return "TOOL"
    if "RAG" in out: return "RAG"
    return "CHITCHAT"
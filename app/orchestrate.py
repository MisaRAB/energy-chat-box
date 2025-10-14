# app/orchestrate.py
import re
from datetime import datetime, timedelta
from dateutil import parser as dateparser
import os
import ollama
from typing import Dict, Any
from .retrieval import retrieve
from .router_llm import route_query
from .tools import get_ci_forecast

os.environ.setdefault("OLLAMA_HOST", "http://127.0.0.1:11434")
ANSWER_SYS = (
    "You are an energy assistant. Answer USING ONLY the provided context. "
    "If missing information, say 'I don't know'. Be concise and include sources if present."
)

_TIME_RE = re.compile(r'\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b', re.I)
_DATE_RE = re.compile(r'\b(20\d{2}-\d{2}-\d{2})(?:[ T](\d{1,2})(?::(\d{2}))?\s*(am|pm)?)?\b', re.I)

def _norm_time(h, m=None, mer=None):
    h = int(h); m = int(m or 0)
    if mer:  # handle am/pm
        mer = mer.lower()
        if mer == 'pm' and h != 12: h += 12
        if mer == 'am' and h == 12: h = 0
    return f"{h:02d}:{m:02d}"

def extract_when(q: str) -> str | None:
    """Return a simple time expression understood by get_ci_forecast(), or None for 'now'."""
    s = q.lower().strip()

    # explicit ISO date (with optional time)
    m = _DATE_RE.search(s)
    if m:
        date = m.group(1)
        h, mm, mer = m.group(2), m.group(3), m.group(4)
        if h:
            return f"{date} {_norm_time(h, mm, mer)}"
        return f"{date} 00:00"

    # keywords
    if "now" in s:
        return None
    if "tomorrow" in s:
        m = _TIME_RE.search(s)
        t = _norm_time(*m.groups()) if m else "18:00"
        dt = (datetime.now() + timedelta(days=1)).date().isoformat()
        return f"{dt} {t}"
    if "today" in s:
        m = _TIME_RE.search(s)
        t = _norm_time(*m.groups()) if m else "18:00"
        dt = datetime.now().date().isoformat()
        return f"{dt} {t}"

    # patterns like "at 6pm" → assume today
    m = re.search(r'\bat\s+' + _TIME_RE.pattern, s)
    if m:
        # groups( ) of inner _TIME_RE are at the tail
        h, mm, mer = m.groups()[-3:]
        dt = datetime.now().date().isoformat()
        return f"{dt} {_norm_time(h, mm, mer)}"

    # bare time like "6pm" or "18:00" → assume today
    m = _TIME_RE.search(s)
    if m:
        dt = datetime.now().date().isoformat()
        return f"{dt} {_norm_time(*m.groups())}"

    return None  # let the tool treat as "now"




def _humanize_window(when_str: str) -> str:
    # e.g. "2025-08-22T18:00Z → 2025-08-22T18:30Z" -> "18:00–18:30 (UTC)"
    try:
        left, right = [s.strip() for s in when_str.split("→")]
        L = dateparser.parse(left)
        R = dateparser.parse(right)
        return f"{L.strftime('%H:%M')}–{R.strftime('%H:%M')} (UTC)"
    except Exception:
        return when_str

def llm_answer(user_q: str, context: str, model: str = "llama3:instruct") -> str:
    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": ANSWER_SYS},
            {"role": "user", "content": f"Question: {user_q}\n\nContext:\n{context}"}
        ],
        options={"temperature": 0.2}
    )
    return resp["message"]["content"]

def answer(user_q: str, model: str = "llama3:instruct") -> Dict[str, Any]:
    route = route_query(user_q, model=model)

    # if route == "TOOL":
    #     data = get_ci_forecast()  # TODO: parse 'user_q' to extract time/region
    #     context = (
    #         f"Forecast carbon intensity: {data['carbon_intensity_gco2_per_kwh']} gCO2/kWh "
    #         f"at {data['when']}. Fuel mix (pct): {data['mix_pct']}. "
    #         f"Source: {data['source']}."
    #     )
    #     text = llm_answer(user_q, context, model=model)
    #     cites = [data["source"]]

    if route == "TOOL":
        when_hint = extract_when(user_q)  # returns e.g. "2025-08-22 18:00" or None
        data = get_ci_forecast(when_hint)
    
        if data.get("carbon_intensity_gco2_per_kwh") is None:
            text = ("I couldn’t find a carbon‑intensity value for that time block. "
                    "Try another time within today/tomorrow, or check the National Grid API.")
            return {"route": route, "text": text, "citations": [data.get("source", "")]}
    
        window = _humanize_window(data["when"])
        ci = data["carbon_intensity_gco2_per_kwh"]
        idx = (data.get("index") or "").capitalize()
        forecast = data.get("forecast"); actual = data.get("actual")
    
        context = (
            f"Time window: {window}\n"
            f"Carbon intensity: {ci} gCO₂/kWh (index: {idx or 'n/a'})\n"
            f"Forecast: {forecast if forecast is not None else 'n/a'}\n"
            f"Actual: {actual if actual is not None else 'n/a'}\n"
            f"Source: {data['source']}"
        )
        text = llm_answer(user_q, context, model=model)
        return {"route": route, "text": text, "citations": [data["source"]]}


    elif route == "RAG":
        # optional hint: if the question looks definitional, prefer glossary/glossary-like
        tag_hint = "glossary" if any(w in user_q.lower() for w in ["what is", "define", "meaning"]) else None
        hits = retrieve(user_q, k=3, tag_hint=tag_hint)
        if not hits:
            context = "No documents loaded yet."
            text = llm_answer(user_q, context, model=model)
            cites = []
        else:
            context = "\n\n".join(f"[{h['title']}] {h['text']}" for h in hits)
            text = llm_answer(user_q, context, model=model)
            cites = list({h["title"] for h in hits})

    else:  # CHITCHAT
        resp = ollama.chat(model=model, messages=[{"role": "user", "content": user_q}])
        text, cites = resp["message"]["content"], []

    return {"route": route, "text": text, "citations": cites}
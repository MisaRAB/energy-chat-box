# app/tools.py
from __future__ import annotations
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional
import requests
from dateutil import parser as dateparser

API_BASE = "https://api.carbonintensity.org.uk"

def _preprocess_when(when: str) -> str:
    when = when.strip().lower()
    today = datetime.now().date()
    if when.startswith("today"):
        # e.g. "today 18:00"
        timepart = when.replace("today", "").strip()
        return f"{today.isoformat()} {timepart}"
    elif when.startswith("tomorrow"):
        tomorrow = datetime.now().date() + timedelta(days=1)
        timepart = when.replace("tomorrow", "").strip()
        return f"{tomorrow.isoformat()} {timepart}"
    return when  # fall back unchanged

def _iso_halfhour_window(dt: datetime) -> tuple[str, str]:
    # Carbon Intensity API works on 30‑min blocks; query a 30‑min window around dt
    dt = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    minute_block = 0 if dt.minute < 30 else 30
    start = dt.replace(minute=minute_block, second=0, microsecond=0)
    end   = start.replace(minute=minute_block)  # same block; API accepts from/to with 30‑min granularity
    # some clients prefer adding 30 minutes to end; the API tolerates equal from/to. To be safe:
    from datetime import timedelta
    end = start + timedelta(minutes=30)
    return start.isoformat(timespec="minutes").replace("+00:00","Z"), end.isoformat(timespec="minutes").replace("+00:00","Z")

def get_ci_forecast(when: Optional[str] = None) -> Dict[str, Any]:
    """
    Query National Grid ESO Carbon Intensity API for the half‑hour block covering 'when'.
    - when: natural string like 'today 18:00', '2025-08-22 18:00', etc.
    Returns a dict with {when, carbon_intensity_gco2_per_kwh, index, forecast, actual(optional), source}.
    """
    # 1) parse target time
    if when:
        when = _preprocess_when(when)
        dt = dateparser.parse(when)  # robust to many formats; assumes local time if tz missing
    else:
        dt = datetime.now()
    # 2) build time window in UTC
    start_iso, end_iso = _iso_halfhour_window(dt)
    url = f"{API_BASE}/intensity/{start_iso}/{end_iso}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()

    # 3) extract the latest block (API returns a list under 'data')
    blocks = data.get("data", [])
    if not blocks:
        return {
            "when": dt.isoformat(timespec="minutes"),
            "carbon_intensity_gco2_per_kwh": None,
            "index": None,
            "forecast": None,
            "actual": None,
            "source": f"{API_BASE}/intensity",
            "note": "No data returned for this window."
        }
    block = blocks[-1]
    intensity = block.get("intensity", {}) or {}
    return {
        "when": f"{block.get('from','?')} → {block.get('to','?')}",
        "carbon_intensity_gco2_per_kwh": intensity.get("forecast") or intensity.get("actual"),
        "index": intensity.get("index"),
        "forecast": intensity.get("forecast"),
        "actual": intensity.get("actual"),
        "source": url
    }

TOOL_SCHEMA = {
    "get_ci_forecast": {
        "args": ["when:str (e.g., 'today 18:00' or '2025-08-21 18:00')"],
        "desc": "Return forecast carbon intensity and fuel mix at a given time."
    }
}

# (optional) you can extend with regional endpoints later if you want a post code region
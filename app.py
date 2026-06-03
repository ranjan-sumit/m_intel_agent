import json
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html import escape, unescape
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional local convenience
    load_dotenv = None


if load_dotenv:
    load_dotenv()


NVIDIA_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_NVIDIA_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1.5"
BLS_CPI_SERIES = {
    "Headline CPI": "CUUR0000SA0",
    "Food at home": "CUUR0000SAF11",
    "Household furnishings": "CUUR0000SAH3",
    "Gasoline": "CUUR0000SETB",
}


st.set_page_config(
    page_title="Dollar Tree Market Intelligence Workbench",
    page_icon="DT",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    :root {
        --bg: #F5F7FB;
        --sf: #FFFFFF;
        --s2: #F8FAFE;
        --s3: #F0F4F9;
        --s4: #E9EFF5;
        --or: #F47B25;
        --olt: #FF9F50;
        --odk: #C45D0A;
        --og: rgba(244,123,37,0.12);
        --ob: rgba(244,123,37,0.07);
        --obr: rgba(244,123,37,0.25);
        --bl: #E2E8F0;
        --t: #1E293B;
        --t2: #475569;
        --t3: #94A3B8;
        --gr: #22C55E;
        --gbg: rgba(34,197,94,0.10);
        --am: #F59E0B;
        --abg: rgba(245,158,11,0.10);
        --rd: #EF4444;
        --rbg: rgba(239,68,68,0.08);
        --r: 12px;
        --rl: 16px;
        --sh: 0 1px 3px rgba(0,0,0,0.04);
        --shm: 0 6px 14px -4px rgba(0,0,0,0.10);
        --tr: 0.2s cubic-bezier(0.4,0,0.2,1);
    }
    * { box-sizing: border-box; }
    html, body, [class*="css"] {
        font-family: 'Inter', ui-sans-serif, system-ui, sans-serif;
        color: var(--t);
    }
    .stApp { background: var(--bg); }
    #MainMenu, footer { visibility: hidden; }
    header { visibility: hidden; }
    .accent-bar {
        height: 3px;
        background: linear-gradient(90deg, var(--odk), var(--or), var(--olt), var(--or));
        background-size: 200%;
        animation: shimmer 3s linear infinite;
        width: 100%;
        margin: -1.25rem 0 0.9rem 0;
    }
    @keyframes shimmer { 0% { background-position: 200%; } 100% { background-position: -200%; } }
    @keyframes pdot { 0% { box-shadow: 0 0 0 0 rgba(34,197,94,0.5); } 50% { box-shadow: 0 0 0 5px rgba(34,197,94,0); } }
    .ldot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--gr);
        animation: pdot 2s infinite;
        display: inline-block;
    }
    .main .block-container {
        padding-top: 1.25rem;
        max-width: 1400px;
    }
    [data-testid="stSidebar"] {
        background: var(--sf) !important;
        border-right: 1px solid var(--bl) !important;
    }
    [data-testid="stSidebar"] * {
        color: var(--t2);
    }
    h1, h2, h3 {
        color: var(--t);
        letter-spacing: 0;
    }
    h1 { font-size: 2.25rem; font-weight: 900; letter-spacing: -0.02em; }
    .subtle {
        color: var(--t2);
        font-size: 0.94rem;
        line-height: 1.55;
    }
    .topbar {
        height: 54px;
        background: var(--sf);
        border: 1px solid var(--bl);
        border-radius: var(--rl);
        display: flex;
        align-items: center;
        padding: 0 18px;
        gap: 12px;
        box-shadow: var(--sh);
        margin-bottom: 18px;
    }
    .tt { font-size: 14px; font-weight: 800; color: var(--t); }
    .tt span { color: var(--t3); font-weight: 500; }
    .tbadge {
        background: var(--ob);
        border: 1px solid var(--obr);
        color: var(--or);
        font-size: 10px;
        font-weight: 800;
        padding: 2px 8px;
        border-radius: 20px;
    }
    .hero-shell {
        background:
            radial-gradient(circle at 88% 18%, rgba(244,123,37,0.18), transparent 28%),
            linear-gradient(135deg, #FFFFFF 0%, #F8FAFE 54%, #FFF7ED 100%);
        border: 1px solid var(--bl);
        border-radius: 22px;
        padding: 24px 26px;
        box-shadow: 0 12px 30px -22px rgba(15,23,42,0.35);
        margin-bottom: 16px;
        position: relative;
        overflow: hidden;
    }
    .hero-shell:before {
        content: "";
        position: absolute;
        left: 0;
        right: 0;
        top: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--odk), var(--or), var(--olt));
    }
    .hero-kicker {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: var(--ob);
        border: 1px solid var(--obr);
        color: var(--or);
        border-radius: 999px;
        padding: 4px 10px;
        font-size: 10px;
        font-weight: 900;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    .hero-title {
        font-size: 42px;
        line-height: 1.02;
        letter-spacing: -0.04em;
        font-weight: 950;
        color: var(--t);
        max-width: 820px;
        margin: 0;
    }
    .hero-copy {
        color: var(--t2);
        font-size: 14px;
        line-height: 1.7;
        max-width: 820px;
        margin: 14px 0 0 0;
    }
    .hero-side {
        background: rgba(255,255,255,0.74);
        border: 1px solid rgba(226,232,240,0.9);
        border-radius: var(--rl);
        padding: 14px;
        box-shadow: var(--sh);
    }
    .hero-side-label {
        color: var(--t3);
        font-size: 9px;
        font-weight: 900;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .hero-side-row {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        border-top: 1px solid var(--bl);
        padding-top: 8px;
        margin-top: 8px;
        font-size: 12px;
        color: var(--t2);
    }
    .hero-side-row strong { color: var(--t); }
    .source-tile {
        background: var(--sf);
        border: 1px solid var(--bl);
        border-radius: 14px;
        padding: 13px 14px;
        box-shadow: var(--sh);
        min-height: 86px;
        transition: all var(--tr);
    }
    .source-tile:hover { border-color: var(--obr); box-shadow: var(--shm); transform: translateY(-1px); }
    .source-name {
        font-size: 12px;
        font-weight: 850;
        color: var(--t);
        margin-bottom: 5px;
    }
    .source-meta {
        color: var(--t2);
        font-size: 11px;
        line-height: 1.45;
    }
    .console-panel {
        background: var(--sf);
        border: 1px solid var(--bl);
        border-radius: 18px;
        padding: 18px;
        box-shadow: var(--sh);
        margin-top: 12px;
    }
    .empty-console {
        background:
            linear-gradient(135deg, rgba(244,123,37,0.08), rgba(255,255,255,0.85)),
            var(--sf);
        border: 1px dashed var(--obr);
        border-radius: 18px;
        padding: 28px;
        min-height: 210px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 20px;
    }
    .empty-title {
        color: var(--t);
        font-size: 22px;
        line-height: 1.15;
        font-weight: 900;
        letter-spacing: -0.025em;
        margin-bottom: 8px;
    }
    .empty-body {
        color: var(--t2);
        font-size: 13px;
        line-height: 1.65;
        max-width: 620px;
    }
    .workflow {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 8px;
        margin-top: 12px;
    }
    .workflow-step {
        background: var(--s2);
        border: 1px solid var(--bl);
        border-radius: 12px;
        padding: 10px;
        font-size: 11px;
        color: var(--t2);
        font-weight: 700;
    }
    .workflow-step span {
        display: block;
        color: var(--or);
        font-size: 9px;
        letter-spacing: 1px;
        text-transform: uppercase;
        font-weight: 900;
        margin-bottom: 3px;
    }
    .metric-card {
        background: var(--sf);
        border: 1px solid var(--bl);
        border-radius: var(--rl);
        padding: 16px 18px;
        min-height: 132px;
        box-shadow: var(--sh);
        transition: all var(--tr);
    }
    .metric-card:hover {
        transform: translateY(-1px);
        box-shadow: var(--shm);
        border-color: var(--obr);
    }
    .metric-label {
        color: var(--t3);
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.11em;
        margin-bottom: 8px;
        font-weight: 800;
    }
    .metric-value {
        color: var(--t);
        font-size: 2.05rem;
        font-weight: 900;
        line-height: 1;
        letter-spacing: -0.04em;
    }
    .metric-note {
        color: var(--t2);
        font-size: 0.82rem;
        margin-top: 10px;
        line-height: 1.4;
    }
    .pill {
        display: inline-block;
        border-radius: 999px;
        padding: 3px 10px;
        font-size: 0.72rem;
        font-weight: 800;
        border: 1px solid var(--bl);
        color: var(--t2);
        background: var(--s2);
        margin-top: 8px;
    }
    .pill-high { color: var(--gr); background: var(--gbg); border-color: rgba(34,197,94,0.2); }
    .pill-medium { color: var(--am); background: var(--abg); border-color: rgba(245,158,11,0.2); }
    .pill-low { color: var(--rd); background: var(--rbg); border-color: rgba(239,68,68,0.2); }
    .brief-box {
        background: var(--sf);
        border: 1px solid var(--obr);
        border-left: 4px solid var(--or);
        border-radius: var(--rl);
        padding: 20px 22px;
        color: var(--t);
        line-height: 1.65;
        box-shadow: 0 0 0 3px var(--og);
    }
    .small-header {
        color: var(--t3);
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-weight: 800;
        margin: 8px 0 12px 0;
        padding-bottom: 6px;
        border-bottom: 1px solid var(--bl);
    }
    .action-card {
        background: var(--sf);
        border: 1px solid var(--bl);
        border-radius: var(--r);
        padding: 12px 14px;
        box-shadow: var(--sh);
        min-height: 112px;
    }
    .action-label {
        font-size: 9px;
        font-weight: 900;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: var(--or);
        margin-bottom: 7px;
    }
    .action-title {
        font-size: 13px;
        font-weight: 800;
        color: var(--t);
        margin-bottom: 5px;
    }
    .action-body {
        font-size: 12px;
        color: var(--t2);
        line-height: 1.45;
    }
    .note-box {
        background: rgba(244,123,37,0.04);
        border-left: 3px solid var(--or);
        border-radius: 0 8px 8px 0;
        padding: 9px 12px;
        font-size: 12px;
        color: var(--t2);
        margin: 8px 0;
    }
    div[role="radiogroup"] {
        background: var(--sf);
        border: 1px solid var(--bl);
        border-radius: 14px;
        padding: 5px;
        display: inline-flex;
        gap: 4px;
        box-shadow: var(--sh);
        margin: 4px 0 12px 0;
    }
    div[role="radiogroup"] label {
        border-radius: 10px !important;
        padding: 6px 13px !important;
        min-height: 34px !important;
        transition: all var(--tr);
    }
    div[role="radiogroup"] label:has(input:checked) {
        background: var(--ob) !important;
        border: 1px solid var(--obr) !important;
        color: var(--or) !important;
        font-weight: 850 !important;
    }
    div[role="radiogroup"] label span {
        font-size: 12px !important;
        font-weight: 750 !important;
    }
    div[data-testid="stButton"] button {
        background: var(--or);
        color: #fff;
        border: none;
        border-radius: var(--r);
        font-weight: 800;
        box-shadow: 0 2px 8px rgba(244,123,37,0.2);
        transition: all var(--tr);
    }
    div[data-testid="stButton"] button:hover {
        background: var(--odk);
        color: #fff;
        border: none;
        transform: translateY(-1px);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 1px solid var(--bl);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 9px 9px 0 0;
        color: var(--t2);
        font-weight: 700;
    }
    .stTabs [aria-selected="true"] {
        background: var(--ob);
        color: var(--or) !important;
        border: 1px solid var(--obr);
        border-bottom-color: transparent;
    }
    .stSelectbox>div>div, .stTextInput>div>div, .stTextArea>div>div {
        background: var(--s2) !important;
        border: 1px solid var(--bl) !important;
        border-radius: var(--r) !important;
        font-size: 13px !important;
        color: var(--t) !important;
    }
    [data-testid="stExpander"] {
        background: var(--sf);
        border: 1px solid var(--bl) !important;
        border-radius: var(--rl) !important;
        box-shadow: var(--sh);
    }
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--bl); border-radius: 2px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def safe_request(
    url: str,
    *,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
) -> Tuple[bool, Any, str]:
    try:
        if method.upper() == "POST":
            response = requests.post(url, headers=headers, params=params, json=json_body, timeout=timeout)
        else:
            response = requests.get(url, headers=headers, params=params, timeout=timeout)
        response.raise_for_status()
        try:
            return True, response.json(), "success"
        except ValueError:
            return True, response.text, "success"
    except requests.RequestException as exc:
        return False, None, str(exc)


def confidence_class(confidence: str) -> str:
    lookup = {"High": "pill-high", "Medium": "pill-medium", "Low": "pill-low"}
    return lookup.get(confidence, "")


def risk_band(score: float) -> str:
    if score >= 8:
        return "High"
    if score >= 5:
        return "Medium"
    return "Low"


def source_confidence(publisher: str) -> str:
    high = ["reuters", "associated press", "ap news", "bloomberg", "sec", "dollar tree", "pr newswire"]
    medium = ["cnbc", "yahoo", "nasdaq", "marketwatch", "forbes", "investing.com", "retail dive"]
    p = (publisher or "").lower()
    if any(name in p for name in high):
        return "High"
    if any(name in p for name in medium):
        return "Medium"
    return "Medium" if publisher else "Low"


def validate_nvidia(api_key: str, model: str) -> Tuple[bool, str]:
    if not api_key:
        return False, "NVIDIA key not provided. Brief generation will use a local fallback."
    prompt = "Return exactly: connected"
    ok, data, msg = safe_request(
        NVIDIA_CHAT_URL,
        method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json_body={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 8,
        },
        timeout=30,
    )
    if not ok:
        return False, msg
    content = data.get("choices", [{}])[0].get("message", {}).get("content")
    content_text = str(content or "").strip()
    return True, f"Connected. Model responded: {content_text or 'ok'}"


def validate_apify(token: str) -> Tuple[bool, str]:
    if not token:
        return False, "Apify token not provided. Apify collectors will be skipped."
    try:
        from apify_client import ApifyClient
    except ImportError:
        return False, "apify-client is not installed. Install requirements before running Apify collectors."
    try:
        client = ApifyClient(token)
        user = client.user().get()
        username = user.get("username") or user.get("email") or "Apify user"
        return True, f"Connected as {username}."
    except Exception as exc:  # pragma: no cover - depends on live Apify service
        return False, str(exc)


def build_cpi_signal(data: Dict[str, Any], label: str, series_id: str) -> Tuple[Optional[Dict[str, Any]], pd.DataFrame]:
    series = data.get("Results", {}).get("series", [])
    rows = []
    for item in (series[0].get("data", []) if series else [])[:24]:
        period = item.get("period", "")
        if period == "M13":
            continue
        try:
            cpi_value = float(item["value"])
        except (TypeError, ValueError, KeyError):
            continue
        rows.append(
            {
                "category": label,
                "series_id": series_id,
                "year": int(item["year"]),
                "period": period,
                "month": item.get("periodName", ""),
                "cpi_value": cpi_value,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return None, df
    df = df.sort_values(["year", "period"]).reset_index(drop=True)
    df["cpi_mom_change_pct"] = df["cpi_value"].pct_change() * 100
    df["cpi_yoy_change_pct"] = df["cpi_value"].pct_change(12) * 100
    latest = df.iloc[-1].to_dict()
    mom = latest.get("cpi_mom_change_pct")
    yoy = latest.get("cpi_yoy_change_pct")
    score = 4.0
    if pd.notna(mom):
        score = min(10.0, max(1.0, 4.0 + float(mom) * 5.0))
    if pd.notna(yoy) and yoy > 4:
        score = min(10.0, score + 1.0)
    signal_name = "inflation_pressure_score" if label == "Headline CPI" else f"{label.lower().replace(' ', '_')}_cpi_pressure_score"
    signal = {
        "date": f"{int(latest['year'])}-{str(latest['period']).replace('M', '').zfill(2)}",
        "retailer": "Dollar Tree",
        "region": "US",
        "source": "BLS CPI",
        "signal_area": "Inflation" if label == "Headline CPI" else "Category CPI",
        "signal_name": signal_name,
        "signal_value": round(float(score), 2),
        "risk_score": round(float(score), 2),
        "confidence": "High",
        "business_impact": f"{label} inflation can affect price sensitivity, category demand, and basket mix.",
        "recommended_action": "Use category CPI as an external regressor and validate against internal category sales.",
        "raw_reference": f"{label}: CPI {latest['cpi_value']}",
    }
    return signal, df


def collect_bls_cpi(bls_key: str = "") -> Dict[str, Any]:
    payload: Dict[str, Any] = {"seriesid": list(BLS_CPI_SERIES.values())}
    if bls_key:
        payload["registrationkey"] = bls_key
    signals = []
    tables = []
    raw = {}
    errors = []
    ok, data, msg = safe_request(
        "https://api.bls.gov/publicAPI/v2/timeseries/data/",
        method="POST",
        headers={"Content-Type": "application/json"},
        json_body=payload,
        timeout=30,
    )
    if not ok:
        return {"status": "failed", "source": "BLS CPI", "error": msg, "raw": None, "rows": []}
    if data.get("status") != "REQUEST_SUCCEEDED":
        return {
            "status": "failed",
            "source": "BLS CPI",
            "error": "; ".join(data.get("message", [])) or "BLS request was not processed.",
            "raw": data,
            "rows": [],
        }
    series_by_id = {
        series.get("seriesID"): series
        for series in data.get("Results", {}).get("series", [])
    }
    for label, series_id in BLS_CPI_SERIES.items():
        series_payload = {"Results": {"series": [series_by_id.get(series_id, {})]}}
        raw[label] = series_payload
        signal, df = build_cpi_signal(series_payload, label, series_id)
        if signal:
            signals.append(signal)
        if not df.empty:
            tables.append(df)
    if not signals:
        return {"status": "failed", "source": "BLS CPI", "error": "; ".join(errors) or "No CPI rows returned.", "raw": raw, "rows": []}
    table = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    return {"status": "success", "source": "BLS CPI", "error": "; ".join(errors), "raw": raw, "rows": signals, "table": table}


def classify_recall(reason: str) -> Tuple[str, float]:
    text = (reason or "").lower()
    if any(word in text for word in ["lead", "heavy metal", "salmonella", "listeria", "e. coli", "contamination"]):
        return "high_safety_risk", 8.0
    if any(word in text for word in ["undeclared", "allergen", "milk", "peanut", "soy", "tree nut"]):
        return "allergen_risk", 6.5
    if any(word in text for word in ["mislabel", "label"]):
        return "labeling_risk", 4.5
    return "general_recall_risk", 5.0


def extract_upcs(text: str) -> List[str]:
    candidates = re.findall(r"(?:UPC(?:\s*Code)?[:\s]*)?(\d(?:[\s-]?\d){7,13})", text or "", flags=re.IGNORECASE)
    cleaned = []
    for candidate in candidates:
        digits = re.sub(r"\D", "", candidate)
        if 8 <= len(digits) <= 14 and digits not in cleaned:
            cleaned.append(digits)
    return cleaned


def adjust_recall_score(base_score: float, classification: str, status: str) -> float:
    score = base_score
    class_text = (classification or "").lower()
    status_text = (status or "").lower()
    if "class i" in class_text:
        score += 1.5
    elif "class ii" in class_text:
        score += 0.8
    if "ongoing" in status_text:
        score += 1.0
    elif "terminated" in status_text:
        score -= 1.0
    return round(min(10.0, max(1.0, score)), 2)


def collect_fda_recalls(query: str, limit: int) -> Dict[str, Any]:
    params = {"search": query, "limit": limit}
    ok, data, msg = safe_request("https://api.fda.gov/food/enforcement.json", params=params, timeout=30)
    if not ok:
        return {"status": "failed", "source": "openFDA", "error": msg, "raw": None, "rows": [], "items": []}
    results = data.get("results", [])
    rows = []
    items = []
    for item in results:
        risk_type, base_score = classify_recall(item.get("reason_for_recall", ""))
        product = item.get("product_description", "Unknown product")
        state = item.get("state", "US")
        classification = item.get("classification", "")
        status = item.get("status", "")
        score = adjust_recall_score(base_score, classification, status)
        upcs = extract_upcs(f"{product} {item.get('code_info', '')}")
        items.append(
            {
                "product": product,
                "reason": item.get("reason_for_recall", ""),
                "state": state,
                "classification": classification,
                "status": status,
                "recall_date": item.get("recall_initiation_date", ""),
                "distribution_pattern": item.get("distribution_pattern", ""),
                "recalling_firm": item.get("recalling_firm", ""),
                "upcs": ", ".join(upcs) if upcs else "",
                "sku_match_status": "unknown",
                "risk_type": risk_type,
                "risk_score": score,
            }
        )
    aggregate_score = max([x["risk_score"] for x in items], default=1.0)
    ongoing_count = sum(1 for x in items if str(x.get("status", "")).lower() == "ongoing")
    class_i_count = sum(1 for x in items if "class i" in str(x.get("classification", "")).lower())
    signal = {
        "date": utc_now()[:10],
        "retailer": "Dollar Tree",
        "region": "US",
        "source": "openFDA",
        "signal_area": "Product Recalls",
        "signal_name": "recall_risk_score",
        "signal_value": len(items),
        "risk_score": round(aggregate_score, 2),
        "confidence": "High",
        "business_impact": "Food and beverage recalls can trigger inventory withdrawal, substitution demand, and safety review.",
        "recommended_action": "Prioritize ongoing and Class I recalls, then match UPCs against Dollar Tree inventory before store-level action.",
        "raw_reference": f"{len(items)} recall records; {ongoing_count} ongoing; {class_i_count} Class I",
    }
    rows.append(signal)
    return {"status": "success", "source": "openFDA", "error": "", "raw": data, "rows": rows, "items": items}


def gnews_package_collect(keyword: str, country: str, language: str, period: str, max_results: int) -> Optional[List[Dict[str, Any]]]:
    try:
        from gnews import GNews
    except ImportError:
        return None
    google_news = GNews(language=language, country=country, period=period, max_results=max_results)
    return google_news.get_news(keyword)


def google_news_rss_collect(keyword: str, country: str, language: str, period: str, max_results: int) -> List[Dict[str, Any]]:
    # Google News RSS supports a "when:" query operator. GNews package is preferred when installed.
    query = quote_plus(f"{keyword} when:{period}")
    country_code = country.upper()
    lang_code = language.lower()
    url = f"https://news.google.com/rss/search?q={query}&hl={lang_code}-{country_code}&gl={country_code}&ceid={country_code}:{lang_code}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    root = ET.fromstring(response.content)
    articles = []
    for item in root.findall(".//item")[:max_results]:
        source_node = item.find("source")
        articles.append(
            {
                "title": item.findtext("title", default=""),
                "description": item.findtext("description", default=""),
                "published date": item.findtext("pubDate", default=""),
                "url": item.findtext("link", default=""),
                "publisher": source_node.text if source_node is not None else "",
            }
        )
    return articles


def clean_news_description(description: str) -> str:
    text = re.sub(r"<[^>]+>", " ", description or "")
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def article_days_old(published_date: str) -> Optional[int]:
    if not published_date:
        return None
    try:
        published = parsedate_to_datetime(published_date)
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        return max(0, (datetime.now(timezone.utc) - published.astimezone(timezone.utc)).days)
    except (TypeError, ValueError):
        return None


def classify_news_title(title: str, description: str) -> Tuple[str, float, str]:
    text = f"{title} {description}".lower()
    if any(word in text for word in ["recall", "contamination", "lawsuit", "closure", "closing", "tariff", "warning"]):
        return "risk_event", 7.0, "negative"
    if any(word in text for word in ["inflation", "prices", "freight", "cost", "margin"]):
        return "price_pressure", 6.0, "negative"
    if any(word in text for word in ["deal", "sale", "promotion", "coupon", "holiday", "seasonal"]):
        return "demand_opportunity", 6.5, "positive"
    if any(word in text for word in ["earnings", "forecast", "guidance", "outlook"]):
        return "financial_update", 5.5, "neutral"
    return "general_market_news", 3.5, "neutral"


def collect_gnews(keywords: List[str], country: str, language: str, period: str, max_results: int) -> Dict[str, Any]:
    all_articles = []
    errors = []
    per_keyword_limit = max(1, int(max_results / max(1, len(keywords))))
    for keyword in keywords:
        try:
            articles = gnews_package_collect(keyword, country, language, period, per_keyword_limit)
            if articles is None:
                articles = google_news_rss_collect(keyword, country, language, period, per_keyword_limit)
            for article in articles or []:
                description = clean_news_description(article.get("description", ""))
                event_type, score, sentiment = classify_news_title(article.get("title", ""), description)
                publisher = article.get("publisher", "")
                if isinstance(publisher, dict):
                    publisher = publisher.get("title") or publisher.get("href") or ""
                published_date = article.get("published date") or article.get("published_date", "")
                days_old = article_days_old(published_date)
                if days_old is not None and days_old > 30:
                    score = max(1.0, score - 1.0)
                all_articles.append(
                    {
                        "keyword": keyword,
                        "title": article.get("title", ""),
                        "description": description,
                        "published_date": published_date,
                        "days_old": days_old,
                        "publisher": publisher,
                        "url": article.get("url", ""),
                        "source_tier": source_confidence(str(publisher)),
                        "event_type": event_type,
                        "sentiment": sentiment,
                        "risk_score": score,
                        "confidence": source_confidence(str(publisher)),
                    }
                )
        except Exception as exc:
            errors.append(f"{keyword}: {exc}")

    deduped = []
    seen = set()
    for article in all_articles:
        key = article["url"] or article["title"]
        if key and key not in seen:
            seen.add(key)
            deduped.append(article)

    score = round(float(pd.Series([a["risk_score"] for a in deduped]).mean()), 2) if deduped else 1.0
    signal = {
        "date": utc_now()[:10],
        "retailer": "Dollar Tree",
        "region": country.upper(),
        "source": "GNews / Google News RSS",
        "signal_area": "Retail News",
        "signal_name": "news_risk_score",
        "signal_value": len(deduped),
        "risk_score": score,
        "confidence": "Medium",
        "business_impact": "Recent news can reveal competitor moves, pricing pressure, recalls, store changes, and supply-chain risk.",
        "recommended_action": "Review high-risk articles and use NVIDIA classification before executive distribution.",
        "raw_reference": f"{len(deduped)} articles",
    }
    status = "success" if deduped or not errors else "failed"
    return {"status": status, "source": "GNews", "error": "; ".join(errors), "raw": deduped, "rows": [signal], "items": deduped}


def collect_apify_trends(token: str, keywords: List[str], geo: str, time_range: str) -> Dict[str, Any]:
    if not token:
        return {"status": "skipped", "source": "Apify Trends", "error": "No Apify token provided.", "raw": None, "rows": [], "items": []}
    try:
        from apify_client import ApifyClient
    except ImportError:
        return {"status": "failed", "source": "Apify Trends", "error": "apify-client is not installed.", "raw": None, "rows": [], "items": []}
    try:
        client = ApifyClient(token)
        run_input = {"geo": geo, "searchTerms": keywords, "timeRange": time_range}
        run = client.actor("apify/google-trends-scraper").call(run_input=run_input)
        items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
    except Exception as exc:
        return {"status": "failed", "source": "Apify Trends", "error": str(exc), "raw": None, "rows": [], "items": []}

    region_rows = []
    for item in items:
        keyword = item.get("searchTerm")
        for rank, region in enumerate(item.get("interestBySubregion", []) or [], start=1):
            values = region.get("value") or []
            if values:
                region_rows.append(
                    {
                        "keyword": keyword,
                        "region": region.get("geoName", ""),
                        "interest_score": values[0],
                        "rank": rank,
                    }
                )
    top_score = max([float(x["interest_score"]) for x in region_rows], default=0.0)
    signal_score = round(min(10.0, max(1.0, top_score / 10.0)), 2) if top_score else 1.0
    signal = {
        "date": utc_now()[:10],
        "retailer": "Dollar Tree",
        "region": geo,
        "source": "Apify Google Trends",
        "signal_area": "Search Demand",
        "signal_name": "search_demand_score",
        "signal_value": top_score,
        "risk_score": signal_score,
        "confidence": "Medium",
        "business_impact": "Search interest can reveal early demand shifts, promotional interest, and seasonal spikes.",
        "recommended_action": "Compare rising regions against sales, inventory, and competitor promotion calendars.",
        "raw_reference": f"{len(region_rows)} regional trend rows",
    }
    return {"status": "success", "source": "Apify Trends", "error": "", "raw": items, "rows": [signal], "items": region_rows}


def generate_fallback_brief(feature_df: pd.DataFrame, retailer: str, region: str) -> str:
    if feature_df.empty:
        return "No signals were collected. Add at least one enabled source and run the workbench again."
    strongest = feature_df.sort_values("risk_score", ascending=False).head(3)
    lines = [
        f"Market intelligence summary for {retailer} in {region}.",
        "",
        "Top signals:",
    ]
    for _, row in strongest.iterrows():
        lines.append(
            f"- {row['signal_area']}: {row['risk_score']}/10 from {row['source']}. {row['business_impact']}"
        )
    lines.extend(
        [
            "",
            "Recommended focus:",
            "Use these signals as forecast-ready external features, then validate them against internal POS, category, and inventory data before operational decisions.",
        ]
    )
    return "\n".join(lines)


def generate_nvidia_brief(api_key: str, model: str, feature_df: pd.DataFrame, articles: List[Dict[str, Any]], retailer: str, region: str) -> Tuple[str, str]:
    if not api_key:
        return generate_fallback_brief(feature_df, retailer, region), "fallback"
    payload = {
        "features": feature_df.to_dict(orient="records")[:12],
        "articles": articles[:8],
    }
    system = (
        "You are a retail market intelligence analyst. Ground your answer only in the supplied signals. "
        "Write concise executive guidance for buyers, category managers, demand planners, and supply-chain teams."
    )
    user = f"""
    Retailer: {retailer}
    Region: {region}

    Signal payload:
    {json.dumps(payload, indent=2)[:9000]}

    Produce:
    1. Top 3 insights
    2. Forecasting relevance
    3. Recommended actions
    4. Confidence and limitations
    """
    ok, data, msg = safe_request(
        NVIDIA_CHAT_URL,
        method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json_body={
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.25,
            "max_tokens": 900,
        },
        timeout=60,
    )
    if not ok:
        return generate_fallback_brief(feature_df, retailer, region), f"fallback: {msg}"
    content = data.get("choices", [{}])[0].get("message", {}).get("content")
    content_text = str(content or "").strip()
    return content_text or generate_fallback_brief(feature_df, retailer, region), "nvidia"


def render_metric_card(title: str, value: str, note: str, confidence: str = "") -> None:
    pill = f'<span class="pill {confidence_class(confidence)}">{confidence}</span>' if confidence else ""
    html = (
        '<div class="metric-card">'
        f'<div class="metric-label">{title}</div>'
        f'<div class="metric-value">{value}</div>'
        f"{pill}"
        f'<div class="metric-note">{note}</div>'
        "</div>"
    )
    st.markdown(
        html,
        unsafe_allow_html=True,
    )


def compute_composite_scores(feature_df: pd.DataFrame) -> Dict[str, float]:
    if feature_df.empty:
        return {"Market Opportunity": 0.0, "Market Risk": 0.0, "Forecast Impact": 0.0}
    area_scores: Dict[str, float] = {}
    for _, row in feature_df.iterrows():
        if pd.isna(row.get("risk_score")):
            continue
        area = str(row["signal_area"])
        area_scores[area] = max(area_scores.get(area, 0.0), float(row["risk_score"]))
    opportunity = max(
        area_scores.get("Retail News", 0.0),
        area_scores.get("Search Demand", 0.0),
        area_scores.get("Category CPI", 0.0) * 0.7,
    )
    risk = max(
        area_scores.get("Product Recalls", 0.0),
        area_scores.get("Inflation", 0.0),
        area_scores.get("Category CPI", 0.0),
    )
    impact = min(10.0, (opportunity * 0.45) + (risk * 0.45) + (len(feature_df) * 0.25))
    return {
        "Market Opportunity": round(opportunity, 2),
        "Market Risk": round(risk, 2),
        "Forecast Impact": round(impact, 2),
    }


def build_recommended_actions(feature_df: pd.DataFrame, results: Dict[str, Dict[str, Any]]) -> List[Dict[str, str]]:
    actions = []
    if feature_df.empty:
        return [{"label": "Setup", "title": "Run signal sources", "body": "Enable at least one source to generate forecast-ready rows."}]
    top_rows = feature_df.sort_values("risk_score", ascending=False).head(3)
    for _, row in top_rows.iterrows():
        actions.append(
            {
                "label": str(row["signal_area"]),
                "title": str(row["signal_name"]).replace("_", " ").title(),
                "body": str(row["recommended_action"]),
            }
        )
    apify_result = results.get("apify")
    if apify_result and apify_result.get("status") == "failed":
        actions.append(
            {
                "label": "Apify",
                "title": "Search demand skipped",
                "body": "Apify connected but failed during collection. If usage limit is exceeded, rerun after quota reset or keep MVP on public sources.",
            }
        )
    return actions[:4]


def render_action_card(label: str, title: str, body: str) -> None:
    html = (
        '<div class="action-card">'
        f'<div class="action-label">{escape(label)}</div>'
        f'<div class="action-title">{escape(title)}</div>'
        f'<div class="action-body">{escape(body)}</div>'
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_source_tile(name: str, status: str, detail: str) -> None:
    status_class = "pill-high" if status == "Active" else "pill-medium" if status == "Optional" else "pill-low"
    html = (
        '<div class="source-tile">'
        f'<div class="source-name">{escape(name)} <span class="pill {status_class}" style="margin-left:6px;margin-top:0;">{escape(status)}</span></div>'
        f'<div class="source-meta">{escape(detail)}</div>'
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_workflow_strip() -> None:
    steps = [
        ("01", "Collect APIs"),
        ("02", "Clean records"),
        ("03", "Score signals"),
        ("04", "Generate brief"),
        ("05", "Export features"),
    ]
    html = "<div class='workflow'>" + "".join(
        f"<div class='workflow-step'><span>{num}</span>{escape(label)}</div>" for num, label in steps
    ) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_score_chart(feature_df: pd.DataFrame) -> None:
    if feature_df.empty:
        st.info("No feature rows yet.")
        return
    fig = go.Figure(
        go.Bar(
            x=feature_df["risk_score"],
            y=feature_df["signal_area"],
            orientation="h",
            marker_color=["#22C55E" if x < 5 else "#F59E0B" if x < 8 else "#EF4444" for x in feature_df["risk_score"]],
            text=feature_df["risk_score"],
            textposition="auto",
        )
    )
    fig.update_layout(
        height=280,
        margin={"l": 10, "r": 20, "t": 10, "b": 10},
        xaxis={"range": [0, 10], "title": "Score"},
        yaxis={"title": ""},
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
    )
    st.plotly_chart(fig, width="stretch")


def parse_lines(text: str) -> List[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


with st.sidebar:
    st.title("Workbench Setup")
    st.caption("Credentials are used only for this Streamlit session.")

    nvidia_key = st.text_input("NVIDIA API key", value=os.getenv("NVIDIA_API_KEY", ""), type="password")
    nvidia_model = st.text_input("NVIDIA model", value=os.getenv("NVIDIA_MODEL", DEFAULT_NVIDIA_MODEL))
    apify_token = st.text_input("Apify token", value=os.getenv("APIFY_API_TOKEN", ""), type="password")
    bls_key = st.text_input("BLS API key optional", value=os.getenv("BLS_API_KEY", ""), type="password")

    st.markdown('<div class="small-header">Retail Context</div>', unsafe_allow_html=True)
    retailer = st.text_input("Retailer", value="Dollar Tree")
    region = st.text_input("Region", value="US")
    country = st.selectbox("News country", ["US", "CA", "GB", "AE", "IN"], index=0)
    language = st.selectbox("News language", ["en", "es", "fr", "ar", "hi"], index=0)

    st.markdown('<div class="small-header">Signal Sources</div>', unsafe_allow_html=True)
    use_gnews = st.checkbox("Retail news via GNews/RSS", value=True)
    use_bls = st.checkbox("Inflation via BLS CPI", value=True)
    use_fda = st.checkbox("Product recalls via openFDA", value=True)
    use_apify = st.checkbox("Search demand via Apify Trends", value=False)

    st.markdown('<div class="small-header">Run Controls</div>', unsafe_allow_html=True)
    validate_button = st.button("Validate credentials", width="stretch")
    run_button = st.button("Run intelligence", type="primary", width="stretch")


DEFAULT_NEWS_KEYWORDS = "\n".join(
    [
        "Dollar Tree inflation",
        "Dollar Tree prices",
        "Dollar Tree store closures",
        "Dollar Tree recall",
        "discount retail tariffs",
        "Dollar General promotion",
    ]
)
DEFAULT_TRENDS_KEYWORDS = "\n".join(
    [
        "Dollar Tree sales",
        "Dollar Tree coupons",
        "Dollar Tree near me",
        "Dollar Tree groceries",
        "cheap groceries",
    ]
)

st.session_state.setdefault("news_keywords_text", DEFAULT_NEWS_KEYWORDS)
st.session_state.setdefault("trends_keywords_text", DEFAULT_TRENDS_KEYWORDS)
st.session_state.setdefault("gnews_period", "7d")
st.session_state.setdefault("max_news", 24)
st.session_state.setdefault("fda_query", "product_description:(snacks OR candy OR beverages)")
st.session_state.setdefault("fda_limit", 8)
st.session_state.setdefault("apify_geo", "US")
st.session_state.setdefault("apify_time_range", "today 1-m")
st.session_state.setdefault("workbench_view", "Configure")
if st.session_state.pop("force_results_view", False):
    st.session_state["workbench_view"] = "Results"

st.markdown('<div class="accent-bar"></div>', unsafe_allow_html=True)
st.markdown(
    "<div class='topbar'>"
    "<div class='tt'>Market Intelligence <span>/ Dollar Tree · External Signals</span></div>"
    "<div class='tbadge'>AI Workbench</div>"
    "<div style='margin-left:auto;display:flex;align-items:center;gap:8px;'>"
    "<span class='ldot'></span>"
    "<span style='font-size:10px;color:var(--t3);font-weight:800;'>Live API Mode</span>"
    "<div style='width:28px;height:28px;border-radius:7px;background:linear-gradient(135deg,var(--odk),var(--or));display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:900;color:#fff;'>DT</div>"
    "</div></div>",
    unsafe_allow_html=True,
)

hero_left, hero_right = st.columns([2.2, 0.9], vertical_alignment="center")
with hero_left:
    st.markdown(
        "<div class='hero-shell'>"
        "<div class='hero-kicker'><span class='ldot'></span> External Signal Layer</div>"
        "<h1 class='hero-title'>Dollar Tree Market Intelligence Command Center</h1>"
        "<p class='hero-copy'>A retail-grade workbench that turns news, CPI, recalls, search demand, and API health into forecast-ready features, composite risk scores, and buyer actions.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
with hero_right:
    active_sources = sum([use_gnews, use_bls, use_fda, use_apify])
    st.markdown(
        "<div class='hero-side'>"
        "<div class='hero-side-label'>Run Profile</div>"
        f"<div class='hero-side-row'><span>Retailer</span><strong>{escape(retailer)}</strong></div>"
        f"<div class='hero-side-row'><span>Region</span><strong>{escape(region)}</strong></div>"
        f"<div class='hero-side-row'><span>Sources</span><strong>{active_sources} enabled</strong></div>"
        f"<div class='hero-side-row'><span>LLM</span><strong>{'NVIDIA' if nvidia_key.strip() else 'Fallback'}</strong></div>"
        "</div>",
        unsafe_allow_html=True,
    )

view = st.segmented_control(
    "Workbench view",
    ["Configure", "Results", "Raw Data"],
    required=True,
    label_visibility="collapsed",
    key="workbench_view",
    width="content",
)

if view == "Configure":
    st.markdown('<div class="small-header">Signal Source Stack</div>', unsafe_allow_html=True)
    src_cols = st.columns(4)
    source_specs = [
        ("GNews/RSS", "Active" if use_gnews else "Off", "Retail articles, sentiment hints, source confidence."),
        ("BLS CPI", "Active" if use_bls else "Off", "Headline and category inflation pressure features."),
        ("openFDA", "Active" if use_fda else "Off", "Recall severity, UPC extraction, SKU-match prep."),
        ("Apify Trends", "Optional" if use_apify else "Off", "Search interest and regional demand spikes."),
    ]
    for col, spec in zip(src_cols, source_specs):
        with col:
            render_source_tile(*spec)

    left, right = st.columns([1.15, 0.85])
    with left:
        st.markdown('<div class="console-panel">', unsafe_allow_html=True)
        st.subheader("Signal Keywords")
        st.text_area("GNews keywords", key="news_keywords_text", height=170)
        st.text_area("Apify Google Trends keywords", key="trends_keywords_text", height=150)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="console-panel">', unsafe_allow_html=True)
        st.subheader("Collector Settings")
        st.selectbox("GNews period", ["1d", "7d", "30d", "3m"], key="gnews_period")
        st.slider("Max news results", min_value=5, max_value=60, step=1, key="max_news")
        st.text_input("FDA recall search", key="fda_query")
        st.slider("FDA recall limit", min_value=1, max_value=25, key="fda_limit")
        st.text_input("Apify geo", key="apify_geo")
        st.selectbox("Apify time range", ["today 7-d", "today 1-m", "today 3-m", "today 12-m"], key="apify_time_range")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="small-header">Agent Flow</div>', unsafe_allow_html=True)
    render_workflow_strip()


if validate_button:
    with st.spinner("Validating credentials..."):
        n_ok, n_msg = validate_nvidia(nvidia_key.strip(), nvidia_model.strip())
        a_ok, a_msg = validate_apify(apify_token.strip())
    st.session_state["validation"] = {"nvidia": (n_ok, n_msg), "apify": (a_ok, a_msg)}

if "validation" in st.session_state:
    n_ok, n_msg = st.session_state["validation"]["nvidia"]
    a_ok, a_msg = st.session_state["validation"]["apify"]
    st.info(f"NVIDIA: {'Connected' if n_ok else 'Not connected'} - {n_msg}")
    st.info(f"Apify: {'Connected' if a_ok else 'Not connected'} - {a_msg}")


if run_button:
    news_keywords = parse_lines(st.session_state.get("news_keywords_text", DEFAULT_NEWS_KEYWORDS))
    trends_keywords = parse_lines(st.session_state.get("trends_keywords_text", DEFAULT_TRENDS_KEYWORDS))
    results: Dict[str, Dict[str, Any]] = {}
    all_rows: List[Dict[str, Any]] = []
    all_articles: List[Dict[str, Any]] = []

    progress = st.progress(0, text="Starting collectors")
    steps = [
        ("gnews", use_gnews),
        ("bls", use_bls),
        ("fda", use_fda),
        ("apify", use_apify),
    ]
    active_steps = [step for step in steps if step[1]]
    total = max(1, len(active_steps))
    completed = 0

    if use_gnews:
        progress.progress(completed / total, text="Collecting retail news")
        results["gnews"] = collect_gnews(
            news_keywords,
            country,
            language,
            st.session_state.get("gnews_period", "7d"),
            int(st.session_state.get("max_news", 24)),
        )
        all_rows.extend(results["gnews"].get("rows", []))
        all_articles.extend(results["gnews"].get("items", []))
        completed += 1

    if use_bls:
        progress.progress(completed / total, text="Collecting CPI inflation")
        results["bls"] = collect_bls_cpi(bls_key.strip())
        all_rows.extend(results["bls"].get("rows", []))
        completed += 1

    if use_fda:
        progress.progress(completed / total, text="Collecting FDA recalls")
        results["fda"] = collect_fda_recalls(
            st.session_state.get("fda_query", "product_description:(snacks OR candy OR beverages)"),
            int(st.session_state.get("fda_limit", 8)),
        )
        all_rows.extend(results["fda"].get("rows", []))
        completed += 1

    if use_apify:
        progress.progress(completed / total, text="Collecting Google Trends via Apify")
        results["apify"] = collect_apify_trends(
            apify_token.strip(),
            trends_keywords,
            st.session_state.get("apify_geo", "US"),
            st.session_state.get("apify_time_range", "today 1-m"),
        )
        all_rows.extend(results["apify"].get("rows", []))
        completed += 1

    progress.progress(1.0, text="Generating intelligence brief")
    feature_df = pd.DataFrame(all_rows)
    if not feature_df.empty:
        feature_df["retailer"] = retailer
        feature_df["region"] = feature_df["region"].replace({"US": region}) if region else feature_df["region"]
    brief, brief_source = generate_nvidia_brief(nvidia_key.strip(), nvidia_model.strip(), feature_df, all_articles, retailer, region)

    st.session_state["run"] = {
        "timestamp": utc_now(),
        "results": results,
        "feature_df": feature_df,
        "articles": all_articles,
        "brief": brief,
        "brief_source": brief_source,
    }
    st.session_state["force_results_view"] = True
    progress.empty()
    st.rerun()


if view == "Results":
    run = st.session_state.get("run")
    if not run:
        st.markdown(
            "<div class='empty-console'>"
            "<div>"
            "<div class='hero-kicker'>Ready for first run</div>"
            "<div class='empty-title'>No intelligence run yet.</div>"
            "<div class='empty-body'>Configure the signal stack, then click <strong>Run intelligence</strong> in the sidebar. The agent will collect public signals, normalize them into feature rows, score the risk/opportunity surface, and produce an executive brief.</div>"
            "</div>"
            "<div class='hero-side' style='min-width:260px;'>"
            "<div class='hero-side-label'>MVP Output</div>"
            "<div class='hero-side-row'><span>Feature rows</span><strong>CSV / JSON</strong></div>"
            "<div class='hero-side-row'><span>Scores</span><strong>0-10</strong></div>"
            "<div class='hero-side-row'><span>Brief</span><strong>NVIDIA / fallback</strong></div>"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        render_workflow_strip()
    else:
        feature_df = run["feature_df"]
        st.markdown(
            f"<div class='note-box'><strong>Last run:</strong> {escape(run['timestamp'])} UTC &nbsp; | &nbsp; <strong>Brief source:</strong> {escape(str(run['brief_source']))}</div>",
            unsafe_allow_html=True,
        )
        cols = st.columns(4)
        if feature_df.empty:
            for col, title in zip(cols, ["Signals", "Avg Score", "Highest Score", "Brief"]):
                with col:
                    render_metric_card(title, "0", "No successful feature rows yet.")
        else:
            avg_score = round(float(feature_df["risk_score"].mean()), 2)
            top = feature_df.sort_values("risk_score", ascending=False).iloc[0]
            composite_scores = compute_composite_scores(feature_df)
            with cols[0]:
                render_metric_card("Signals", str(len(feature_df)), "Forecast-ready rows generated.")
            with cols[1]:
                render_metric_card("Average Score", str(avg_score), f"{risk_band(avg_score)} overall signal intensity.")
            with cols[2]:
                render_metric_card("Top Signal", str(top["risk_score"]), str(top["signal_area"]), str(top["confidence"]))
            with cols[3]:
                render_metric_card("News Articles", str(len(run["articles"])), "Deduplicated retail news items.")

            st.markdown('<div class="small-header">Composite Agent Scores</div>', unsafe_allow_html=True)
            cscore_cols = st.columns(3)
            for col, (name, score) in zip(cscore_cols, composite_scores.items()):
                with col:
                    render_metric_card(name, str(score), f"{risk_band(score)} priority for planning.")

            st.markdown('<div class="small-header">Recommended Actions</div>', unsafe_allow_html=True)
            action_cols = st.columns(4)
            for col, action in zip(action_cols, build_recommended_actions(feature_df, run["results"])):
                with col:
                    render_action_card(action["label"], action["title"], action["body"])

            st.markdown('<div class="small-header">Signal Scores</div>', unsafe_allow_html=True)
            render_score_chart(feature_df)

            st.markdown('<div class="small-header">Forecast Feature Table</div>', unsafe_allow_html=True)
            st.dataframe(feature_df, width="stretch", hide_index=True)

            csv_data = feature_df.to_csv(index=False).encode("utf-8")
            json_data = json.dumps(feature_df.to_dict(orient="records"), indent=2).encode("utf-8")
            c1, c2 = st.columns([1, 1])
            with c1:
                st.download_button("Download forecast_features.csv", csv_data, "forecast_features.csv", "text/csv", width="stretch")
            with c2:
                st.download_button("Download normalized_signals.json", json_data, "normalized_signals.json", "application/json", width="stretch")

        st.markdown('<div class="small-header">Executive Brief</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="brief-box">{run["brief"].replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)


if view == "Raw Data":
    run = st.session_state.get("run")
    if not run:
        st.markdown(
            "<div class='empty-console'>"
            "<div>"
            "<div class='hero-kicker'>Raw Evidence</div>"
            "<div class='empty-title'>Collector payloads will appear after a run.</div>"
            "<div class='empty-body'>This view is intentionally evidence-first: GNews articles, CPI JSON, FDA recall records, Apify errors, and cleaned item tables stay inspectable for validation.</div>"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        for name, result in run["results"].items():
            status = result.get("status", "unknown")
            label = f"{name.upper()} - {status}"
            with st.expander(label, expanded=False):
                if result.get("error"):
                    st.warning(result["error"])
                if result.get("items"):
                    st.dataframe(pd.DataFrame(result["items"]), width="stretch", hide_index=True)
                raw_preview = result.get("raw")
                if raw_preview is not None:
                    st.code(json.dumps(raw_preview, indent=2, default=str)[:7000], language="json")


st.markdown(
    """
    <p class="subtle">
    Note: Google Trends values are relative indexes, GNews is a lightweight news signal, and recall data should be matched
    against internal SKU/UPC and inventory records before operational decisions.
    </p>
    """,
    unsafe_allow_html=True,
)

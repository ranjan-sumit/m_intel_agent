import json
import hashlib
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
APIFY_SAFE_TIME_RANGE = "today 7-d"
APIFY_HARD_KEYWORD_LIMIT = 2

SOURCE_ORDER = ["gnews", "bls", "fda", "weather", "apify"]
SOURCE_LABELS = {
    "gnews": "GNews / Google News RSS",
    "bls": "BLS CPI",
    "fda": "openFDA Food Enforcement",
    "weather": "NOAA Weather Alerts",
    "apify": "Apify Google Trends",
}
SOURCE_ENDPOINTS = {
    "gnews": "GNews package or Google News RSS search feed",
    "bls": "https://api.bls.gov/publicAPI/v2/timeseries/data/",
    "fda": "https://api.fda.gov/food/enforcement.json",
    "weather": "https://api.weather.gov/alerts/active",
    "apify": "apify/google-trends-scraper",
}
ANALYSIS_METHODS = {
    "gnews": "Classifies article titles/descriptions into event types, sentiment, source confidence, then averages article risk into one news feature.",
    "bls": "Batches CPI series, calculates month-over-month and year-over-year movement, then scores inflation pressure.",
    "fda": "Classifies recall reason, FDA class, status, UPC presence, and state coverage, then uses highest adjusted recall severity.",
    "weather": "Collects active NOAA alerts for the selected state, weights severity, urgency, and certainty, then caps supply-chain weather risk at 10.",
    "apify": "Uses top regional Google Trends index divided by 10; backend hard-limits time range and keyword count to protect quota.",
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
    .brief-shell {
        background:
            linear-gradient(135deg, rgba(255,255,255,0.98), rgba(248,250,254,0.94)),
            var(--sf);
        border: 1px solid var(--bl);
        border-radius: 20px;
        box-shadow: 0 18px 36px -28px rgba(15,23,42,0.55);
        overflow: hidden;
        margin-top: 8px;
    }
    .brief-header {
        background:
            radial-gradient(circle at 92% 18%, rgba(244,123,37,0.18), transparent 28%),
            linear-gradient(135deg, rgba(244,123,37,0.09), rgba(255,255,255,0.96));
        border-bottom: 1px solid var(--bl);
        padding: 18px 20px;
    }
    .brief-kicker {
        color: var(--or);
        font-size: 10px;
        font-weight: 900;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .brief-title {
        color: var(--t);
        font-size: 22px;
        line-height: 1.15;
        font-weight: 950;
        letter-spacing: -0.02em;
        margin-bottom: 8px;
    }
    .brief-summary-text {
        color: var(--t2);
        font-size: 13px;
        line-height: 1.55;
        max-width: 980px;
    }
    .brief-meta-strip {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 8px;
        margin-top: 14px;
    }
    .brief-meta-chip {
        background: rgba(255,255,255,0.82);
        border: 1px solid rgba(226,232,240,0.95);
        border-radius: 12px;
        padding: 10px 11px;
    }
    .brief-meta-label {
        color: var(--t3);
        font-size: 9px;
        font-weight: 900;
        letter-spacing: .9px;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .brief-meta-value {
        color: var(--t);
        font-size: 13px;
        line-height: 1.25;
        font-weight: 900;
    }
    .brief-section-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
        padding: 16px;
    }
    .brief-section-card {
        background: var(--sf);
        border: 1px solid var(--bl);
        border-radius: 14px;
        padding: 14px 15px;
        min-height: 150px;
        box-shadow: var(--sh);
    }
    .brief-section-card.primary {
        grid-column: 1 / -1;
        min-height: 0;
        border-color: var(--obr);
        background: linear-gradient(135deg, rgba(244,123,37,0.055), rgba(255,255,255,0.96));
    }
    .brief-section-title {
        color: var(--or);
        font-size: 11px;
        font-weight: 900;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin: 0 0 9px 0;
    }
    .brief-section-card p,
    .brief-box p {
        margin: 0 0 8px 0;
        font-size: 13px;
        color: var(--t2);
        line-height: 1.55;
    }
    .brief-list {
        margin: 0 0 0 18px;
        padding: 0;
    }
    .brief-list li {
        margin-bottom: 8px;
        color: var(--t2);
        font-size: 13px;
        line-height: 1.45;
    }
    .score-explain-card {
        background: var(--sf);
        border: 1px solid var(--bl);
        border-radius: var(--rl);
        padding: 14px 16px;
        box-shadow: var(--sh);
        margin: 8px 0;
    }
    .score-explain-head {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 12px;
        border-bottom: 1px solid var(--bl);
        padding-bottom: 10px;
        margin-bottom: 10px;
    }
    .score-explain-title {
        color: var(--t);
        font-size: 13px;
        font-weight: 900;
        margin-bottom: 3px;
    }
    .score-explain-meta {
        color: var(--t3);
        font-size: 10px;
        font-weight: 800;
        letter-spacing: .7px;
        text-transform: uppercase;
    }
    .score-number {
        min-width: 74px;
        text-align: center;
        border-radius: 12px;
        border: 1px solid var(--obr);
        background: var(--ob);
        color: var(--or);
        font-size: 24px;
        line-height: 1;
        font-weight: 950;
        padding: 9px 8px;
    }
    .score-number span {
        display: block;
        color: var(--t3);
        font-size: 9px;
        font-weight: 900;
        letter-spacing: .8px;
        margin-top: 3px;
    }
    .score-explain-label {
        color: var(--t3);
        font-size: 9px;
        font-weight: 900;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin: 8px 0 3px 0;
    }
    .score-explain-text {
        color: var(--t2);
        font-size: 12px;
        line-height: 1.5;
    }
    .brief-grounding {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 8px;
        margin-bottom: 12px;
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
    .run-monitor {
        background: var(--sf);
        border: 1px solid var(--obr);
        border-radius: 16px;
        padding: 14px 16px;
        box-shadow: 0 0 0 3px var(--og);
        margin: 8px 0 16px 0;
    }
    .run-monitor-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 10px;
    }
    .run-monitor-title {
        color: var(--t);
        font-size: 13px;
        font-weight: 900;
    }
    .run-monitor-sub {
        color: var(--t3);
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .run-progress-track {
        height: 7px;
        background: var(--s3);
        border-radius: 999px;
        overflow: hidden;
        margin: 8px 0 12px 0;
    }
    .run-progress-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--odk), var(--or), var(--olt));
        border-radius: 999px;
        transition: width var(--tr);
    }
    .run-status-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 8px;
    }
    .run-status-card {
        background: var(--s2);
        border: 1px solid var(--bl);
        border-radius: 12px;
        padding: 10px;
        min-height: 72px;
    }
    .run-status-name {
        font-size: 11px;
        font-weight: 900;
        color: var(--t);
        margin-bottom: 5px;
    }
    .run-status-detail {
        font-size: 10px;
        color: var(--t2);
        line-height: 1.35;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 2px 7px;
        border-radius: 999px;
        font-size: 9px;
        font-weight: 900;
        letter-spacing: .6px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .status-running { background: var(--ob); color: var(--or); border: 1px solid var(--obr); }
    .status-success { background: var(--gbg); color: var(--gr); border: 1px solid rgba(34,197,94,.2); }
    .status-failed { background: var(--rbg); color: var(--rd); border: 1px solid rgba(239,68,68,.2); }
    .status-skipped { background: rgba(100,116,139,.08); color: var(--t2); border: 1px solid var(--bl); }
    .status-queued { background: var(--s3); color: var(--t3); border: 1px solid var(--bl); }
    .audit-banner {
        background: linear-gradient(135deg, rgba(34,197,94,0.10), rgba(255,255,255,0.92));
        border: 1px solid rgba(34,197,94,0.22);
        border-left: 4px solid var(--gr);
        border-radius: var(--rl);
        padding: 14px 16px;
        color: var(--t);
        margin: 8px 0 14px 0;
        box-shadow: var(--sh);
    }
    .audit-banner.warn {
        background: linear-gradient(135deg, rgba(245,158,11,0.12), rgba(255,255,255,0.92));
        border-color: rgba(245,158,11,0.24);
        border-left-color: var(--am);
    }
    .audit-title {
        font-size: 13px;
        font-weight: 900;
        margin-bottom: 5px;
    }
    .audit-body {
        font-size: 12px;
        color: var(--t2);
        line-height: 1.5;
    }
    .audit-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
        margin: 10px 0 14px 0;
    }
    .audit-card {
        background: var(--sf);
        border: 1px solid var(--bl);
        border-radius: var(--rl);
        padding: 13px 14px;
        min-height: 92px;
        box-shadow: var(--sh);
    }
    .audit-label {
        color: var(--t3);
        font-size: 9px;
        font-weight: 900;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 7px;
    }
    .audit-value {
        color: var(--t);
        font-size: 18px;
        line-height: 1.1;
        font-weight: 900;
    }
    .audit-note {
        color: var(--t2);
        font-size: 11px;
        line-height: 1.35;
        margin-top: 7px;
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


def count_raw_records(raw_payload: Any, items: Optional[List[Dict[str, Any]]] = None) -> int:
    if items:
        return len(items)
    if raw_payload is None:
        return 0
    if isinstance(raw_payload, list):
        return len(raw_payload)
    if isinstance(raw_payload, dict):
        if isinstance(raw_payload.get("features"), list):
            return len(raw_payload["features"])
        if isinstance(raw_payload.get("results"), list):
            return len(raw_payload["results"])
        series = raw_payload.get("Results", {}).get("series") if isinstance(raw_payload.get("Results"), dict) else None
        if isinstance(series, list):
            return sum(len(s.get("data", [])) for s in series if isinstance(s, dict))
        nested_counts = [
            count_raw_records(value)
            for value in raw_payload.values()
            if isinstance(value, (dict, list))
        ]
        return sum(nested_counts) if nested_counts else 1
    return 1


def request_summary(source_key: str, run_config: Dict[str, Any]) -> str:
    if source_key == "gnews":
        keywords = run_config.get("news_keywords", [])
        return (
            f"{len(keywords)} keyword(s), country={run_config.get('country', '')}, "
            f"language={run_config.get('language', '')}, period={run_config.get('gnews_period', '')}, "
            f"max_results={run_config.get('max_news', '')}"
        )
    if source_key == "bls":
        return "series=" + ", ".join(BLS_CPI_SERIES.values())
    if source_key == "fda":
        return f"search={run_config.get('fda_query', '')}; limit={run_config.get('fda_limit', '')}"
    if source_key == "weather":
        return f"area={run_config.get('weather_area', '')}; limit={run_config.get('weather_limit', '')}; active NOAA alerts"
    if source_key == "apify":
        return (
            f"enabled={run_config.get('use_apify', False)}, token_present={run_config.get('apify_token_present', False)}, "
            f"run_mode={run_config.get('apify_run_mode', 'Skip Apify')}, confirmed={run_config.get('apify_live_confirm', False)}, "
            f"geo={run_config.get('apify_geo', '')}, time_range={APIFY_SAFE_TIME_RANGE}, "
            f"max_keywords={run_config.get('apify_max_keywords', APIFY_HARD_KEYWORD_LIMIT)}"
        )
    return ""


def build_collector_evidence(results: Dict[str, Dict[str, Any]], run_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    records = []
    enabled_sources = run_config.get("enabled_sources", {})
    for source_key in SOURCE_ORDER:
        result = results.get(source_key, {})
        status = result.get("status", "disabled" if not enabled_sources.get(source_key, False) else "not_run")
        normalized_rows = len(result.get("rows", []) or [])
        raw_records = count_raw_records(result.get("raw"), result.get("items"))
        live_request = status in {"success", "failed", "empty"} or (source_key == "apify" and result.get("raw") is not None)
        mock_used = bool(result.get("mock_used", False) or result.get("meta", {}).get("mock_used", False))
        records.append(
            {
                "source": SOURCE_LABELS[source_key],
                "status": status,
                "endpoint_or_actor": SOURCE_ENDPOINTS[source_key],
                "request_scope": request_summary(source_key, run_config),
                "raw_records_pulled": raw_records,
                "normalized_feature_rows": normalized_rows,
                "live_request_made": "Yes" if live_request else "No",
                "used_in_llm_payload": "Yes" if normalized_rows > 0 else "No",
                "mock_data_used": "Yes" if mock_used else "No",
                "analysis_method": ANALYSIS_METHODS[source_key],
                "error_or_note": result.get("error", "") or "",
            }
        )
    return records


def build_llm_payload(feature_df: pd.DataFrame, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "features": feature_df.to_dict(orient="records")[:12],
        "articles": articles[:8],
    }


def llm_system_prompt() -> str:
    return (
        "You are a retail market intelligence analyst. Ground your answer only in the supplied signals. "
        "Write concise executive guidance for buyers, category managers, demand planners, and supply-chain teams."
    )


def llm_user_prompt(retailer: str, region: str, payload: Dict[str, Any]) -> str:
    return f"""
    Retailer: {retailer}
    Region: {region}

    Signal payload:
    {json.dumps(payload, indent=2, default=str)[:9000]}

    Produce:
    1. Executive summary in 2-3 sentences
    2. Top 3 insights, and for each one cite signal_area, source, risk_score, and score_reason
    3. Forecasting relevance, clearly stating how the signal can become a feature
    4. Recommended buyer/category/demand-planning actions
    5. Confidence and limitations

    Rules:
    - Ground every claim only in the supplied payload.
    - Do not invent internal sales, POS, inventory, margin, or category-performance facts.
    - If a signal is missing, say it is missing instead of estimating it.
    - Explain why each score matters; do not only repeat the number.
    """


def payload_hash(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def build_base_llm_audit(
    feature_df: pd.DataFrame,
    articles: List[Dict[str, Any]],
    retailer: str,
    region: str,
    model: str,
) -> Dict[str, Any]:
    payload = build_llm_payload(feature_df, articles)
    return {
        "provider": "NVIDIA",
        "model": model,
        "sent_to_llm": False,
        "brief_source": "not_generated",
        "fallback_used": False,
        "fallback_reason": "",
        "mock_data_used": False,
        "feature_rows_available": int(len(feature_df)),
        "feature_rows_sent": int(len(payload["features"])),
        "articles_available": int(len(articles)),
        "articles_sent": int(len(payload["articles"])),
        "payload_hash_sha256": payload_hash(payload),
        "system_prompt": llm_system_prompt(),
        "user_prompt": llm_user_prompt(retailer, region, payload),
        "payload": payload,
        "analysis_contract": "The brief must be grounded only in the collected feature rows and article records shown in this audit view.",
    }


def any_mock_used(results: Dict[str, Dict[str, Any]], llm_audit: Dict[str, Any]) -> bool:
    collector_mock = any(
        bool(result.get("mock_used", False) or result.get("meta", {}).get("mock_used", False))
        for result in results.values()
    )
    return collector_mock or bool(llm_audit.get("mock_data_used", False))


def render_audit_banner(mock_used: bool, brief_source: str) -> None:
    banner_class = "audit-banner warn" if mock_used else "audit-banner"
    title = "Mock Data Detected" if mock_used else "No Mock Data Used In This Run"
    body = (
        "At least one collector or analysis step is marked as using mock data. Review the evidence table below."
        if mock_used
        else f"Every feature row shown below comes from the collector outputs for this run. Brief source: {brief_source}."
    )
    st.markdown(
        f"<div class='{banner_class}'><div class='audit-title'>{escape(title)}</div><div class='audit-body'>{escape(body)}</div></div>",
        unsafe_allow_html=True,
    )


def render_audit_card(label: str, value: str, note: str) -> None:
    st.markdown(
        "<div class='audit-card'>"
        f"<div class='audit-label'>{escape(label)}</div>"
        f"<div class='audit-value'>{escape(value)}</div>"
        f"<div class='audit-note'>{escape(note)}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


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


def build_cpi_signal(data: Dict[str, Any], label: str, series_id: str, retailer: str) -> Tuple[Optional[Dict[str, Any]], pd.DataFrame]:
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
    mom_label = "unavailable" if pd.isna(mom) else f"{float(mom):.2f}% MoM"
    yoy_label = "unavailable" if pd.isna(yoy) else f"{float(yoy):.2f}% YoY"
    signal_name = "inflation_pressure_score" if label == "Headline CPI" else f"{label.lower().replace(' ', '_')}_cpi_pressure_score"
    signal = {
        "date": f"{int(latest['year'])}-{str(latest['period']).replace('M', '').zfill(2)}",
        "retailer": retailer,
        "region": "US",
        "region_scope": "national",
        "source": "BLS CPI",
        "signal_area": "Inflation" if label == "Headline CPI" else "Category CPI",
        "signal_name": signal_name,
        "signal_value": round(float(score), 2),
        "risk_score": round(float(score), 2),
        "confidence": "High",
        "score_reason": f"{label} CPI latest value {latest['cpi_value']}; change is {mom_label} and {yoy_label}. Score rises with monthly inflation pressure and elevated YoY inflation.",
        "business_impact": f"{label} inflation can affect price sensitivity, category demand, and basket mix.",
        "recommended_action": "Use category CPI as an external regressor and validate against internal category sales.",
        "raw_reference": f"{label}: CPI {latest['cpi_value']}",
    }
    return signal, df


def collect_bls_cpi(bls_key: str = "", retailer: str = "Dollar Tree") -> Dict[str, Any]:
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
        signal, df = build_cpi_signal(series_payload, label, series_id, retailer)
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


def collect_fda_recalls(query: str, limit: int, retailer: str = "Dollar Tree") -> Dict[str, Any]:
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
    states = sorted({str(x.get("state", "")).strip() for x in items if str(x.get("state", "")).strip()})
    upc_count = sum(1 for x in items if x.get("upcs"))
    top_risk_type = max(items, key=lambda x: x["risk_score"]).get("risk_type", "none") if items else "none"
    signal = {
        "date": utc_now()[:10],
        "retailer": retailer,
        "region": "US",
        "region_scope": "national_with_state_records",
        "source": "openFDA",
        "signal_area": "Product Recalls",
        "signal_name": "recall_risk_score",
        "signal_value": len(items),
        "risk_score": round(aggregate_score, 2),
        "confidence": "High",
        "score_reason": f"Score uses highest adjusted recall severity. Inputs: {len(items)} records, {ongoing_count} ongoing, {class_i_count} Class I, {upc_count} records with UPCs, top risk type {top_risk_type}.",
        "affected_states": ", ".join(states[:8]) if states else "Unknown",
        "ongoing_count": ongoing_count,
        "class_i_count": class_i_count,
        "upc_record_count": upc_count,
        "sku_match_status": "unknown",
        "affected_category": "Food / snacks / candy / beverages",
        "business_impact": "Food and beverage recalls can trigger inventory withdrawal, substitution demand, and safety review.",
        "recommended_action": f"Prioritize ongoing and Class I recalls, then match UPCs against {retailer} inventory before store-level action.",
        "raw_reference": f"{len(items)} recall records; {ongoing_count} ongoing; {class_i_count} Class I",
    }
    rows.append(signal)
    return {"status": "success", "source": "openFDA", "error": "", "raw": data, "rows": rows, "items": items}


def normalize_weather_area(area: str) -> str:
    candidate = re.sub(r"[^A-Za-z]", "", area or "").upper()
    if len(candidate) == 2:
        return candidate
    return "TX"


def weather_alert_weight(severity: str, urgency: str, certainty: str) -> float:
    severity_score = {
        "extreme": 5.0,
        "severe": 3.0,
        "moderate": 2.0,
        "minor": 1.0,
        "unknown": 1.0,
    }.get(str(severity or "").lower(), 1.0)
    urgency_bonus = {
        "immediate": 1.5,
        "expected": 0.75,
    }.get(str(urgency or "").lower(), 0.0)
    certainty_bonus = {
        "observed": 0.5,
        "likely": 0.5,
    }.get(str(certainty or "").lower(), 0.0)
    return severity_score + urgency_bonus + certainty_bonus


def collect_weather_alerts(area: str, limit: int, retailer: str = "Dollar Tree") -> Dict[str, Any]:
    state_area = normalize_weather_area(area)
    params = {"area": state_area}
    ok, data, msg = safe_request(
        "https://api.weather.gov/alerts/active",
        headers={"User-Agent": "MarketIntelligenceWorkbench/1.0", "Accept": "application/geo+json"},
        params=params,
        timeout=30,
    )
    if not ok:
        return {"status": "failed", "source": "NOAA Weather Alerts", "error": msg, "raw": None, "rows": [], "items": []}
    features = data.get("features", []) if isinstance(data, dict) else []
    selected_alerts = features[: max(1, int(limit))]
    items = []
    score_components = []
    severe_count = 0
    extreme_count = 0
    for alert in selected_alerts:
        props = alert.get("properties", {}) if isinstance(alert, dict) else {}
        severity = props.get("severity", "Unknown")
        urgency = props.get("urgency", "Unknown")
        certainty = props.get("certainty", "Unknown")
        component = weather_alert_weight(severity, urgency, certainty)
        score_components.append(component)
        severity_text = str(severity or "").lower()
        if severity_text == "extreme":
            extreme_count += 1
        if severity_text in {"severe", "extreme"}:
            severe_count += 1
        items.append(
            {
                "event": props.get("event", ""),
                "severity": severity,
                "urgency": urgency,
                "certainty": certainty,
                "headline": props.get("headline", ""),
                "area_desc": props.get("areaDesc", ""),
                "effective": props.get("effective", ""),
                "expires": props.get("expires", ""),
                "instruction": props.get("instruction", ""),
                "risk_component": round(component, 2),
            }
        )
    risk_score = round(min(10.0, sum(score_components)), 2) if items else 0.0
    if items:
        score_reason = (
            f"Score sums weighted active NOAA alerts for {state_area}, capped at 10. "
            f"Inputs: {len(items)} alert(s), {severe_count} severe/extreme, {extreme_count} extreme; "
            f"severity/urgency/certainty components total {sum(score_components):.2f}."
        )
        raw_reference = f"{len(items)} active NOAA alert(s); top event: {items[0].get('event') or 'Unknown'}"
        recommended_action = "Check affected counties against store and DC routes; use alert severity as a short-horizon disruption and emergency-demand feature."
    else:
        score_reason = f"NOAA returned 0 active alerts for {state_area}. Score is 0 because no current weather disruption signal is present."
        raw_reference = "0 active NOAA alerts"
        recommended_action = "Keep weather feature at baseline for this state, then refresh before short-horizon replenishment decisions."
    signal = {
        "date": utc_now()[:10],
        "retailer": retailer,
        "region": state_area,
        "region_scope": "state_weather_alerts",
        "source": "NOAA Weather Alerts",
        "signal_area": "Weather Risk",
        "signal_name": "supply_chain_weather_risk_score",
        "signal_value": len(items),
        "risk_score": risk_score,
        "confidence": "High",
        "score_reason": score_reason,
        "alert_count": len(items),
        "severe_or_extreme_count": severe_count,
        "extreme_count": extreme_count,
        "business_impact": "Active weather alerts can disrupt store traffic, DC-to-store routes, staffing, replenishment timing, and emergency-demand categories.",
        "recommended_action": recommended_action,
        "raw_reference": raw_reference,
    }
    return {"status": "success", "source": "NOAA Weather Alerts", "error": "", "raw": data, "rows": [signal], "items": items}


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


def collect_gnews(keywords: List[str], country: str, language: str, period: str, max_results: int, retailer: str = "Dollar Tree") -> Dict[str, Any]:
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

    if not deduped:
        return {
            "status": "failed" if errors else "empty",
            "source": "GNews",
            "error": "; ".join(errors) if errors else "No meaningful articles returned for selected keywords.",
            "raw": [],
            "rows": [],
            "items": [],
        }

    score = round(float(pd.Series([a["risk_score"] for a in deduped]).mean()), 2) if deduped else 1.0
    negative_count = sum(1 for article in deduped if article.get("sentiment") == "negative")
    high_conf_count = sum(1 for article in deduped if article.get("confidence") == "High")
    event_counts = pd.Series([article.get("event_type", "unknown") for article in deduped]).value_counts().to_dict()
    signal = {
        "date": utc_now()[:10],
        "retailer": retailer,
        "region": country.upper(),
        "region_scope": "country_news",
        "source": "GNews / Google News RSS",
        "signal_area": "Retail News",
        "signal_name": "news_risk_score",
        "signal_value": len(deduped),
        "risk_score": score,
        "confidence": "Medium",
        "score_reason": f"Average article risk across {len(deduped)} deduped articles; {negative_count} negative articles; {high_conf_count} high-confidence publishers; event mix {event_counts}.",
        "business_impact": "Recent news can reveal competitor moves, pricing pressure, recalls, store changes, and supply-chain risk.",
        "recommended_action": "Review high-risk articles and use NVIDIA classification before executive distribution.",
        "raw_reference": f"{len(deduped)} articles",
    }
    return {"status": "success", "source": "GNews", "error": "; ".join(errors), "raw": deduped, "rows": [signal], "items": deduped}


def get_apify_value(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    if hasattr(obj, key):
        return getattr(obj, key)
    try:
        return obj[key]
    except (TypeError, KeyError, AttributeError):
        return None


def collect_apify_trends(
    token: str,
    keywords: List[str],
    geo: str,
    time_range: str,
    retailer: str = "Dollar Tree",
    max_keywords: int = APIFY_HARD_KEYWORD_LIMIT,
) -> Dict[str, Any]:
    if not token:
        return {"status": "skipped", "source": "Apify Trends", "error": "No Apify token provided.", "raw": None, "rows": [], "items": []}
    try:
        from apify_client import ApifyClient
    except ImportError:
        return {"status": "failed", "source": "Apify Trends", "error": "apify-client is not installed.", "raw": None, "rows": [], "items": []}
    try:
        client = ApifyClient(token)
        safe_max_keywords = min(max(1, int(max_keywords)), APIFY_HARD_KEYWORD_LIMIT)
        safe_time_range = APIFY_SAFE_TIME_RANGE
        selected_keywords = [kw for kw in keywords if kw][:safe_max_keywords]
        if not selected_keywords:
            return {"status": "skipped", "source": "Apify Trends", "error": "No trend keywords provided.", "raw": None, "rows": [], "items": []}
        run_input = {"geo": geo, "searchTerms": selected_keywords, "timeRange": safe_time_range}
        run = client.actor("apify/google-trends-scraper").call(run_input=run_input)
        dataset_id = get_apify_value(run, "defaultDatasetId") or get_apify_value(run, "default_dataset_id")
        if not dataset_id:
            return {
                "status": "failed",
                "source": "Apify Trends",
                "error": "Apify run completed but no default dataset ID was found.",
                "raw": run_input,
                "rows": [],
                "items": [],
            }
        items = list(client.dataset(dataset_id).iterate_items())
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
        "retailer": retailer,
        "region": geo,
        "region_scope": "trend_geo",
        "source": "Apify Google Trends",
        "signal_area": "Search Demand",
        "signal_name": "search_demand_score",
        "signal_value": top_score,
        "risk_score": signal_score,
        "confidence": "Medium",
        "score_reason": f"Score is top regional Google Trends interest divided by 10. Run hard-limited to {len(selected_keywords)} keyword(s) over {safe_time_range} to control Apify quota and memory.",
        "business_impact": "Search interest can reveal early demand shifts, promotional interest, and seasonal spikes.",
        "recommended_action": "Compare rising regions against sales, inventory, and competitor promotion calendars.",
        "raw_reference": f"{len(region_rows)} regional trend rows",
    }
    return {"status": "success", "source": "Apify Trends", "error": "", "raw": items, "rows": [signal], "items": region_rows}


def generate_fallback_brief(feature_df: pd.DataFrame, retailer: str, region: str) -> str:
    if feature_df.empty:
        return (
            "Executive Summary:\n"
            "- No external signals were collected for this run.\n"
            "- Enable at least one source and run the workbench again before using the output for planning.\n\n"
            "Confidence And Limitations:\n"
            "- No score can be explained because no feature rows exist."
        )
    strongest = feature_df.sort_values("risk_score", ascending=False).head(3)
    avg_score = round(float(feature_df["risk_score"].mean()), 2)
    top = strongest.iloc[0]
    sources = ", ".join(sorted({str(src) for src in feature_df["source"].dropna().tolist()}))
    lines = [
        "Executive Summary:",
        f"- {retailer} in {region} has {risk_band(avg_score).lower()} external signal intensity with an average score of {avg_score}/10 across {len(feature_df)} forecast-ready row(s).",
        f"- The strongest current signal is {top['signal_area']} at {float(top['risk_score']):.2f}/10 from {top['source']}.",
        f"- This brief is grounded in collected source output only: {sources}.",
        "",
        "Top Signal Evidence:",
    ]
    for idx, (_, row) in enumerate(strongest.iterrows(), start=1):
        lines.append(
            f"{idx}. {row['signal_area']} scored {float(row['risk_score']):.2f}/10 from {row['source']}. Why: {row.get('score_reason', 'No score reason available.')}"
        )
        lines.append(f"- Business impact: {row.get('business_impact', 'No business impact available.')}")
    lines.extend(
        [
            "",
            "Forecasting Relevance:",
            "- Treat each score as an external regressor candidate, not as a final demand forecast.",
            "- Join these rows to internal POS, category, store, promotion, and inventory data before model training or operational action.",
            "",
            "Recommended Actions:",
        ]
    )
    for idx, (_, row) in enumerate(strongest.iterrows(), start=1):
        lines.append(f"{idx}. {row.get('recommended_action', 'Review this signal with the category owner.')}")
    lines.extend(
        [
            "",
            "Confidence And Limitations:",
            "- Scores are explainable directional signals from public API data, not proof of actual Dollar Tree demand movement.",
            "- Category performance still requires internal sales/POS data; external APIs explain context but do not replace internal performance data.",
        ]
    )
    return "\n".join(lines)


def generate_nvidia_brief(
    api_key: str,
    model: str,
    feature_df: pd.DataFrame,
    articles: List[Dict[str, Any]],
    retailer: str,
    region: str,
) -> Tuple[str, str, Dict[str, Any]]:
    audit = build_base_llm_audit(feature_df, articles, retailer, region, model)
    if not api_key:
        audit.update(
            {
                "provider": "Local deterministic fallback",
                "model": "rule_based_summary",
                "brief_source": "fallback",
                "fallback_used": True,
                "fallback_reason": "No NVIDIA API key provided. No external LLM call was made.",
            }
        )
        return generate_fallback_brief(feature_df, retailer, region), "fallback", audit
    ok, data, msg = safe_request(
        NVIDIA_CHAT_URL,
        method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json_body={
            "model": model,
            "messages": [{"role": "system", "content": audit["system_prompt"]}, {"role": "user", "content": audit["user_prompt"]}],
            "temperature": 0.25,
            "max_tokens": 900,
        },
        timeout=60,
    )
    if not ok:
        audit.update(
            {
                "provider": "Local deterministic fallback",
                "model": "rule_based_summary",
                "brief_source": f"fallback: {msg}",
                "fallback_used": True,
                "fallback_reason": f"NVIDIA request failed: {msg}",
            }
        )
        return generate_fallback_brief(feature_df, retailer, region), f"fallback: {msg}", audit
    content = data.get("choices", [{}])[0].get("message", {}).get("content")
    content_text = str(content or "").strip()
    if not content_text:
        audit.update(
            {
                "provider": "Local deterministic fallback",
                "model": "rule_based_summary",
                "brief_source": "fallback: empty NVIDIA response",
                "fallback_used": True,
                "fallback_reason": "NVIDIA returned an empty response.",
            }
        )
        return generate_fallback_brief(feature_df, retailer, region), "fallback: empty NVIDIA response", audit
    audit.update(
        {
            "provider": "NVIDIA",
            "model": model,
            "sent_to_llm": True,
            "brief_source": "nvidia",
            "fallback_used": False,
            "fallback_reason": "",
            "response_chars": len(content_text),
        }
    )
    return content_text, "nvidia", audit


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
        area_scores.get("Weather Risk", 0.0),
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
                "body": f"{row['recommended_action']} Reason: {row.get('score_reason', 'No score reason available.')}",
            }
        )
    apify_result = results.get("apify")
    if apify_result and apify_result.get("status") in {"failed", "skipped"}:
        actions.append(
            {
                "label": "Apify",
                "title": "Search demand not collected",
                "body": apify_result.get("error") or "Apify did not return a usable trends signal. Keep MVP on public sources or run one guarded live query.",
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


def render_run_monitor(slot: Any, title: str, states: Dict[str, Dict[str, str]], progress_pct: int) -> None:
    cards = []
    for name, info in states.items():
        status = info.get("status", "queued")
        detail = info.get("detail", "")
        status_class = {
            "running": "status-running",
            "success": "status-success",
            "failed": "status-failed",
            "skipped": "status-skipped",
            "queued": "status-queued",
        }.get(status, "status-queued")
        cards.append(
            "<div class='run-status-card'>"
            f"<div class='status-badge {status_class}'>{escape(status)}</div>"
            f"<div class='run-status-name'>{escape(name)}</div>"
            f"<div class='run-status-detail'>{escape(detail)}</div>"
            "</div>"
        )
    html = (
        "<div class='run-monitor'>"
        "<div class='run-monitor-head'>"
        f"<div><div class='run-monitor-sub'>Pipeline Status</div><div class='run-monitor-title'>{escape(title)}</div></div>"
        f"<div class='tbadge'>{int(progress_pct)}%</div>"
        "</div>"
        "<div class='run-progress-track'>"
        f"<div class='run-progress-fill' style='width:{max(0, min(100, int(progress_pct)))}%;'></div>"
        "</div>"
        "<div class='run-status-grid'>"
        + "".join(cards)
        + "</div></div>"
    )
    slot.markdown(html, unsafe_allow_html=True)


def render_sidebar_status(slot: Any, message: str, state: str = "info") -> None:
    if state == "success":
        slot.success(message)
    elif state == "warning":
        slot.warning(message)
    elif state == "error":
        slot.error(message)
    else:
        slot.info(message)


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


def scoring_formula_for_row(row: pd.Series) -> str:
    source = str(row.get("source", "")).lower()
    area = str(row.get("signal_area", "")).lower()
    if "bls" in source or "cpi" in area:
        return "CPI scoring starts from a neutral 4.0, adjusts upward or downward using monthly CPI change, adds pressure when YoY inflation is elevated, then clips to a 1-10 range."
    if "fda" in source or "recall" in area:
        return "Recall scoring starts from reason severity, then adjusts for FDA classification and recall status. Class I and ongoing recalls increase the score; terminated recalls reduce it."
    if "noaa" in source or "weather" in area:
        return "Weather scoring sums active NOAA alert severity weights for the selected state, adds urgency/certainty pressure, then caps the supply-chain risk score at 10."
    if "gnews" in source or "news" in area:
        return "News scoring classifies each article into event type and sentiment, adjusts for recency/source quality, then averages deduplicated article risk."
    if "apify" in source or "search" in area:
        return f"Search scoring uses top regional Google Trends interest divided by 10, with backend limits of {APIFY_SAFE_TIME_RANGE} and {APIFY_HARD_KEYWORD_LIMIT} keyword(s)."
    return "Score is normalized to a 1-10 signal intensity scale using the collector-specific scoring rule."


def render_score_explainability(feature_df: pd.DataFrame) -> None:
    if feature_df.empty:
        st.info("No signal explanations available.")
        return
    explanation_rows = feature_df.sort_values("risk_score", ascending=False).reset_index(drop=True)
    for _, row in explanation_rows.iterrows():
        score = float(row.get("risk_score", 0) or 0)
        band = risk_band(score)
        title = f"{row.get('signal_area', 'Signal')} · {str(row.get('signal_name', '')).replace('_', ' ').title()}"
        meta = f"{row.get('source', 'Unknown source')} / {row.get('region_scope', row.get('region', ''))}"
        reason = str(row.get("score_reason") or "No score reason was returned by this collector.")
        evidence = str(row.get("raw_reference") or row.get("signal_value") or "No raw reference available.")
        action = str(row.get("recommended_action") or "Review this signal before using it in planning.")
        formula = scoring_formula_for_row(row)
        html = (
            "<div class='score-explain-card'>"
            "<div class='score-explain-head'>"
            f"<div><div class='score-explain-title'>{escape(title)}</div><div class='score-explain-meta'>{escape(meta)}</div></div>"
            f"<div class='score-number'>{score:.2f}<span>{escape(band)}</span></div>"
            "</div>"
            "<div class='score-explain-label'>Scoring rule</div>"
            f"<div class='score-explain-text'>{escape(formula)}</div>"
            "<div class='score-explain-label'>Why this score</div>"
            f"<div class='score-explain-text'>{escape(reason)}</div>"
            "<div class='score-explain-label'>Evidence used</div>"
            f"<div class='score-explain-text'>{escape(evidence)}</div>"
            "<div class='score-explain-label'>Planning action</div>"
            f"<div class='score-explain-text'>{escape(action)}</div>"
            "</div>"
        )
        st.markdown(html, unsafe_allow_html=True)


def brief_to_html(brief: str) -> str:
    parts: List[str] = []
    in_list = False
    for raw_line in str(brief or "").splitlines():
        line = raw_line.strip()
        if not line:
            if in_list:
                parts.append("</ul>")
                in_list = False
            continue
        normalized = re.sub(r"^\*\*(.*?)\*\*$", r"\1", line)
        normalized = normalized.replace("**", "")
        is_header = normalized.endswith(":") and len(normalized) <= 90 and not re.match(r"^[-\d]", normalized)
        bullet_match = re.match(r"^(?:[-*]\s+|\d+\.\s+)(.*)$", normalized)
        if is_header:
            if in_list:
                parts.append("</ul>")
                in_list = False
            parts.append(f"<div class='brief-section-title'>{escape(normalized.rstrip(':'))}</div>")
        elif bullet_match:
            if not in_list:
                parts.append("<ul class='brief-list'>")
                in_list = True
            parts.append(f"<li>{escape(bullet_match.group(1))}</li>")
        else:
            if in_list:
                parts.append("</ul>")
                in_list = False
            parts.append(f"<p>{escape(normalized)}</p>")
    if in_list:
        parts.append("</ul>")
    return "".join(parts)


def parse_brief_sections(brief: str) -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []
    current = {"title": "Executive Summary", "items": []}
    for raw_line in str(brief or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        normalized = re.sub(r"^\*\*(.*?)\*\*$", r"\1", line).replace("**", "")
        is_header = normalized.endswith(":") and len(normalized) <= 90 and not re.match(r"^[-\d]", normalized)
        bullet_match = re.match(r"^(?:[-*]\s+|\d+\.\s+)(.*)$", normalized)
        if is_header:
            if current["items"]:
                sections.append(current)
            current = {"title": normalized.rstrip(":"), "items": []}
        elif bullet_match:
            current["items"].append({"kind": "bullet", "text": bullet_match.group(1)})
        else:
            current["items"].append({"kind": "text", "text": normalized})
    if current["items"]:
        sections.append(current)
    return sections


def brief_sections_to_html(brief: str) -> str:
    sections = parse_brief_sections(brief)
    if not sections:
        return "<div class='brief-section-grid'><div class='brief-section-card primary'><div class='brief-section-title'>Executive Summary</div><p>No brief content was generated.</p></div></div>"
    cards = []
    for idx, section in enumerate(sections):
        paragraphs = []
        bullets = []
        for item in section["items"]:
            if item["kind"] == "bullet":
                bullets.append(f"<li>{escape(str(item['text']))}</li>")
            else:
                paragraphs.append(f"<p>{escape(str(item['text']))}</p>")
        body = "".join(paragraphs)
        if bullets:
            body += "<ul class='brief-list'>" + "".join(bullets) + "</ul>"
        primary = " primary" if idx == 0 else ""
        cards.append(
            f"<div class='brief-section-card{primary}'>"
            f"<div class='brief-section-title'>{escape(str(section['title']))}</div>"
            f"{body}"
            "</div>"
        )
    return "<div class='brief-section-grid'>" + "".join(cards) + "</div>"


def render_executive_brief(run: Dict[str, Any], feature_df: pd.DataFrame) -> None:
    brief_source = str(run.get("brief_source", "unknown"))
    articles = run.get("articles", [])
    llm_audit = run.get("llm_audit", {})
    source_count = feature_df["source"].nunique() if not feature_df.empty and "source" in feature_df.columns else 0
    top_label = "No signal"
    avg_score_label = "0.00"
    if not feature_df.empty:
        top = feature_df.sort_values("risk_score", ascending=False).iloc[0]
        top_label = f"{top.get('signal_area', 'Signal')} {float(top.get('risk_score', 0) or 0):.2f}/10"
        avg_score_label = f"{float(feature_df['risk_score'].mean()):.2f}"
    articles_sent = llm_audit.get("articles_sent", min(len(articles), 8))
    header_note = (
        "NVIDIA grounded response" if llm_audit.get("sent_to_llm") else "Local deterministic summary using collected feature rows"
    )
    html = (
        "<div class='brief-shell'>"
        "<div class='brief-header'>"
        "<div class='brief-kicker'>Executive Brief</div>"
        f"<div class='brief-title'>{escape(str(run.get('run_config', {}).get('retailer', retailer_label)))} External Signal Readout</div>"
        f"<div class='brief-summary-text'>{escape(header_note)}. The section cards below are grounded in collector rows and score reasons from this run.</div>"
        "<div class='brief-meta-strip'>"
        f"<div class='brief-meta-chip'><div class='brief-meta-label'>Brief Source</div><div class='brief-meta-value'>{escape(brief_source)}</div></div>"
        f"<div class='brief-meta-chip'><div class='brief-meta-label'>Rows Grounded</div><div class='brief-meta-value'>{len(feature_df)} rows / {source_count} source(s)</div></div>"
        f"<div class='brief-meta-chip'><div class='brief-meta-label'>Average Score</div><div class='brief-meta-value'>{escape(avg_score_label)} / 10</div></div>"
        f"<div class='brief-meta-chip'><div class='brief-meta-label'>Top Evidence</div><div class='brief-meta-value'>{escape(top_label)}</div></div>"
        "</div>"
        f"<div class='brief-summary-text' style='margin-top:10px;'>Articles used in brief context: {len(articles)} available, {articles_sent} sent.</div>"
        "</div>"
        f"{brief_sections_to_html(str(run.get('brief', '')))}"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def parse_lines(text: str) -> List[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def build_default_news_keywords(retailer_name: str) -> str:
    name = retailer_name.strip() or "Retailer"
    return "\n".join(
        [
            f"{name} inflation",
            f"{name} prices",
            f"{name} store closures",
            f"{name} recall",
            "discount retail tariffs",
            "Dollar General promotion",
        ]
    )


def build_default_trends_keywords(retailer_name: str) -> str:
    name = retailer_name.strip() or "Retailer"
    return "\n".join(
        [
            f"{name} sales",
            f"{name} coupons",
            f"{name} near me",
            f"{name} groceries",
            "cheap groceries",
        ]
    )


def retailer_initials(retailer_name: str) -> str:
    words = [word for word in re.split(r"\s+", retailer_name.strip()) if word]
    if not words:
        return "AI"
    return "".join(word[0].upper() for word in words[:2])


st.session_state.setdefault("gnews_period", "7d")
st.session_state.setdefault("max_news", 24)
st.session_state.setdefault("fda_query", "product_description:(snacks OR candy OR beverages)")
st.session_state.setdefault("fda_limit", 8)
st.session_state.setdefault("weather_area", "TX")
st.session_state.setdefault("weather_limit", 5)
st.session_state.setdefault("apify_geo", "US")
st.session_state.setdefault("apify_time_range", APIFY_SAFE_TIME_RANGE)
st.session_state.setdefault("apify_max_keywords", APIFY_HARD_KEYWORD_LIMIT)
st.session_state.setdefault("apify_run_mode", "Skip Apify")
st.session_state.setdefault("apify_live_confirm", False)
st.session_state.setdefault("workbench_view", "Configure")
if st.session_state.pop("force_results_view", False):
    st.session_state["workbench_view"] = "Results"
if st.session_state.pop("reset_apify_live_confirm", False):
    st.session_state["apify_run_mode"] = "Skip Apify"
    st.session_state["apify_live_confirm"] = False


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
    use_weather = st.checkbox("Weather risk via NOAA", value=True)
    use_apify = st.checkbox("Search demand via Apify Trends", value=False)
    if use_apify or apify_token.strip():
        st.markdown('<div class="small-header">Apify Guardrail</div>', unsafe_allow_html=True)
        if apify_token.strip():
            if st.session_state.get("apify_live_confirm", False) and st.session_state.get("apify_run_mode") == "Skip Apify":
                st.session_state["apify_run_mode"] = "Run one live Apify call"
            st.radio(
                "Apify live mode",
                ["Skip Apify", "Run one live Apify call"],
                key="apify_run_mode",
                horizontal=False,
            )
            st.session_state["apify_live_confirm"] = st.session_state.get("apify_run_mode") == "Run one live Apify call"
            st.caption(f"Guarded live mode: {APIFY_SAFE_TIME_RANGE}, max {APIFY_HARD_KEYWORD_LIMIT} keyword(s). This resets after the run.")
            if st.session_state.get("apify_run_mode") == "Run one live Apify call":
                st.success("Next run will call Apify once.")
            else:
                st.info("Apify mode is Skip Apify. Select Run one live Apify call to collect trends.")
        else:
            st.session_state["apify_run_mode"] = "Skip Apify"
            st.session_state["apify_live_confirm"] = False
            st.warning("Add an Apify token before allowing a live trends run.")

    st.markdown('<div class="small-header">Run Controls</div>', unsafe_allow_html=True)
    validate_button = st.button("Validate credentials", width="stretch")
    run_button = st.button("Run intelligence", type="primary", width="stretch")
    sidebar_status_slot = st.empty()


retailer_label = retailer.strip() or "Retailer"
previous_keyword_retailer = st.session_state.get("keyword_template_retailer")
previous_news_template = build_default_news_keywords(previous_keyword_retailer or retailer_label)
previous_trends_template = build_default_trends_keywords(previous_keyword_retailer or retailer_label)
next_news_template = build_default_news_keywords(retailer_label)
next_trends_template = build_default_trends_keywords(retailer_label)
if "news_keywords_text" not in st.session_state or st.session_state.get("news_keywords_text") == previous_news_template:
    st.session_state["news_keywords_text"] = next_news_template
if "trends_keywords_text" not in st.session_state or st.session_state.get("trends_keywords_text") == previous_trends_template:
    st.session_state["trends_keywords_text"] = next_trends_template
st.session_state["keyword_template_retailer"] = retailer_label
apify_source_active = bool(use_apify or (apify_token.strip() and st.session_state.get("apify_run_mode") == "Run one live Apify call"))

st.markdown('<div class="accent-bar"></div>', unsafe_allow_html=True)
st.markdown(
    "<div class='topbar'>"
    f"<div class='tt'>Market Intelligence <span>/ {escape(retailer_label)} · External Signals</span></div>"
    "<div class='tbadge'>AI Workbench</div>"
    "<div style='margin-left:auto;display:flex;align-items:center;gap:8px;'>"
    "<span class='ldot'></span>"
    "<span style='font-size:10px;color:var(--t3);font-weight:800;'>Live API Mode</span>"
    f"<div style='width:28px;height:28px;border-radius:7px;background:linear-gradient(135deg,var(--odk),var(--or));display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:900;color:#fff;'>{escape(retailer_initials(retailer_label))}</div>"
    "</div></div>",
    unsafe_allow_html=True,
)

hero_left, hero_right = st.columns([2.2, 0.9], vertical_alignment="center")
with hero_left:
    st.markdown(
        "<div class='hero-shell'>"
        "<div class='hero-kicker'><span class='ldot'></span> External Signal Layer</div>"
        f"<h1 class='hero-title'>{escape(retailer_label)} Market Intelligence Command Center</h1>"
        "<p class='hero-copy'>A retail-grade workbench that turns news, CPI, recalls, weather alerts, search demand, and API health into forecast-ready features, composite risk scores, and buyer actions.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
with hero_right:
    active_sources = sum([use_gnews, use_bls, use_fda, use_weather, apify_source_active])
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
    ["Configure", "Results", "Evidence Audit", "Raw Data"],
    required=True,
    label_visibility="collapsed",
    key="workbench_view",
    width="content",
)

run_status_slot = st.empty()
if not run_button and st.session_state.get("run"):
    last_run = st.session_state["run"]
    last_states = {}
    for key, result in last_run.get("results", {}).items():
        status = result.get("status", "unknown")
        last_states[key.upper()] = {
            "status": status if status in {"success", "failed", "skipped"} else "queued",
            "detail": result.get("error") or f"{len(result.get('rows', []))} feature row(s)",
        }
    if last_states:
        render_run_monitor(run_status_slot, f"Last run completed at {last_run.get('timestamp', '')}", last_states, 100)
    render_sidebar_status(sidebar_status_slot, f"Last run complete: {last_run.get('timestamp', '')}", "success")
elif not run_button:
    render_sidebar_status(sidebar_status_slot, "Status: idle. Apify runs only when token is present and live mode is set to Run one live Apify call.", "info")

if view == "Configure":
    st.markdown('<div class="small-header">Signal Source Stack</div>', unsafe_allow_html=True)
    src_cols = st.columns(5)
    source_specs = [
        ("GNews/RSS", "Active" if use_gnews else "Off", "Retail articles, sentiment hints, source confidence."),
        ("BLS CPI", "Active" if use_bls else "Off", "Headline and category inflation pressure features."),
        ("openFDA", "Active" if use_fda else "Off", "Recall severity, UPC extraction, SKU-match prep."),
        ("NOAA Weather", "Active" if use_weather else "Off", "Weather alerts, route disruption, emergency demand."),
        ("Apify Trends", "Active" if apify_source_active else "Off", "Search interest and regional demand spikes."),
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
        st.text_input("NOAA weather area", key="weather_area", help="Two-letter US state code, such as TX, NY, CA.")
        st.slider("NOAA alert limit", min_value=1, max_value=25, key="weather_limit")
        st.text_input("Apify geo", key="apify_geo")
        st.selectbox("Apify time range", [APIFY_SAFE_TIME_RANGE], key="apify_time_range")
        st.slider("Apify max keywords", min_value=1, max_value=APIFY_HARD_KEYWORD_LIMIT, key="apify_max_keywords")
        st.caption("Apify live mode is in the sidebar next to Run intelligence.")
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
    news_keywords = parse_lines(st.session_state.get("news_keywords_text", build_default_news_keywords(retailer_label)))
    trends_keywords = parse_lines(st.session_state.get("trends_keywords_text", build_default_trends_keywords(retailer_label)))
    apify_token_value = apify_token.strip()
    apify_run_mode = st.session_state.get("apify_run_mode", "Skip Apify")
    apify_live_confirmed = bool(apify_token_value and apify_run_mode == "Run one live Apify call")
    apify_requested = bool(use_apify or apify_live_confirmed)
    run_config = {
        "retailer": retailer.strip() or "Retailer",
        "region": region,
        "country": country,
        "language": language,
        "enabled_sources": {"gnews": use_gnews, "bls": use_bls, "fda": use_fda, "weather": use_weather, "apify": apify_requested},
        "use_gnews": use_gnews,
        "use_bls": use_bls,
        "use_fda": use_fda,
        "use_weather": use_weather,
        "use_apify": apify_requested,
        "news_keywords": news_keywords,
        "trends_keywords": trends_keywords[:APIFY_HARD_KEYWORD_LIMIT],
        "gnews_period": st.session_state.get("gnews_period", "7d"),
        "max_news": int(st.session_state.get("max_news", 24)),
        "fda_query": st.session_state.get("fda_query", "product_description:(snacks OR candy OR beverages)"),
        "fda_limit": int(st.session_state.get("fda_limit", 8)),
        "weather_area": normalize_weather_area(st.session_state.get("weather_area", "TX")),
        "weather_limit": int(st.session_state.get("weather_limit", 5)),
        "bls_series": BLS_CPI_SERIES,
        "apify_token_present": bool(apify_token_value),
        "apify_run_mode": apify_run_mode,
        "apify_geo": st.session_state.get("apify_geo", "US"),
        "apify_time_range": APIFY_SAFE_TIME_RANGE,
        "apify_max_keywords": int(st.session_state.get("apify_max_keywords", APIFY_HARD_KEYWORD_LIMIT)),
        "apify_live_confirm": apify_live_confirmed,
    }
    results: Dict[str, Dict[str, Any]] = {}
    all_rows: List[Dict[str, Any]] = []
    all_articles: List[Dict[str, Any]] = []

    steps = [
        ("gnews", use_gnews),
        ("bls", use_bls),
        ("fda", use_fda),
        ("weather", use_weather),
        ("apify", apify_requested),
    ]
    active_steps = [step for step in steps if step[1]]
    total = max(1, len(active_steps))
    completed = 0
    collector_states: Dict[str, Dict[str, str]] = {
        "GNEWS": {"status": "queued" if use_gnews else "skipped", "detail": "Retail news collector" if use_gnews else "Disabled"},
        "BLS": {"status": "queued" if use_bls else "skipped", "detail": "CPI collector" if use_bls else "Disabled"},
        "FDA": {"status": "queued" if use_fda else "skipped", "detail": "Recall collector" if use_fda else "Disabled"},
        "WEATHER": {"status": "queued" if use_weather else "skipped", "detail": "NOAA alert collector" if use_weather else "Disabled"},
        "APIFY": {"status": "queued" if apify_requested else "skipped", "detail": f"Trends collector; mode: {apify_run_mode}" if apify_requested else "Disabled"},
    }
    render_run_monitor(run_status_slot, "Starting collectors", collector_states, 2)
    render_sidebar_status(sidebar_status_slot, "Running: starting collectors...", "info")

    if use_gnews:
        collector_states["GNEWS"] = {"status": "running", "detail": "Collecting and deduplicating retail news"}
        render_run_monitor(run_status_slot, "Collecting retail news", collector_states, int((completed / total) * 100))
        render_sidebar_status(sidebar_status_slot, "Running: collecting retail news...", "info")
        results["gnews"] = collect_gnews(
            news_keywords,
            country,
            language,
            st.session_state.get("gnews_period", "7d"),
            int(st.session_state.get("max_news", 24)),
            retailer.strip() or "Retailer",
        )
        all_rows.extend(results["gnews"].get("rows", []))
        all_articles.extend(results["gnews"].get("items", []))
        completed += 1
        gnews_status = results["gnews"].get("status", "failed")
        collector_states["GNEWS"] = {
            "status": "success" if gnews_status == "success" else "failed" if gnews_status == "failed" else "skipped",
            "detail": results["gnews"].get("error") or f"{len(results['gnews'].get('items', []))} article(s), {len(results['gnews'].get('rows', []))} feature row(s)",
        }
        render_run_monitor(run_status_slot, "Retail news complete", collector_states, int((completed / total) * 100))
        render_sidebar_status(sidebar_status_slot, f"GNews {collector_states['GNEWS']['status']}: {collector_states['GNEWS']['detail']}", "warning" if gnews_status != "success" else "info")

    if use_bls:
        collector_states["BLS"] = {"status": "running", "detail": "Collecting headline and category CPI"}
        render_run_monitor(run_status_slot, "Collecting CPI inflation", collector_states, int((completed / total) * 100))
        render_sidebar_status(sidebar_status_slot, "Running: collecting BLS CPI...", "info")
        results["bls"] = collect_bls_cpi(bls_key.strip(), retailer.strip() or "Retailer")
        all_rows.extend(results["bls"].get("rows", []))
        completed += 1
        bls_status = results["bls"].get("status", "failed")
        collector_states["BLS"] = {
            "status": "success" if bls_status == "success" else "failed",
            "detail": results["bls"].get("error") or f"{len(results['bls'].get('rows', []))} CPI feature row(s)",
        }
        render_run_monitor(run_status_slot, "CPI collection complete", collector_states, int((completed / total) * 100))
        render_sidebar_status(sidebar_status_slot, f"BLS {collector_states['BLS']['status']}: {collector_states['BLS']['detail']}", "warning" if bls_status != "success" else "info")

    if use_fda:
        collector_states["FDA"] = {"status": "running", "detail": "Collecting food recall records"}
        render_run_monitor(run_status_slot, "Collecting FDA recalls", collector_states, int((completed / total) * 100))
        render_sidebar_status(sidebar_status_slot, "Running: collecting FDA recalls...", "info")
        results["fda"] = collect_fda_recalls(
            st.session_state.get("fda_query", "product_description:(snacks OR candy OR beverages)"),
            int(st.session_state.get("fda_limit", 8)),
            retailer.strip() or "Retailer",
        )
        all_rows.extend(results["fda"].get("rows", []))
        completed += 1
        fda_status = results["fda"].get("status", "failed")
        collector_states["FDA"] = {
            "status": "success" if fda_status == "success" else "failed",
            "detail": results["fda"].get("error") or f"{len(results['fda'].get('items', []))} recall item(s), {len(results['fda'].get('rows', []))} feature row(s)",
        }
        render_run_monitor(run_status_slot, "FDA recall collection complete", collector_states, int((completed / total) * 100))
        render_sidebar_status(sidebar_status_slot, f"FDA {collector_states['FDA']['status']}: {collector_states['FDA']['detail']}", "warning" if fda_status != "success" else "info")

    if use_weather:
        weather_area = normalize_weather_area(st.session_state.get("weather_area", "TX"))
        collector_states["WEATHER"] = {"status": "running", "detail": f"Collecting active NOAA alerts for {weather_area}"}
        render_run_monitor(run_status_slot, "Collecting weather alerts", collector_states, int((completed / total) * 100))
        render_sidebar_status(sidebar_status_slot, f"Running: collecting NOAA weather alerts for {weather_area}...", "info")
        results["weather"] = collect_weather_alerts(
            weather_area,
            int(st.session_state.get("weather_limit", 5)),
            retailer.strip() or "Retailer",
        )
        all_rows.extend(results["weather"].get("rows", []))
        completed += 1
        weather_status = results["weather"].get("status", "failed")
        collector_states["WEATHER"] = {
            "status": "success" if weather_status == "success" else "failed",
            "detail": results["weather"].get("error") or f"{len(results['weather'].get('items', []))} active alert item(s), {len(results['weather'].get('rows', []))} feature row(s)",
        }
        render_run_monitor(run_status_slot, "Weather alert collection complete", collector_states, int((completed / total) * 100))
        render_sidebar_status(sidebar_status_slot, f"Weather {collector_states['WEATHER']['status']}: {collector_states['WEATHER']['detail']}", "warning" if weather_status != "success" else "info")

    if apify_requested and not apify_token_value:
        results["apify"] = {
            "status": "skipped",
            "source": "Apify Trends",
            "error": "Apify was enabled but no Apify token was provided. No Apify credits were used.",
            "raw": None,
            "rows": [],
            "items": [],
        }
        completed += 1
        collector_states["APIFY"] = {"status": "skipped", "detail": results["apify"]["error"]}
        render_run_monitor(run_status_slot, "Apify skipped: token missing", collector_states, int((completed / total) * 100))
        render_sidebar_status(sidebar_status_slot, "Apify skipped: token missing, no credits used.", "warning")

    elif apify_requested and not apify_live_confirmed:
        results["apify"] = {
            "status": "skipped",
            "source": "Apify Trends",
            "error": f"Apify was enabled but Apify live mode is '{apify_run_mode}'. Select 'Run one live Apify call' in the sidebar to spend one guarded Apify run. No Apify credits were used.",
            "raw": None,
            "rows": [],
            "items": [],
        }
        completed += 1
        collector_states["APIFY"] = {"status": "skipped", "detail": results["apify"]["error"]}
        render_run_monitor(run_status_slot, "Apify skipped without spending credits", collector_states, int((completed / total) * 100))
        render_sidebar_status(sidebar_status_slot, "Apify skipped: no credits used.", "warning")

    elif apify_requested and apify_live_confirmed:
        collector_states["APIFY"] = {"status": "running", "detail": f"One guarded live run: {APIFY_SAFE_TIME_RANGE}, max {APIFY_HARD_KEYWORD_LIMIT} keyword(s)"}
        render_run_monitor(run_status_slot, "Collecting Google Trends via Apify", collector_states, int((completed / total) * 100))
        render_sidebar_status(sidebar_status_slot, f"Running: Apify Trends live call ({APIFY_SAFE_TIME_RANGE}, max {APIFY_HARD_KEYWORD_LIMIT} keywords)...", "warning")
        results["apify"] = collect_apify_trends(
            apify_token_value,
            trends_keywords,
            st.session_state.get("apify_geo", "US"),
            APIFY_SAFE_TIME_RANGE,
            retailer.strip() or "Retailer",
            int(st.session_state.get("apify_max_keywords", APIFY_HARD_KEYWORD_LIMIT)),
        )
        all_rows.extend(results["apify"].get("rows", []))
        completed += 1
        st.session_state["reset_apify_live_confirm"] = True
        apify_status = results["apify"].get("status", "failed")
        collector_states["APIFY"] = {
            "status": "success" if apify_status == "success" else "failed" if apify_status == "failed" else "skipped",
            "detail": results["apify"].get("error") or f"{len(results['apify'].get('items', []))} trends row(s)",
        }
        render_run_monitor(run_status_slot, "Apify collection complete", collector_states, int((completed / total) * 100))
        render_sidebar_status(sidebar_status_slot, f"Apify {collector_states['APIFY']['status']}: {collector_states['APIFY']['detail']}", "warning" if apify_status != "success" else "info")

    render_run_monitor(run_status_slot, "Generating intelligence brief", collector_states, 98)
    render_sidebar_status(sidebar_status_slot, "Running: generating executive brief...", "info")
    feature_df = pd.DataFrame(all_rows)
    if not feature_df.empty:
        feature_df["retailer"] = retailer.strip() or "Retailer"
        if region:
            feature_df["selected_market"] = region
    brief, brief_source, llm_audit = generate_nvidia_brief(nvidia_key.strip(), nvidia_model.strip(), feature_df, all_articles, retailer, region)

    st.session_state["run"] = {
        "timestamp": utc_now(),
        "run_config": run_config,
        "results": results,
        "feature_df": feature_df,
        "articles": all_articles,
        "brief": brief,
        "brief_source": brief_source,
        "llm_audit": llm_audit,
    }
    render_sidebar_status(sidebar_status_slot, "Run complete. Opening Results...", "success")
    st.session_state["force_results_view"] = True
    st.session_state["last_collector_states"] = collector_states
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
            st.markdown('<div class="small-header">Score Explainability</div>', unsafe_allow_html=True)
            render_score_explainability(feature_df)

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
        render_executive_brief(run, feature_df)


if view == "Evidence Audit":
    run = st.session_state.get("run")
    if not run:
        st.markdown(
            "<div class='empty-console'>"
            "<div>"
            "<div class='hero-kicker'>Evidence Audit</div>"
            "<div class='empty-title'>No run evidence yet.</div>"
            "<div class='empty-body'>Run the workbench once to see raw collector payloads, normalized feature rows, score reasoning, and the exact LLM or fallback analysis input.</div>"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        feature_df = run.get("feature_df", pd.DataFrame())
        run_config = run.get(
            "run_config",
            {
                "retailer": retailer_label,
                "region": region,
                "country": country,
                "language": language,
                "enabled_sources": {key: key in run.get("results", {}) for key in SOURCE_ORDER},
            },
        )
        llm_audit = run.get("llm_audit") or build_base_llm_audit(
            feature_df,
            run.get("articles", []),
            run_config.get("retailer", retailer_label),
            run_config.get("region", region),
            nvidia_model.strip() or DEFAULT_NVIDIA_MODEL,
        )
        llm_audit["brief_source"] = run.get("brief_source", llm_audit.get("brief_source", "unknown"))
        evidence_records = build_collector_evidence(run.get("results", {}), run_config)
        evidence_df = pd.DataFrame(evidence_records)
        mock_used = any_mock_used(run.get("results", {}), llm_audit)
        pulled_sources = int((evidence_df["raw_records_pulled"] > 0).sum()) if not evidence_df.empty else 0
        raw_records = int(evidence_df["raw_records_pulled"].sum()) if not evidence_df.empty else 0

        render_audit_banner(mock_used, str(run.get("brief_source", "unknown")))

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_audit_card("Mock Data Used", "Yes" if mock_used else "No", "Collector rows are marked per run result.")
        with c2:
            render_audit_card("Raw Records", str(raw_records), f"{pulled_sources} source(s) returned inspectable data.")
        with c3:
            render_audit_card(
                "Rows Sent To Brief",
                f"{llm_audit.get('feature_rows_sent', 0)} / {llm_audit.get('feature_rows_available', 0)}",
                "Feature payload is capped for concise LLM context.",
            )
        with c4:
            render_audit_card(
                "LLM Call",
                "NVIDIA" if llm_audit.get("sent_to_llm") else "No external LLM",
                str(llm_audit.get("fallback_reason") or "NVIDIA analyzed the shown payload."),
            )

        st.markdown('<div class="small-header">Collector Evidence Table</div>', unsafe_allow_html=True)
        st.dataframe(evidence_df, width="stretch", hide_index=True)

        st.markdown('<div class="small-header">Normalized Feature Rows And Score Reasoning</div>', unsafe_allow_html=True)
        if feature_df.empty:
            st.info("No normalized feature rows were generated. Check collector statuses and errors above.")
        else:
            preferred_cols = [
                "source",
                "signal_area",
                "signal_name",
                "region",
                "region_scope",
                "signal_value",
                "risk_score",
                "confidence",
                "score_reason",
                "business_impact",
                "recommended_action",
                "raw_reference",
            ]
            visible_cols = [col for col in preferred_cols if col in feature_df.columns]
            st.dataframe(feature_df[visible_cols], width="stretch", hide_index=True)

        st.markdown('<div class="small-header">LLM Analysis Trace</div>', unsafe_allow_html=True)
        trace_cols = st.columns(4)
        with trace_cols[0]:
            render_metric_card("Brief Source", str(run.get("brief_source", "unknown")), "NVIDIA if sent; fallback if local rules were used.")
        with trace_cols[1]:
            render_metric_card("Payload Hash", str(llm_audit.get("payload_hash_sha256", ""))[:12], "SHA-256 fingerprint of features/articles payload.")
        with trace_cols[2]:
            render_metric_card("Articles Sent", str(llm_audit.get("articles_sent", 0)), f"{llm_audit.get('articles_available', 0)} available.")
        with trace_cols[3]:
            render_metric_card("Fallback Used", "Yes" if llm_audit.get("fallback_used") else "No", str(llm_audit.get("fallback_reason") or "External LLM response used."))

        with st.expander("Prompt and payload used for the brief", expanded=True):
            if llm_audit.get("sent_to_llm"):
                st.success("NVIDIA was called with only the feature rows and article records shown below.")
            else:
                st.warning("No external LLM call was made for this run. The brief was generated by deterministic local rules using the same feature rows.")
            st.markdown("System prompt")
            st.code(str(llm_audit.get("system_prompt", "")), language="text")
            st.markdown("User prompt")
            st.code(str(llm_audit.get("user_prompt", "")), language="text")
            st.markdown("Payload")
            st.code(json.dumps(llm_audit.get("payload", {}), indent=2, default=str)[:16000], language="json")

        with st.expander("Brief output", expanded=False):
            st.markdown(f"<div class='brief-box'>{brief_to_html(str(run.get('brief', '')))}</div>", unsafe_allow_html=True)

        st.markdown('<div class="small-header">Raw Pulled Data By Source</div>', unsafe_allow_html=True)
        for name, result in run.get("results", {}).items():
            source_name = SOURCE_LABELS.get(name, name.upper())
            raw_count = count_raw_records(result.get("raw"), result.get("items"))
            with st.expander(f"{source_name} - {result.get('status', 'unknown')} - {raw_count} raw record(s)", expanded=False):
                if result.get("error"):
                    st.warning(result["error"])
                if result.get("items"):
                    st.markdown("Cleaned items")
                    st.dataframe(pd.DataFrame(result["items"]), width="stretch", hide_index=True)
                raw_preview = result.get("raw")
                if raw_preview is not None:
                    st.markdown("Raw payload")
                    st.code(json.dumps(raw_preview, indent=2, default=str)[:16000], language="json")
                if result.get("rows"):
                    st.markdown("Normalized feature rows from this source")
                    st.dataframe(pd.DataFrame(result["rows"]), width="stretch", hide_index=True)

        evidence_bundle = {
            "timestamp": run.get("timestamp"),
            "mock_data_used": mock_used,
            "run_config": run_config,
            "collector_evidence": evidence_records,
            "normalized_features": feature_df.to_dict(orient="records") if not feature_df.empty else [],
            "llm_audit": llm_audit,
            "results": {
                name: {
                    "status": result.get("status"),
                    "source": result.get("source"),
                    "error": result.get("error"),
                    "rows": result.get("rows", []),
                    "items": result.get("items", []),
                    "raw": result.get("raw"),
                }
                for name, result in run.get("results", {}).items()
            },
        }
        st.download_button(
            "Download evidence_audit.json",
            json.dumps(evidence_bundle, indent=2, default=str).encode("utf-8"),
            "evidence_audit.json",
            "application/json",
            width="stretch",
        )


if view == "Raw Data":
    run = st.session_state.get("run")
    if not run:
        st.markdown(
            "<div class='empty-console'>"
            "<div>"
            "<div class='hero-kicker'>Raw Evidence</div>"
            "<div class='empty-title'>Collector payloads will appear after a run.</div>"
            "<div class='empty-body'>This view is intentionally evidence-first: GNews articles, CPI JSON, FDA recall records, NOAA weather alerts, Apify errors, and cleaned item tables stay inspectable for validation.</div>"
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


# import json
# import os
# import re
# import xml.etree.ElementTree as ET
# from datetime import datetime, timezone
# from email.utils import parsedate_to_datetime
# from html import escape, unescape
# from typing import Any, Dict, List, Optional, Tuple
# from urllib.parse import quote_plus

# import pandas as pd
# import plotly.graph_objects as go
# import requests
# import streamlit as st

# try:
#     from dotenv import load_dotenv
# except ImportError:  # pragma: no cover - optional local convenience
#     load_dotenv = None


# if load_dotenv:
#     load_dotenv()


# NVIDIA_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
# DEFAULT_NVIDIA_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1.5"
# BLS_CPI_SERIES = {
#     "Headline CPI": "CUUR0000SA0",
#     "Food at home": "CUUR0000SAF11",
#     "Household furnishings": "CUUR0000SAH3",
#     "Gasoline": "CUUR0000SETB",
# }


# st.set_page_config(
#     page_title="Dollar Tree Market Intelligence Workbench",
#     page_icon="DT",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )


# st.markdown(
#     """
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
#     :root {
#         --bg: #F5F7FB;
#         --sf: #FFFFFF;
#         --s2: #F8FAFE;
#         --s3: #F0F4F9;
#         --s4: #E9EFF5;
#         --or: #F47B25;
#         --olt: #FF9F50;
#         --odk: #C45D0A;
#         --og: rgba(244,123,37,0.12);
#         --ob: rgba(244,123,37,0.07);
#         --obr: rgba(244,123,37,0.25);
#         --bl: #E2E8F0;
#         --t: #1E293B;
#         --t2: #475569;
#         --t3: #94A3B8;
#         --gr: #22C55E;
#         --gbg: rgba(34,197,94,0.10);
#         --am: #F59E0B;
#         --abg: rgba(245,158,11,0.10);
#         --rd: #EF4444;
#         --rbg: rgba(239,68,68,0.08);
#         --r: 12px;
#         --rl: 16px;
#         --sh: 0 1px 3px rgba(0,0,0,0.04);
#         --shm: 0 6px 14px -4px rgba(0,0,0,0.10);
#         --tr: 0.2s cubic-bezier(0.4,0,0.2,1);
#     }
#     * { box-sizing: border-box; }
#     html, body, [class*="css"] {
#         font-family: 'Inter', ui-sans-serif, system-ui, sans-serif;
#         color: var(--t);
#     }
#     .stApp { background: var(--bg); }
#     #MainMenu, footer { visibility: hidden; }
#     header { visibility: hidden; }
#     .accent-bar {
#         height: 3px;
#         background: linear-gradient(90deg, var(--odk), var(--or), var(--olt), var(--or));
#         background-size: 200%;
#         animation: shimmer 3s linear infinite;
#         width: 100%;
#         margin: -1.25rem 0 0.9rem 0;
#     }
#     @keyframes shimmer { 0% { background-position: 200%; } 100% { background-position: -200%; } }
#     @keyframes pdot { 0% { box-shadow: 0 0 0 0 rgba(34,197,94,0.5); } 50% { box-shadow: 0 0 0 5px rgba(34,197,94,0); } }
#     .ldot {
#         width: 7px;
#         height: 7px;
#         border-radius: 50%;
#         background: var(--gr);
#         animation: pdot 2s infinite;
#         display: inline-block;
#     }
#     .main .block-container {
#         padding-top: 1.25rem;
#         max-width: 1400px;
#     }
#     [data-testid="stSidebar"] {
#         background: var(--sf) !important;
#         border-right: 1px solid var(--bl) !important;
#     }
#     [data-testid="stSidebar"] * {
#         color: var(--t2);
#     }
#     h1, h2, h3 {
#         color: var(--t);
#         letter-spacing: 0;
#     }
#     h1 { font-size: 2.25rem; font-weight: 900; letter-spacing: -0.02em; }
#     .subtle {
#         color: var(--t2);
#         font-size: 0.94rem;
#         line-height: 1.55;
#     }
#     .topbar {
#         height: 54px;
#         background: var(--sf);
#         border: 1px solid var(--bl);
#         border-radius: var(--rl);
#         display: flex;
#         align-items: center;
#         padding: 0 18px;
#         gap: 12px;
#         box-shadow: var(--sh);
#         margin-bottom: 18px;
#     }
#     .tt { font-size: 14px; font-weight: 800; color: var(--t); }
#     .tt span { color: var(--t3); font-weight: 500; }
#     .tbadge {
#         background: var(--ob);
#         border: 1px solid var(--obr);
#         color: var(--or);
#         font-size: 10px;
#         font-weight: 800;
#         padding: 2px 8px;
#         border-radius: 20px;
#     }
#     .hero-shell {
#         background:
#             radial-gradient(circle at 88% 18%, rgba(244,123,37,0.18), transparent 28%),
#             linear-gradient(135deg, #FFFFFF 0%, #F8FAFE 54%, #FFF7ED 100%);
#         border: 1px solid var(--bl);
#         border-radius: 22px;
#         padding: 24px 26px;
#         box-shadow: 0 12px 30px -22px rgba(15,23,42,0.35);
#         margin-bottom: 16px;
#         position: relative;
#         overflow: hidden;
#     }
#     .hero-shell:before {
#         content: "";
#         position: absolute;
#         left: 0;
#         right: 0;
#         top: 0;
#         height: 4px;
#         background: linear-gradient(90deg, var(--odk), var(--or), var(--olt));
#     }
#     .hero-kicker {
#         display: inline-flex;
#         align-items: center;
#         gap: 7px;
#         background: var(--ob);
#         border: 1px solid var(--obr);
#         color: var(--or);
#         border-radius: 999px;
#         padding: 4px 10px;
#         font-size: 10px;
#         font-weight: 900;
#         letter-spacing: 0.8px;
#         text-transform: uppercase;
#         margin-bottom: 12px;
#     }
#     .hero-title {
#         font-size: 42px;
#         line-height: 1.02;
#         letter-spacing: -0.04em;
#         font-weight: 950;
#         color: var(--t);
#         max-width: 820px;
#         margin: 0;
#     }
#     .hero-copy {
#         color: var(--t2);
#         font-size: 14px;
#         line-height: 1.7;
#         max-width: 820px;
#         margin: 14px 0 0 0;
#     }
#     .hero-side {
#         background: rgba(255,255,255,0.74);
#         border: 1px solid rgba(226,232,240,0.9);
#         border-radius: var(--rl);
#         padding: 14px;
#         box-shadow: var(--sh);
#     }
#     .hero-side-label {
#         color: var(--t3);
#         font-size: 9px;
#         font-weight: 900;
#         letter-spacing: 1.2px;
#         text-transform: uppercase;
#         margin-bottom: 8px;
#     }
#     .hero-side-row {
#         display: flex;
#         justify-content: space-between;
#         gap: 12px;
#         border-top: 1px solid var(--bl);
#         padding-top: 8px;
#         margin-top: 8px;
#         font-size: 12px;
#         color: var(--t2);
#     }
#     .hero-side-row strong { color: var(--t); }
#     .source-tile {
#         background: var(--sf);
#         border: 1px solid var(--bl);
#         border-radius: 14px;
#         padding: 13px 14px;
#         box-shadow: var(--sh);
#         min-height: 86px;
#         transition: all var(--tr);
#     }
#     .source-tile:hover { border-color: var(--obr); box-shadow: var(--shm); transform: translateY(-1px); }
#     .source-name {
#         font-size: 12px;
#         font-weight: 850;
#         color: var(--t);
#         margin-bottom: 5px;
#     }
#     .source-meta {
#         color: var(--t2);
#         font-size: 11px;
#         line-height: 1.45;
#     }
#     .console-panel {
#         background: var(--sf);
#         border: 1px solid var(--bl);
#         border-radius: 18px;
#         padding: 18px;
#         box-shadow: var(--sh);
#         margin-top: 12px;
#     }
#     .empty-console {
#         background:
#             linear-gradient(135deg, rgba(244,123,37,0.08), rgba(255,255,255,0.85)),
#             var(--sf);
#         border: 1px dashed var(--obr);
#         border-radius: 18px;
#         padding: 28px;
#         min-height: 210px;
#         display: flex;
#         align-items: center;
#         justify-content: space-between;
#         gap: 20px;
#     }
#     .empty-title {
#         color: var(--t);
#         font-size: 22px;
#         line-height: 1.15;
#         font-weight: 900;
#         letter-spacing: -0.025em;
#         margin-bottom: 8px;
#     }
#     .empty-body {
#         color: var(--t2);
#         font-size: 13px;
#         line-height: 1.65;
#         max-width: 620px;
#     }
#     .workflow {
#         display: grid;
#         grid-template-columns: repeat(5, minmax(0, 1fr));
#         gap: 8px;
#         margin-top: 12px;
#     }
#     .workflow-step {
#         background: var(--s2);
#         border: 1px solid var(--bl);
#         border-radius: 12px;
#         padding: 10px;
#         font-size: 11px;
#         color: var(--t2);
#         font-weight: 700;
#     }
#     .workflow-step span {
#         display: block;
#         color: var(--or);
#         font-size: 9px;
#         letter-spacing: 1px;
#         text-transform: uppercase;
#         font-weight: 900;
#         margin-bottom: 3px;
#     }
#     .metric-card {
#         background: var(--sf);
#         border: 1px solid var(--bl);
#         border-radius: var(--rl);
#         padding: 16px 18px;
#         min-height: 132px;
#         box-shadow: var(--sh);
#         transition: all var(--tr);
#     }
#     .metric-card:hover {
#         transform: translateY(-1px);
#         box-shadow: var(--shm);
#         border-color: var(--obr);
#     }
#     .metric-label {
#         color: var(--t3);
#         font-size: 0.68rem;
#         text-transform: uppercase;
#         letter-spacing: 0.11em;
#         margin-bottom: 8px;
#         font-weight: 800;
#     }
#     .metric-value {
#         color: var(--t);
#         font-size: 2.05rem;
#         font-weight: 900;
#         line-height: 1;
#         letter-spacing: -0.04em;
#     }
#     .metric-note {
#         color: var(--t2);
#         font-size: 0.82rem;
#         margin-top: 10px;
#         line-height: 1.4;
#     }
#     .pill {
#         display: inline-block;
#         border-radius: 999px;
#         padding: 3px 10px;
#         font-size: 0.72rem;
#         font-weight: 800;
#         border: 1px solid var(--bl);
#         color: var(--t2);
#         background: var(--s2);
#         margin-top: 8px;
#     }
#     .pill-high { color: var(--gr); background: var(--gbg); border-color: rgba(34,197,94,0.2); }
#     .pill-medium { color: var(--am); background: var(--abg); border-color: rgba(245,158,11,0.2); }
#     .pill-low { color: var(--rd); background: var(--rbg); border-color: rgba(239,68,68,0.2); }
#     .brief-box {
#         background: var(--sf);
#         border: 1px solid var(--obr);
#         border-left: 4px solid var(--or);
#         border-radius: var(--rl);
#         padding: 20px 22px;
#         color: var(--t);
#         line-height: 1.65;
#         box-shadow: 0 0 0 3px var(--og);
#     }
#     .small-header {
#         color: var(--t3);
#         font-size: 0.68rem;
#         text-transform: uppercase;
#         letter-spacing: 0.12em;
#         font-weight: 800;
#         margin: 8px 0 12px 0;
#         padding-bottom: 6px;
#         border-bottom: 1px solid var(--bl);
#     }
#     .action-card {
#         background: var(--sf);
#         border: 1px solid var(--bl);
#         border-radius: var(--r);
#         padding: 12px 14px;
#         box-shadow: var(--sh);
#         min-height: 112px;
#     }
#     .action-label {
#         font-size: 9px;
#         font-weight: 900;
#         letter-spacing: 1px;
#         text-transform: uppercase;
#         color: var(--or);
#         margin-bottom: 7px;
#     }
#     .action-title {
#         font-size: 13px;
#         font-weight: 800;
#         color: var(--t);
#         margin-bottom: 5px;
#     }
#     .action-body {
#         font-size: 12px;
#         color: var(--t2);
#         line-height: 1.45;
#     }
#     .note-box {
#         background: rgba(244,123,37,0.04);
#         border-left: 3px solid var(--or);
#         border-radius: 0 8px 8px 0;
#         padding: 9px 12px;
#         font-size: 12px;
#         color: var(--t2);
#         margin: 8px 0;
#     }
#     div[role="radiogroup"] {
#         background: var(--sf);
#         border: 1px solid var(--bl);
#         border-radius: 14px;
#         padding: 5px;
#         display: inline-flex;
#         gap: 4px;
#         box-shadow: var(--sh);
#         margin: 4px 0 12px 0;
#     }
#     div[role="radiogroup"] label {
#         border-radius: 10px !important;
#         padding: 6px 13px !important;
#         min-height: 34px !important;
#         transition: all var(--tr);
#     }
#     div[role="radiogroup"] label:has(input:checked) {
#         background: var(--ob) !important;
#         border: 1px solid var(--obr) !important;
#         color: var(--or) !important;
#         font-weight: 850 !important;
#     }
#     div[role="radiogroup"] label span {
#         font-size: 12px !important;
#         font-weight: 750 !important;
#     }
#     div[data-testid="stButton"] button {
#         background: var(--or);
#         color: #fff;
#         border: none;
#         border-radius: var(--r);
#         font-weight: 800;
#         box-shadow: 0 2px 8px rgba(244,123,37,0.2);
#         transition: all var(--tr);
#     }
#     div[data-testid="stButton"] button:hover {
#         background: var(--odk);
#         color: #fff;
#         border: none;
#         transform: translateY(-1px);
#     }
#     .stTabs [data-baseweb="tab-list"] {
#         gap: 4px;
#         border-bottom: 1px solid var(--bl);
#     }
#     .stTabs [data-baseweb="tab"] {
#         border-radius: 9px 9px 0 0;
#         color: var(--t2);
#         font-weight: 700;
#     }
#     .stTabs [aria-selected="true"] {
#         background: var(--ob);
#         color: var(--or) !important;
#         border: 1px solid var(--obr);
#         border-bottom-color: transparent;
#     }
#     .stSelectbox>div>div, .stTextInput>div>div, .stTextArea>div>div {
#         background: var(--s2) !important;
#         border: 1px solid var(--bl) !important;
#         border-radius: var(--r) !important;
#         font-size: 13px !important;
#         color: var(--t) !important;
#     }
#     [data-testid="stExpander"] {
#         background: var(--sf);
#         border: 1px solid var(--bl) !important;
#         border-radius: var(--rl) !important;
#         box-shadow: var(--sh);
#     }
#     ::-webkit-scrollbar { width: 4px; height: 4px; }
#     ::-webkit-scrollbar-track { background: transparent; }
#     ::-webkit-scrollbar-thumb { background: var(--bl); border-radius: 2px; }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )


# def utc_now() -> str:
#     return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# def safe_request(
#     url: str,
#     *,
#     method: str = "GET",
#     headers: Optional[Dict[str, str]] = None,
#     params: Optional[Dict[str, Any]] = None,
#     json_body: Optional[Dict[str, Any]] = None,
#     timeout: int = 30,
# ) -> Tuple[bool, Any, str]:
#     try:
#         if method.upper() == "POST":
#             response = requests.post(url, headers=headers, params=params, json=json_body, timeout=timeout)
#         else:
#             response = requests.get(url, headers=headers, params=params, timeout=timeout)
#         response.raise_for_status()
#         try:
#             return True, response.json(), "success"
#         except ValueError:
#             return True, response.text, "success"
#     except requests.RequestException as exc:
#         return False, None, str(exc)


# def confidence_class(confidence: str) -> str:
#     lookup = {"High": "pill-high", "Medium": "pill-medium", "Low": "pill-low"}
#     return lookup.get(confidence, "")


# def risk_band(score: float) -> str:
#     if score >= 8:
#         return "High"
#     if score >= 5:
#         return "Medium"
#     return "Low"


# def source_confidence(publisher: str) -> str:
#     high = ["reuters", "associated press", "ap news", "bloomberg", "sec", "dollar tree", "pr newswire"]
#     medium = ["cnbc", "yahoo", "nasdaq", "marketwatch", "forbes", "investing.com", "retail dive"]
#     p = (publisher or "").lower()
#     if any(name in p for name in high):
#         return "High"
#     if any(name in p for name in medium):
#         return "Medium"
#     return "Medium" if publisher else "Low"


# def validate_nvidia(api_key: str, model: str) -> Tuple[bool, str]:
#     if not api_key:
#         return False, "NVIDIA key not provided. Brief generation will use a local fallback."
#     prompt = "Return exactly: connected"
#     ok, data, msg = safe_request(
#         NVIDIA_CHAT_URL,
#         method="POST",
#         headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
#         json_body={
#             "model": model,
#             "messages": [{"role": "user", "content": prompt}],
#             "temperature": 0,
#             "max_tokens": 8,
#         },
#         timeout=30,
#     )
#     if not ok:
#         return False, msg
#     content = data.get("choices", [{}])[0].get("message", {}).get("content")
#     content_text = str(content or "").strip()
#     return True, f"Connected. Model responded: {content_text or 'ok'}"


# def validate_apify(token: str) -> Tuple[bool, str]:
#     if not token:
#         return False, "Apify token not provided. Apify collectors will be skipped."
#     try:
#         from apify_client import ApifyClient
#     except ImportError:
#         return False, "apify-client is not installed. Install requirements before running Apify collectors."
#     try:
#         client = ApifyClient(token)
#         user = client.user().get()
#         username = user.get("username") or user.get("email") or "Apify user"
#         return True, f"Connected as {username}."
#     except Exception as exc:  # pragma: no cover - depends on live Apify service
#         return False, str(exc)


# def build_cpi_signal(data: Dict[str, Any], label: str, series_id: str) -> Tuple[Optional[Dict[str, Any]], pd.DataFrame]:
#     series = data.get("Results", {}).get("series", [])
#     rows = []
#     for item in (series[0].get("data", []) if series else [])[:24]:
#         period = item.get("period", "")
#         if period == "M13":
#             continue
#         try:
#             cpi_value = float(item["value"])
#         except (TypeError, ValueError, KeyError):
#             continue
#         rows.append(
#             {
#                 "category": label,
#                 "series_id": series_id,
#                 "year": int(item["year"]),
#                 "period": period,
#                 "month": item.get("periodName", ""),
#                 "cpi_value": cpi_value,
#             }
#         )
#     df = pd.DataFrame(rows)
#     if df.empty:
#         return None, df
#     df = df.sort_values(["year", "period"]).reset_index(drop=True)
#     df["cpi_mom_change_pct"] = df["cpi_value"].pct_change() * 100
#     df["cpi_yoy_change_pct"] = df["cpi_value"].pct_change(12) * 100
#     latest = df.iloc[-1].to_dict()
#     mom = latest.get("cpi_mom_change_pct")
#     yoy = latest.get("cpi_yoy_change_pct")
#     score = 4.0
#     if pd.notna(mom):
#         score = min(10.0, max(1.0, 4.0 + float(mom) * 5.0))
#     if pd.notna(yoy) and yoy > 4:
#         score = min(10.0, score + 1.0)
#     signal_name = "inflation_pressure_score" if label == "Headline CPI" else f"{label.lower().replace(' ', '_')}_cpi_pressure_score"
#     signal = {
#         "date": f"{int(latest['year'])}-{str(latest['period']).replace('M', '').zfill(2)}",
#         "retailer": "Dollar Tree",
#         "region": "US",
#         "source": "BLS CPI",
#         "signal_area": "Inflation" if label == "Headline CPI" else "Category CPI",
#         "signal_name": signal_name,
#         "signal_value": round(float(score), 2),
#         "risk_score": round(float(score), 2),
#         "confidence": "High",
#         "business_impact": f"{label} inflation can affect price sensitivity, category demand, and basket mix.",
#         "recommended_action": "Use category CPI as an external regressor and validate against internal category sales.",
#         "raw_reference": f"{label}: CPI {latest['cpi_value']}",
#     }
#     return signal, df


# def collect_bls_cpi(bls_key: str = "") -> Dict[str, Any]:
#     payload: Dict[str, Any] = {"seriesid": list(BLS_CPI_SERIES.values())}
#     if bls_key:
#         payload["registrationkey"] = bls_key
#     signals = []
#     tables = []
#     raw = {}
#     errors = []
#     ok, data, msg = safe_request(
#         "https://api.bls.gov/publicAPI/v2/timeseries/data/",
#         method="POST",
#         headers={"Content-Type": "application/json"},
#         json_body=payload,
#         timeout=30,
#     )
#     if not ok:
#         return {"status": "failed", "source": "BLS CPI", "error": msg, "raw": None, "rows": []}
#     if data.get("status") != "REQUEST_SUCCEEDED":
#         return {
#             "status": "failed",
#             "source": "BLS CPI",
#             "error": "; ".join(data.get("message", [])) or "BLS request was not processed.",
#             "raw": data,
#             "rows": [],
#         }
#     series_by_id = {
#         series.get("seriesID"): series
#         for series in data.get("Results", {}).get("series", [])
#     }
#     for label, series_id in BLS_CPI_SERIES.items():
#         series_payload = {"Results": {"series": [series_by_id.get(series_id, {})]}}
#         raw[label] = series_payload
#         signal, df = build_cpi_signal(series_payload, label, series_id)
#         if signal:
#             signals.append(signal)
#         if not df.empty:
#             tables.append(df)
#     if not signals:
#         return {"status": "failed", "source": "BLS CPI", "error": "; ".join(errors) or "No CPI rows returned.", "raw": raw, "rows": []}
#     table = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
#     return {"status": "success", "source": "BLS CPI", "error": "; ".join(errors), "raw": raw, "rows": signals, "table": table}


# def classify_recall(reason: str) -> Tuple[str, float]:
#     text = (reason or "").lower()
#     if any(word in text for word in ["lead", "heavy metal", "salmonella", "listeria", "e. coli", "contamination"]):
#         return "high_safety_risk", 8.0
#     if any(word in text for word in ["undeclared", "allergen", "milk", "peanut", "soy", "tree nut"]):
#         return "allergen_risk", 6.5
#     if any(word in text for word in ["mislabel", "label"]):
#         return "labeling_risk", 4.5
#     return "general_recall_risk", 5.0


# def extract_upcs(text: str) -> List[str]:
#     candidates = re.findall(r"(?:UPC(?:\s*Code)?[:\s]*)?(\d(?:[\s-]?\d){7,13})", text or "", flags=re.IGNORECASE)
#     cleaned = []
#     for candidate in candidates:
#         digits = re.sub(r"\D", "", candidate)
#         if 8 <= len(digits) <= 14 and digits not in cleaned:
#             cleaned.append(digits)
#     return cleaned


# def adjust_recall_score(base_score: float, classification: str, status: str) -> float:
#     score = base_score
#     class_text = (classification or "").lower()
#     status_text = (status or "").lower()
#     if "class i" in class_text:
#         score += 1.5
#     elif "class ii" in class_text:
#         score += 0.8
#     if "ongoing" in status_text:
#         score += 1.0
#     elif "terminated" in status_text:
#         score -= 1.0
#     return round(min(10.0, max(1.0, score)), 2)


# def collect_fda_recalls(query: str, limit: int) -> Dict[str, Any]:
#     params = {"search": query, "limit": limit}
#     ok, data, msg = safe_request("https://api.fda.gov/food/enforcement.json", params=params, timeout=30)
#     if not ok:
#         return {"status": "failed", "source": "openFDA", "error": msg, "raw": None, "rows": [], "items": []}
#     results = data.get("results", [])
#     rows = []
#     items = []
#     for item in results:
#         risk_type, base_score = classify_recall(item.get("reason_for_recall", ""))
#         product = item.get("product_description", "Unknown product")
#         state = item.get("state", "US")
#         classification = item.get("classification", "")
#         status = item.get("status", "")
#         score = adjust_recall_score(base_score, classification, status)
#         upcs = extract_upcs(f"{product} {item.get('code_info', '')}")
#         items.append(
#             {
#                 "product": product,
#                 "reason": item.get("reason_for_recall", ""),
#                 "state": state,
#                 "classification": classification,
#                 "status": status,
#                 "recall_date": item.get("recall_initiation_date", ""),
#                 "distribution_pattern": item.get("distribution_pattern", ""),
#                 "recalling_firm": item.get("recalling_firm", ""),
#                 "upcs": ", ".join(upcs) if upcs else "",
#                 "sku_match_status": "unknown",
#                 "risk_type": risk_type,
#                 "risk_score": score,
#             }
#         )
#     aggregate_score = max([x["risk_score"] for x in items], default=1.0)
#     ongoing_count = sum(1 for x in items if str(x.get("status", "")).lower() == "ongoing")
#     class_i_count = sum(1 for x in items if "class i" in str(x.get("classification", "")).lower())
#     signal = {
#         "date": utc_now()[:10],
#         "retailer": "Dollar Tree",
#         "region": "US",
#         "source": "openFDA",
#         "signal_area": "Product Recalls",
#         "signal_name": "recall_risk_score",
#         "signal_value": len(items),
#         "risk_score": round(aggregate_score, 2),
#         "confidence": "High",
#         "business_impact": "Food and beverage recalls can trigger inventory withdrawal, substitution demand, and safety review.",
#         "recommended_action": "Prioritize ongoing and Class I recalls, then match UPCs against Dollar Tree inventory before store-level action.",
#         "raw_reference": f"{len(items)} recall records; {ongoing_count} ongoing; {class_i_count} Class I",
#     }
#     rows.append(signal)
#     return {"status": "success", "source": "openFDA", "error": "", "raw": data, "rows": rows, "items": items}


# def gnews_package_collect(keyword: str, country: str, language: str, period: str, max_results: int) -> Optional[List[Dict[str, Any]]]:
#     try:
#         from gnews import GNews
#     except ImportError:
#         return None
#     google_news = GNews(language=language, country=country, period=period, max_results=max_results)
#     return google_news.get_news(keyword)


# def google_news_rss_collect(keyword: str, country: str, language: str, period: str, max_results: int) -> List[Dict[str, Any]]:
#     # Google News RSS supports a "when:" query operator. GNews package is preferred when installed.
#     query = quote_plus(f"{keyword} when:{period}")
#     country_code = country.upper()
#     lang_code = language.lower()
#     url = f"https://news.google.com/rss/search?q={query}&hl={lang_code}-{country_code}&gl={country_code}&ceid={country_code}:{lang_code}"
#     response = requests.get(url, timeout=30)
#     response.raise_for_status()
#     root = ET.fromstring(response.content)
#     articles = []
#     for item in root.findall(".//item")[:max_results]:
#         source_node = item.find("source")
#         articles.append(
#             {
#                 "title": item.findtext("title", default=""),
#                 "description": item.findtext("description", default=""),
#                 "published date": item.findtext("pubDate", default=""),
#                 "url": item.findtext("link", default=""),
#                 "publisher": source_node.text if source_node is not None else "",
#             }
#         )
#     return articles


# def clean_news_description(description: str) -> str:
#     text = re.sub(r"<[^>]+>", " ", description or "")
#     text = unescape(text)
#     text = re.sub(r"\s+", " ", text).strip()
#     return text


# def article_days_old(published_date: str) -> Optional[int]:
#     if not published_date:
#         return None
#     try:
#         published = parsedate_to_datetime(published_date)
#         if published.tzinfo is None:
#             published = published.replace(tzinfo=timezone.utc)
#         return max(0, (datetime.now(timezone.utc) - published.astimezone(timezone.utc)).days)
#     except (TypeError, ValueError):
#         return None


# def classify_news_title(title: str, description: str) -> Tuple[str, float, str]:
#     text = f"{title} {description}".lower()
#     if any(word in text for word in ["recall", "contamination", "lawsuit", "closure", "closing", "tariff", "warning"]):
#         return "risk_event", 7.0, "negative"
#     if any(word in text for word in ["inflation", "prices", "freight", "cost", "margin"]):
#         return "price_pressure", 6.0, "negative"
#     if any(word in text for word in ["deal", "sale", "promotion", "coupon", "holiday", "seasonal"]):
#         return "demand_opportunity", 6.5, "positive"
#     if any(word in text for word in ["earnings", "forecast", "guidance", "outlook"]):
#         return "financial_update", 5.5, "neutral"
#     return "general_market_news", 3.5, "neutral"


# def collect_gnews(keywords: List[str], country: str, language: str, period: str, max_results: int) -> Dict[str, Any]:
#     all_articles = []
#     errors = []
#     per_keyword_limit = max(1, int(max_results / max(1, len(keywords))))
#     for keyword in keywords:
#         try:
#             articles = gnews_package_collect(keyword, country, language, period, per_keyword_limit)
#             if articles is None:
#                 articles = google_news_rss_collect(keyword, country, language, period, per_keyword_limit)
#             for article in articles or []:
#                 description = clean_news_description(article.get("description", ""))
#                 event_type, score, sentiment = classify_news_title(article.get("title", ""), description)
#                 publisher = article.get("publisher", "")
#                 if isinstance(publisher, dict):
#                     publisher = publisher.get("title") or publisher.get("href") or ""
#                 published_date = article.get("published date") or article.get("published_date", "")
#                 days_old = article_days_old(published_date)
#                 if days_old is not None and days_old > 30:
#                     score = max(1.0, score - 1.0)
#                 all_articles.append(
#                     {
#                         "keyword": keyword,
#                         "title": article.get("title", ""),
#                         "description": description,
#                         "published_date": published_date,
#                         "days_old": days_old,
#                         "publisher": publisher,
#                         "url": article.get("url", ""),
#                         "source_tier": source_confidence(str(publisher)),
#                         "event_type": event_type,
#                         "sentiment": sentiment,
#                         "risk_score": score,
#                         "confidence": source_confidence(str(publisher)),
#                     }
#                 )
#         except Exception as exc:
#             errors.append(f"{keyword}: {exc}")

#     deduped = []
#     seen = set()
#     for article in all_articles:
#         key = article["url"] or article["title"]
#         if key and key not in seen:
#             seen.add(key)
#             deduped.append(article)

#     score = round(float(pd.Series([a["risk_score"] for a in deduped]).mean()), 2) if deduped else 1.0
#     signal = {
#         "date": utc_now()[:10],
#         "retailer": "Dollar Tree",
#         "region": country.upper(),
#         "source": "GNews / Google News RSS",
#         "signal_area": "Retail News",
#         "signal_name": "news_risk_score",
#         "signal_value": len(deduped),
#         "risk_score": score,
#         "confidence": "Medium",
#         "business_impact": "Recent news can reveal competitor moves, pricing pressure, recalls, store changes, and supply-chain risk.",
#         "recommended_action": "Review high-risk articles and use NVIDIA classification before executive distribution.",
#         "raw_reference": f"{len(deduped)} articles",
#     }
#     status = "success" if deduped or not errors else "failed"
#     return {"status": status, "source": "GNews", "error": "; ".join(errors), "raw": deduped, "rows": [signal], "items": deduped}


# def collect_apify_trends(token: str, keywords: List[str], geo: str, time_range: str) -> Dict[str, Any]:
#     if not token:
#         return {"status": "skipped", "source": "Apify Trends", "error": "No Apify token provided.", "raw": None, "rows": [], "items": []}
#     try:
#         from apify_client import ApifyClient
#     except ImportError:
#         return {"status": "failed", "source": "Apify Trends", "error": "apify-client is not installed.", "raw": None, "rows": [], "items": []}
#     try:
#         client = ApifyClient(token)
#         run_input = {"geo": geo, "searchTerms": keywords, "timeRange": time_range}
#         run = client.actor("apify/google-trends-scraper").call(run_input=run_input)
#         items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
#     except Exception as exc:
#         return {"status": "failed", "source": "Apify Trends", "error": str(exc), "raw": None, "rows": [], "items": []}

#     region_rows = []
#     for item in items:
#         keyword = item.get("searchTerm")
#         for rank, region in enumerate(item.get("interestBySubregion", []) or [], start=1):
#             values = region.get("value") or []
#             if values:
#                 region_rows.append(
#                     {
#                         "keyword": keyword,
#                         "region": region.get("geoName", ""),
#                         "interest_score": values[0],
#                         "rank": rank,
#                     }
#                 )
#     top_score = max([float(x["interest_score"]) for x in region_rows], default=0.0)
#     signal_score = round(min(10.0, max(1.0, top_score / 10.0)), 2) if top_score else 1.0
#     signal = {
#         "date": utc_now()[:10],
#         "retailer": "Dollar Tree",
#         "region": geo,
#         "source": "Apify Google Trends",
#         "signal_area": "Search Demand",
#         "signal_name": "search_demand_score",
#         "signal_value": top_score,
#         "risk_score": signal_score,
#         "confidence": "Medium",
#         "business_impact": "Search interest can reveal early demand shifts, promotional interest, and seasonal spikes.",
#         "recommended_action": "Compare rising regions against sales, inventory, and competitor promotion calendars.",
#         "raw_reference": f"{len(region_rows)} regional trend rows",
#     }
#     return {"status": "success", "source": "Apify Trends", "error": "", "raw": items, "rows": [signal], "items": region_rows}


# def generate_fallback_brief(feature_df: pd.DataFrame, retailer: str, region: str) -> str:
#     if feature_df.empty:
#         return "No signals were collected. Add at least one enabled source and run the workbench again."
#     strongest = feature_df.sort_values("risk_score", ascending=False).head(3)
#     lines = [
#         f"Market intelligence summary for {retailer} in {region}.",
#         "",
#         "Top signals:",
#     ]
#     for _, row in strongest.iterrows():
#         lines.append(
#             f"- {row['signal_area']}: {row['risk_score']}/10 from {row['source']}. {row['business_impact']}"
#         )
#     lines.extend(
#         [
#             "",
#             "Recommended focus:",
#             "Use these signals as forecast-ready external features, then validate them against internal POS, category, and inventory data before operational decisions.",
#         ]
#     )
#     return "\n".join(lines)


# def generate_nvidia_brief(api_key: str, model: str, feature_df: pd.DataFrame, articles: List[Dict[str, Any]], retailer: str, region: str) -> Tuple[str, str]:
#     if not api_key:
#         return generate_fallback_brief(feature_df, retailer, region), "fallback"
#     payload = {
#         "features": feature_df.to_dict(orient="records")[:12],
#         "articles": articles[:8],
#     }
#     system = (
#         "You are a retail market intelligence analyst. Ground your answer only in the supplied signals. "
#         "Write concise executive guidance for buyers, category managers, demand planners, and supply-chain teams."
#     )
#     user = f"""
#     Retailer: {retailer}
#     Region: {region}

#     Signal payload:
#     {json.dumps(payload, indent=2)[:9000]}

#     Produce:
#     1. Top 3 insights
#     2. Forecasting relevance
#     3. Recommended actions
#     4. Confidence and limitations
#     """
#     ok, data, msg = safe_request(
#         NVIDIA_CHAT_URL,
#         method="POST",
#         headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
#         json_body={
#             "model": model,
#             "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
#             "temperature": 0.25,
#             "max_tokens": 900,
#         },
#         timeout=60,
#     )
#     if not ok:
#         return generate_fallback_brief(feature_df, retailer, region), f"fallback: {msg}"
#     content = data.get("choices", [{}])[0].get("message", {}).get("content")
#     content_text = str(content or "").strip()
#     return content_text or generate_fallback_brief(feature_df, retailer, region), "nvidia"


# def render_metric_card(title: str, value: str, note: str, confidence: str = "") -> None:
#     pill = f'<span class="pill {confidence_class(confidence)}">{confidence}</span>' if confidence else ""
#     html = (
#         '<div class="metric-card">'
#         f'<div class="metric-label">{title}</div>'
#         f'<div class="metric-value">{value}</div>'
#         f"{pill}"
#         f'<div class="metric-note">{note}</div>'
#         "</div>"
#     )
#     st.markdown(
#         html,
#         unsafe_allow_html=True,
#     )


# def compute_composite_scores(feature_df: pd.DataFrame) -> Dict[str, float]:
#     if feature_df.empty:
#         return {"Market Opportunity": 0.0, "Market Risk": 0.0, "Forecast Impact": 0.0}
#     area_scores: Dict[str, float] = {}
#     for _, row in feature_df.iterrows():
#         if pd.isna(row.get("risk_score")):
#             continue
#         area = str(row["signal_area"])
#         area_scores[area] = max(area_scores.get(area, 0.0), float(row["risk_score"]))
#     opportunity = max(
#         area_scores.get("Retail News", 0.0),
#         area_scores.get("Search Demand", 0.0),
#         area_scores.get("Category CPI", 0.0) * 0.7,
#     )
#     risk = max(
#         area_scores.get("Product Recalls", 0.0),
#         area_scores.get("Inflation", 0.0),
#         area_scores.get("Category CPI", 0.0),
#     )
#     impact = min(10.0, (opportunity * 0.45) + (risk * 0.45) + (len(feature_df) * 0.25))
#     return {
#         "Market Opportunity": round(opportunity, 2),
#         "Market Risk": round(risk, 2),
#         "Forecast Impact": round(impact, 2),
#     }


# def build_recommended_actions(feature_df: pd.DataFrame, results: Dict[str, Dict[str, Any]]) -> List[Dict[str, str]]:
#     actions = []
#     if feature_df.empty:
#         return [{"label": "Setup", "title": "Run signal sources", "body": "Enable at least one source to generate forecast-ready rows."}]
#     top_rows = feature_df.sort_values("risk_score", ascending=False).head(3)
#     for _, row in top_rows.iterrows():
#         actions.append(
#             {
#                 "label": str(row["signal_area"]),
#                 "title": str(row["signal_name"]).replace("_", " ").title(),
#                 "body": str(row["recommended_action"]),
#             }
#         )
#     apify_result = results.get("apify")
#     if apify_result and apify_result.get("status") == "failed":
#         actions.append(
#             {
#                 "label": "Apify",
#                 "title": "Search demand skipped",
#                 "body": "Apify connected but failed during collection. If usage limit is exceeded, rerun after quota reset or keep MVP on public sources.",
#             }
#         )
#     return actions[:4]


# def render_action_card(label: str, title: str, body: str) -> None:
#     html = (
#         '<div class="action-card">'
#         f'<div class="action-label">{escape(label)}</div>'
#         f'<div class="action-title">{escape(title)}</div>'
#         f'<div class="action-body">{escape(body)}</div>'
#         "</div>"
#     )
#     st.markdown(html, unsafe_allow_html=True)


# def render_source_tile(name: str, status: str, detail: str) -> None:
#     status_class = "pill-high" if status == "Active" else "pill-medium" if status == "Optional" else "pill-low"
#     html = (
#         '<div class="source-tile">'
#         f'<div class="source-name">{escape(name)} <span class="pill {status_class}" style="margin-left:6px;margin-top:0;">{escape(status)}</span></div>'
#         f'<div class="source-meta">{escape(detail)}</div>'
#         "</div>"
#     )
#     st.markdown(html, unsafe_allow_html=True)


# def render_workflow_strip() -> None:
#     steps = [
#         ("01", "Collect APIs"),
#         ("02", "Clean records"),
#         ("03", "Score signals"),
#         ("04", "Generate brief"),
#         ("05", "Export features"),
#     ]
#     html = "<div class='workflow'>" + "".join(
#         f"<div class='workflow-step'><span>{num}</span>{escape(label)}</div>" for num, label in steps
#     ) + "</div>"
#     st.markdown(html, unsafe_allow_html=True)


# def render_score_chart(feature_df: pd.DataFrame) -> None:
#     if feature_df.empty:
#         st.info("No feature rows yet.")
#         return
#     fig = go.Figure(
#         go.Bar(
#             x=feature_df["risk_score"],
#             y=feature_df["signal_area"],
#             orientation="h",
#             marker_color=["#22C55E" if x < 5 else "#F59E0B" if x < 8 else "#EF4444" for x in feature_df["risk_score"]],
#             text=feature_df["risk_score"],
#             textposition="auto",
#         )
#     )
#     fig.update_layout(
#         height=280,
#         margin={"l": 10, "r": 20, "t": 10, "b": 10},
#         xaxis={"range": [0, 10], "title": "Score"},
#         yaxis={"title": ""},
#         plot_bgcolor="#FFFFFF",
#         paper_bgcolor="#FFFFFF",
#     )
#     st.plotly_chart(fig, width="stretch")


# def parse_lines(text: str) -> List[str]:
#     return [line.strip() for line in text.splitlines() if line.strip()]


# with st.sidebar:
#     st.title("Workbench Setup")
#     st.caption("Credentials are used only for this Streamlit session.")

#     nvidia_key = st.text_input("NVIDIA API key", value=os.getenv("NVIDIA_API_KEY", ""), type="password")
#     nvidia_model = st.text_input("NVIDIA model", value=os.getenv("NVIDIA_MODEL", DEFAULT_NVIDIA_MODEL))
#     apify_token = st.text_input("Apify token", value=os.getenv("APIFY_API_TOKEN", ""), type="password")
#     bls_key = st.text_input("BLS API key optional", value=os.getenv("BLS_API_KEY", ""), type="password")

#     st.markdown('<div class="small-header">Retail Context</div>', unsafe_allow_html=True)
#     retailer = st.text_input("Retailer", value="Dollar Tree")
#     region = st.text_input("Region", value="US")
#     country = st.selectbox("News country", ["US", "CA", "GB", "AE", "IN"], index=0)
#     language = st.selectbox("News language", ["en", "es", "fr", "ar", "hi"], index=0)

#     st.markdown('<div class="small-header">Signal Sources</div>', unsafe_allow_html=True)
#     use_gnews = st.checkbox("Retail news via GNews/RSS", value=True)
#     use_bls = st.checkbox("Inflation via BLS CPI", value=True)
#     use_fda = st.checkbox("Product recalls via openFDA", value=True)
#     use_apify = st.checkbox("Search demand via Apify Trends", value=False)

#     st.markdown('<div class="small-header">Run Controls</div>', unsafe_allow_html=True)
#     validate_button = st.button("Validate credentials", width="stretch")
#     run_button = st.button("Run intelligence", type="primary", width="stretch")


# DEFAULT_NEWS_KEYWORDS = "\n".join(
#     [
#         "Dollar Tree inflation",
#         "Dollar Tree prices",
#         "Dollar Tree store closures",
#         "Dollar Tree recall",
#         "discount retail tariffs",
#         "Dollar General promotion",
#     ]
# )
# DEFAULT_TRENDS_KEYWORDS = "\n".join(
#     [
#         "Dollar Tree sales",
#         "Dollar Tree coupons",
#         "Dollar Tree near me",
#         "Dollar Tree groceries",
#         "cheap groceries",
#     ]
# )

# st.session_state.setdefault("news_keywords_text", DEFAULT_NEWS_KEYWORDS)
# st.session_state.setdefault("trends_keywords_text", DEFAULT_TRENDS_KEYWORDS)
# st.session_state.setdefault("gnews_period", "7d")
# st.session_state.setdefault("max_news", 24)
# st.session_state.setdefault("fda_query", "product_description:(snacks OR candy OR beverages)")
# st.session_state.setdefault("fda_limit", 8)
# st.session_state.setdefault("apify_geo", "US")
# st.session_state.setdefault("apify_time_range", "today 1-m")
# st.session_state.setdefault("workbench_view", "Configure")
# if st.session_state.pop("force_results_view", False):
#     st.session_state["workbench_view"] = "Results"

# st.markdown('<div class="accent-bar"></div>', unsafe_allow_html=True)
# st.markdown(
#     "<div class='topbar'>"
#     "<div class='tt'>Market Intelligence <span>/ Dollar Tree · External Signals</span></div>"
#     "<div class='tbadge'>AI Workbench</div>"
#     "<div style='margin-left:auto;display:flex;align-items:center;gap:8px;'>"
#     "<span class='ldot'></span>"
#     "<span style='font-size:10px;color:var(--t3);font-weight:800;'>Live API Mode</span>"
#     "<div style='width:28px;height:28px;border-radius:7px;background:linear-gradient(135deg,var(--odk),var(--or));display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:900;color:#fff;'>DT</div>"
#     "</div></div>",
#     unsafe_allow_html=True,
# )

# hero_left, hero_right = st.columns([2.2, 0.9], vertical_alignment="center")
# with hero_left:
#     st.markdown(
#         "<div class='hero-shell'>"
#         "<div class='hero-kicker'><span class='ldot'></span> External Signal Layer</div>"
#         "<h1 class='hero-title'>Dollar Tree Market Intelligence Command Center</h1>"
#         "<p class='hero-copy'>A retail-grade workbench that turns news, CPI, recalls, search demand, and API health into forecast-ready features, composite risk scores, and buyer actions.</p>"
#         "</div>",
#         unsafe_allow_html=True,
#     )
# with hero_right:
#     active_sources = sum([use_gnews, use_bls, use_fda, use_apify])
#     st.markdown(
#         "<div class='hero-side'>"
#         "<div class='hero-side-label'>Run Profile</div>"
#         f"<div class='hero-side-row'><span>Retailer</span><strong>{escape(retailer)}</strong></div>"
#         f"<div class='hero-side-row'><span>Region</span><strong>{escape(region)}</strong></div>"
#         f"<div class='hero-side-row'><span>Sources</span><strong>{active_sources} enabled</strong></div>"
#         f"<div class='hero-side-row'><span>LLM</span><strong>{'NVIDIA' if nvidia_key.strip() else 'Fallback'}</strong></div>"
#         "</div>",
#         unsafe_allow_html=True,
#     )

# view = st.segmented_control(
#     "Workbench view",
#     ["Configure", "Results", "Raw Data"],
#     required=True,
#     label_visibility="collapsed",
#     key="workbench_view",
#     width="content",
# )

# if view == "Configure":
#     st.markdown('<div class="small-header">Signal Source Stack</div>', unsafe_allow_html=True)
#     src_cols = st.columns(4)
#     source_specs = [
#         ("GNews/RSS", "Active" if use_gnews else "Off", "Retail articles, sentiment hints, source confidence."),
#         ("BLS CPI", "Active" if use_bls else "Off", "Headline and category inflation pressure features."),
#         ("openFDA", "Active" if use_fda else "Off", "Recall severity, UPC extraction, SKU-match prep."),
#         ("Apify Trends", "Optional" if use_apify else "Off", "Search interest and regional demand spikes."),
#     ]
#     for col, spec in zip(src_cols, source_specs):
#         with col:
#             render_source_tile(*spec)

#     left, right = st.columns([1.15, 0.85])
#     with left:
#         st.markdown('<div class="console-panel">', unsafe_allow_html=True)
#         st.subheader("Signal Keywords")
#         st.text_area("GNews keywords", key="news_keywords_text", height=170)
#         st.text_area("Apify Google Trends keywords", key="trends_keywords_text", height=150)
#         st.markdown("</div>", unsafe_allow_html=True)
#     with right:
#         st.markdown('<div class="console-panel">', unsafe_allow_html=True)
#         st.subheader("Collector Settings")
#         st.selectbox("GNews period", ["1d", "7d", "30d", "3m"], key="gnews_period")
#         st.slider("Max news results", min_value=5, max_value=60, step=1, key="max_news")
#         st.text_input("FDA recall search", key="fda_query")
#         st.slider("FDA recall limit", min_value=1, max_value=25, key="fda_limit")
#         st.text_input("Apify geo", key="apify_geo")
#         st.selectbox("Apify time range", ["today 7-d", "today 1-m", "today 3-m", "today 12-m"], key="apify_time_range")
#         st.markdown("</div>", unsafe_allow_html=True)

#     st.markdown('<div class="small-header">Agent Flow</div>', unsafe_allow_html=True)
#     render_workflow_strip()


# if validate_button:
#     with st.spinner("Validating credentials..."):
#         n_ok, n_msg = validate_nvidia(nvidia_key.strip(), nvidia_model.strip())
#         a_ok, a_msg = validate_apify(apify_token.strip())
#     st.session_state["validation"] = {"nvidia": (n_ok, n_msg), "apify": (a_ok, a_msg)}

# if "validation" in st.session_state:
#     n_ok, n_msg = st.session_state["validation"]["nvidia"]
#     a_ok, a_msg = st.session_state["validation"]["apify"]
#     st.info(f"NVIDIA: {'Connected' if n_ok else 'Not connected'} - {n_msg}")
#     st.info(f"Apify: {'Connected' if a_ok else 'Not connected'} - {a_msg}")


# if run_button:
#     news_keywords = parse_lines(st.session_state.get("news_keywords_text", DEFAULT_NEWS_KEYWORDS))
#     trends_keywords = parse_lines(st.session_state.get("trends_keywords_text", DEFAULT_TRENDS_KEYWORDS))
#     results: Dict[str, Dict[str, Any]] = {}
#     all_rows: List[Dict[str, Any]] = []
#     all_articles: List[Dict[str, Any]] = []

#     progress = st.progress(0, text="Starting collectors")
#     steps = [
#         ("gnews", use_gnews),
#         ("bls", use_bls),
#         ("fda", use_fda),
#         ("apify", use_apify),
#     ]
#     active_steps = [step for step in steps if step[1]]
#     total = max(1, len(active_steps))
#     completed = 0

#     if use_gnews:
#         progress.progress(completed / total, text="Collecting retail news")
#         results["gnews"] = collect_gnews(
#             news_keywords,
#             country,
#             language,
#             st.session_state.get("gnews_period", "7d"),
#             int(st.session_state.get("max_news", 24)),
#         )
#         all_rows.extend(results["gnews"].get("rows", []))
#         all_articles.extend(results["gnews"].get("items", []))
#         completed += 1

#     if use_bls:
#         progress.progress(completed / total, text="Collecting CPI inflation")
#         results["bls"] = collect_bls_cpi(bls_key.strip())
#         all_rows.extend(results["bls"].get("rows", []))
#         completed += 1

#     if use_fda:
#         progress.progress(completed / total, text="Collecting FDA recalls")
#         results["fda"] = collect_fda_recalls(
#             st.session_state.get("fda_query", "product_description:(snacks OR candy OR beverages)"),
#             int(st.session_state.get("fda_limit", 8)),
#         )
#         all_rows.extend(results["fda"].get("rows", []))
#         completed += 1

#     if use_apify:
#         progress.progress(completed / total, text="Collecting Google Trends via Apify")
#         results["apify"] = collect_apify_trends(
#             apify_token.strip(),
#             trends_keywords,
#             st.session_state.get("apify_geo", "US"),
#             st.session_state.get("apify_time_range", "today 1-m"),
#         )
#         all_rows.extend(results["apify"].get("rows", []))
#         completed += 1

#     progress.progress(1.0, text="Generating intelligence brief")
#     feature_df = pd.DataFrame(all_rows)
#     if not feature_df.empty:
#         feature_df["retailer"] = retailer
#         feature_df["region"] = feature_df["region"].replace({"US": region}) if region else feature_df["region"]
#     brief, brief_source = generate_nvidia_brief(nvidia_key.strip(), nvidia_model.strip(), feature_df, all_articles, retailer, region)

#     st.session_state["run"] = {
#         "timestamp": utc_now(),
#         "results": results,
#         "feature_df": feature_df,
#         "articles": all_articles,
#         "brief": brief,
#         "brief_source": brief_source,
#     }
#     st.session_state["force_results_view"] = True
#     progress.empty()
#     st.rerun()


# if view == "Results":
#     run = st.session_state.get("run")
#     if not run:
#         st.markdown(
#             "<div class='empty-console'>"
#             "<div>"
#             "<div class='hero-kicker'>Ready for first run</div>"
#             "<div class='empty-title'>No intelligence run yet.</div>"
#             "<div class='empty-body'>Configure the signal stack, then click <strong>Run intelligence</strong> in the sidebar. The agent will collect public signals, normalize them into feature rows, score the risk/opportunity surface, and produce an executive brief.</div>"
#             "</div>"
#             "<div class='hero-side' style='min-width:260px;'>"
#             "<div class='hero-side-label'>MVP Output</div>"
#             "<div class='hero-side-row'><span>Feature rows</span><strong>CSV / JSON</strong></div>"
#             "<div class='hero-side-row'><span>Scores</span><strong>0-10</strong></div>"
#             "<div class='hero-side-row'><span>Brief</span><strong>NVIDIA / fallback</strong></div>"
#             "</div>"
#             "</div>",
#             unsafe_allow_html=True,
#         )
#         render_workflow_strip()
#     else:
#         feature_df = run["feature_df"]
#         st.markdown(
#             f"<div class='note-box'><strong>Last run:</strong> {escape(run['timestamp'])} UTC &nbsp; | &nbsp; <strong>Brief source:</strong> {escape(str(run['brief_source']))}</div>",
#             unsafe_allow_html=True,
#         )
#         cols = st.columns(4)
#         if feature_df.empty:
#             for col, title in zip(cols, ["Signals", "Avg Score", "Highest Score", "Brief"]):
#                 with col:
#                     render_metric_card(title, "0", "No successful feature rows yet.")
#         else:
#             avg_score = round(float(feature_df["risk_score"].mean()), 2)
#             top = feature_df.sort_values("risk_score", ascending=False).iloc[0]
#             composite_scores = compute_composite_scores(feature_df)
#             with cols[0]:
#                 render_metric_card("Signals", str(len(feature_df)), "Forecast-ready rows generated.")
#             with cols[1]:
#                 render_metric_card("Average Score", str(avg_score), f"{risk_band(avg_score)} overall signal intensity.")
#             with cols[2]:
#                 render_metric_card("Top Signal", str(top["risk_score"]), str(top["signal_area"]), str(top["confidence"]))
#             with cols[3]:
#                 render_metric_card("News Articles", str(len(run["articles"])), "Deduplicated retail news items.")

#             st.markdown('<div class="small-header">Composite Agent Scores</div>', unsafe_allow_html=True)
#             cscore_cols = st.columns(3)
#             for col, (name, score) in zip(cscore_cols, composite_scores.items()):
#                 with col:
#                     render_metric_card(name, str(score), f"{risk_band(score)} priority for planning.")

#             st.markdown('<div class="small-header">Recommended Actions</div>', unsafe_allow_html=True)
#             action_cols = st.columns(4)
#             for col, action in zip(action_cols, build_recommended_actions(feature_df, run["results"])):
#                 with col:
#                     render_action_card(action["label"], action["title"], action["body"])

#             st.markdown('<div class="small-header">Signal Scores</div>', unsafe_allow_html=True)
#             render_score_chart(feature_df)

#             st.markdown('<div class="small-header">Forecast Feature Table</div>', unsafe_allow_html=True)
#             st.dataframe(feature_df, width="stretch", hide_index=True)

#             csv_data = feature_df.to_csv(index=False).encode("utf-8")
#             json_data = json.dumps(feature_df.to_dict(orient="records"), indent=2).encode("utf-8")
#             c1, c2 = st.columns([1, 1])
#             with c1:
#                 st.download_button("Download forecast_features.csv", csv_data, "forecast_features.csv", "text/csv", width="stretch")
#             with c2:
#                 st.download_button("Download normalized_signals.json", json_data, "normalized_signals.json", "application/json", width="stretch")

#         st.markdown('<div class="small-header">Executive Brief</div>', unsafe_allow_html=True)
#         st.markdown(f'<div class="brief-box">{run["brief"].replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)


# if view == "Raw Data":
#     run = st.session_state.get("run")
#     if not run:
#         st.markdown(
#             "<div class='empty-console'>"
#             "<div>"
#             "<div class='hero-kicker'>Raw Evidence</div>"
#             "<div class='empty-title'>Collector payloads will appear after a run.</div>"
#             "<div class='empty-body'>This view is intentionally evidence-first: GNews articles, CPI JSON, FDA recall records, Apify errors, and cleaned item tables stay inspectable for validation.</div>"
#             "</div>"
#             "</div>",
#             unsafe_allow_html=True,
#         )
#     else:
#         for name, result in run["results"].items():
#             status = result.get("status", "unknown")
#             label = f"{name.upper()} - {status}"
#             with st.expander(label, expanded=False):
#                 if result.get("error"):
#                     st.warning(result["error"])
#                 if result.get("items"):
#                     st.dataframe(pd.DataFrame(result["items"]), width="stretch", hide_index=True)
#                 raw_preview = result.get("raw")
#                 if raw_preview is not None:
#                     st.code(json.dumps(raw_preview, indent=2, default=str)[:7000], language="json")


# st.markdown(
#     """
#     <p class="subtle">
#     Note: Google Trends values are relative indexes, GNews is a lightweight news signal, and recall data should be matched
#     against internal SKU/UPC and inventory records before operational decisions.
#     </p>
#     """,
#     unsafe_allow_html=True,
# )

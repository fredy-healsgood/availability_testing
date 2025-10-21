# app.py — Flexzo Availability Tester (Health / Teach)
# ---------------------------------------------------
# Quick start (Windows/uv):
#   uv venv .venv
#   .venv\Scripts\Activate.ps1
#   uv pip install streamlit pandas requests websocket-client openpyxl
#   uv run streamlit run app.py

import json
import re
import time
import ast
from io import BytesIO
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
from websocket import create_connection, WebSocketTimeoutException

# ======================= Page & Global Styles =======================

st.set_page_config(
    page_title="Flexzo Availability — Health/Teach",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Flexzo purple theme
st.markdown(
    """
    <style>
      :root {
        --fxz-bg: #130b2b;
        --fxz-bg-2: #1e0f4a;
        --fxz-card: #2a1363;
        --fxz-stroke: #5a2ab8;
        --fxz-text: #f3ecff;
        --fxz-muted: #cbb8ff;
        --fxz-chip: #3a1a87;
        --fxz-accent: #ff69ff;
        --fxz-accent-2: #58d8ff;
        --fxz-ok: #22c55e;
        --fxz-warn: #f59e0b;
        --fxz-bad: #ef4444;
      }
      html, body, .block-container { background: radial-gradient(1200px 600px at 20% -10%, #3a1a87 0%, #1b0b3d 35%, #10072b 100%); }
      .block-container { padding-top: 1.0rem; }
      h1, h2, h3, h4, h5, h6, p, label, span, div { color: var(--fxz-text); }
      .fxz-hero {
        background: linear-gradient(135deg, rgba(136,86,255,0.35), rgba(88,216,255,0.15));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 24px 26px;
        margin-bottom: 16px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.25), 0 0 0 1px rgba(255,255,255,0.03) inset;
      }
      .fxz-hero h1 { margin: 0; font-weight: 900; font-size: 2.0rem; letter-spacing: 0.3px; }
      .fxz-hero p { margin: 6px 0 0; color: var(--fxz-muted); }
      .fxz-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 16px; padding: 16px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.25), 0 0 0 1px rgba(255,255,255,0.03) inset;
      }
      .fxz-metric { display:flex; gap:10px; align-items:center; color: var(--fxz-text); }
      .fxz-chip {
        display:inline-block; padding: 4px 10px; border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.18); background: var(--fxz-chip);
        color: #e9dfff; font-size: 0.85rem;
      }
      .fxz-divider { height: 1px; background: rgba(255,255,255,0.10); margin: 10px 0 12px; }
      .stSelectbox > label, .stTextInput > label, .stNumberInput > label { color: var(--fxz-muted) !important; }
      .stDownloadButton > button, .stButton > button {
        border-radius: 999px !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        background: linear-gradient(180deg, rgba(255,255,255,0.10), rgba(255,255,255,0.03)) !important;
        color: var(--fxz-text) !important;
      }
      .ok { color: var(--fxz-ok); } .warn { color: var(--fxz-warn); } .bad { color: var(--fxz-bad); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ======================= Config =======================

API_HOST = "35.197.253.66"          # QA host
FLOWER_URL = "http://34.142.119.139/"

# ======================= Helpers =======================

def _fix_timestr(s):
    """Normalize spaced time strings like '08: 00: 00' → '08:00:00'."""
    if not s:
        return s
    return re.sub(r"(\d{2}):\s*(\d{2}):\s*(\d{2})", r"\1:\2:\3", str(s).strip())

def _parse_ws_payload(text):
    """Robustly parse WS frames: JSON, JSON with spaced H:M:S, Python dict strings, or embedded {...}."""
    if isinstance(text, (bytes, bytearray)):
        text = text.decode("utf-8", "ignore")
    s = (text or "").strip()
    if not s:
        raise ValueError("empty WebSocket frame")

    # 1) Strict JSON
    try:
        return json.loads(s)
    except Exception:
        pass

    # 2) Normalize spaced times/newlines → JSON
    def _join_hms(m): return f"{m.group(1)}:{m.group(2)}:{m.group(3)}"
    fixed = re.sub(r"(\d{2}):\s*(\d{2}):\s*(\d{2})", _join_hms, s)
    fixed = fixed.replace("\r", "").replace("\n", " ")
    try:
        return json.loads(fixed)
    except Exception:
        pass

    # 3) Python literal (single-quoted dict)
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 4) Extract first {...} and parse
    m = re.search(r"(\{[\s\S]*\})", s)
    if m:
        inner = m.group(1)
        try:
            return json.loads(inner)
        except Exception:
            try:
                obj2 = ast.literal_eval(inner)
                if isinstance(obj2, dict):
                    return obj2
            except Exception:
                pass

    raise ValueError("unrecognized WebSocket payload format")

def _post_request(project, content, callback_uri, timeout=30):
    """POST to get_availability_alt and return task_id."""
    url = f"http://{API_HOST}/ai/{project}/crm/get_availability_alt"
    payload = {"content": content}
    if callback_uri:
        payload["callbackUri"] = callback_uri
    r = requests.post(url, json=payload, headers={"accept": "application/json"}, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if "task_id" not in data:
        raise RuntimeError(f"Expected task_id in response, got: {data}")
    return data["task_id"]

def _fetch_ws(project, task_id, timeout=60, debug=False):
    """Connect to task WS and wait until a parseable payload arrives or we time out."""
    url = f"ws://{API_HOST}/ws/ai/{project}/crm/task/{task_id}"
    ws = create_connection(url, timeout=timeout)
    end = time.time() + timeout
    try:
        try:
            ws.settimeout(1)    # poll every 1s so we can timeout cleanly
        except Exception:
            pass
        last_non_json = None
        while time.time() < end:
            try:
                msg = ws.recv()
            except WebSocketTimeoutException:
                continue
            except Exception as e:
                return {"error": f"WebSocket error: {e}", "task_id": task_id}
            if debug:
                print(f"WS frame (first 200 chars): {repr(msg)[:200]}")
            if not msg or (isinstance(msg, str) and not msg.strip()):
                continue
            try:
                return _parse_ws_payload(msg)
            except Exception:
                last_non_json = msg
                continue
        return {"error": f"WebSocket timeout after {timeout}s", "task_id": task_id, "last_frame": (repr(last_non_json)[:500] if last_non_json is not None else None)}
    finally:
        try:
            ws.close()
        except Exception:
            pass

def _duration_hours(start, end):
    if not start or not end:
        return None
    try:
        s = _fix_timestr(start); e = _fix_timestr(end)
        t1 = datetime.strptime(s, "%H:%M:%S")
        t2 = datetime.strptime(e, "%H:%M:%S")
        if s == "00:00:00" and e in ("23:59:59", "24:00:00"):
            return 24.0
        delta = (t2 - t1).seconds / 3600
        return round(delta, 2)
    except Exception:
        return None

def _flatten_result(result, project, content, task_id):
    """Long/normalized rows: one row per availability OR not_available item."""
    rows = []
    avail = result.get("availability", []) or []
    not_av = result.get("not_available", []) or []
    reasoning = result.get("reasoning")

    for i, e in enumerate(avail):
        t = e.get("time", {}) if isinstance(e, dict) else {}
        start = _fix_timestr(t.get("start"))
        end = _fix_timestr(t.get("end"))
        is_full = (e.get("availabilityTime") == "fullDay") or (start == "00:00:00" and end in ("23:59:59", "24:00:00"))
        rows.append({
            "project": project,
            "task_id": task_id,
            "content": content,
            "status": "available",
            "date": e.get("date"),
            "start": start,
            "end": end,
            "availabilityTime": e.get("availabilityTime"),
            "isFullDay": is_full,
            "durationHours": _duration_hours(start, end),
            "spanIndex": i,
            "reasoning": reasoning,
            "raw_json": json.dumps(result, ensure_ascii=False)
        })

    for j, d in enumerate(not_av):
        rows.append({
            "project": project,
            "task_id": task_id,
            "content": content,
            "status": "not_available",
            "date": d,
            "start": None,
            "end": None,
            "availabilityTime": None,
            "isFullDay": None,
            "durationHours": None,
            "spanIndex": j,
            "reasoning": reasoning,
            "raw_json": json.dumps(result, ensure_ascii=False)
        })

    if not rows:
        rows.append({
            "project": project,
            "task_id": task_id,
            "content": content,
            "status": "no_parse",
            "date": None,
            "start": None,
            "end": None,
            "availabilityTime": None,
            "isFullDay": None,
            "durationHours": None,
            "spanIndex": None,
            "reasoning": reasoning or result.get("error"),
            "raw_json": json.dumps(result, ensure_ascii=False)
        })

    return pd.DataFrame(rows)

def _to_excel(df):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="results")
    buffer.seek(0)
    return buffer.read()

def _sample_csv():
    df = pd.DataFrame({"content": [
        "i will be available tomorrow",
        "i will be unavailable tomorrow",
        "i will be available next week",
        "free tomorrow morning",
        "tomorrow after 3pm",
    ]})
    return df.to_csv(index=False).encode("utf-8")

# ---- concat & dedupe utilities (fix FutureWarning + unhashables) ----

def _concat_nonempty(frames, columns=None):
    nonempty = [f for f in frames if f is not None and isinstance(f, pd.DataFrame) and not f.empty]
    if not nonempty:
        return pd.DataFrame(columns=columns or [])
    return pd.concat(nonempty, ignore_index=True)

def _stringify_unhashables(df):
    """Make dict/list cells safe for drop_duplicates."""
    out = df.copy()
    for c in out.columns:
        if out[c].map(lambda x: isinstance(x, (dict, list))).any():
            out[c] = out[c].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else x)
    return out

def _dedupe(df, mode: str):
    if df is None or df.empty or mode == "off":
        return df
    subsets = {
        "by task_id": ["task_id"],
        "by values": ["project", "content", "status", "date", "start", "end", "availabilityTime"],
    }
    subset = subsets.get(mode, subsets["by values"])
    subset = [c for c in subset if c in df.columns]  # guard
    if not subset:
        return df
    safe = _stringify_unhashables(df)
    return safe.drop_duplicates(subset=subset, keep="last").reset_index(drop=True)

# ======================= Header =======================

st.markdown(
    """
    <div class="fxz-hero">
      <h1>Flexzo Teach/Health — Availability Generator & Parser</h1>
      <p>Create and validate availability from natural language. Upload CSV/XLSX, parse via QA APIs, and export clean results.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(
        f"<div class='fxz-card fxz-metric'>🌐 <b>API Host:</b> <span class='fxz-chip'>{API_HOST}</span></div>",
        unsafe_allow_html=True
    )
with m2:
    st.markdown(
        f"<div class='fxz-card fxz-metric'>📊 <b>Flower:</b> <a href='{FLOWER_URL}' target='_blank' class='fxz-chip'>{FLOWER_URL}</a></div>",
        unsafe_allow_html=True
    )
with m3:
    st.markdown(
        "<div class='fxz-card fxz-metric'>⏱️ <b>WS Timeout:</b> <span class='fxz-chip'>60s (default)</span></div>",
        unsafe_allow_html=True
    )

# ======================= Sidebar =======================

with st.sidebar:
    st.header("⚙️ Settings")
    project = st.selectbox("Project", options=["health", "teach"], index=0)
    callback_uri = st.text_input("Optional callbackUri", placeholder="https://... (optional)")
    ws_timeout = st.number_input("WebSocket timeout (s)", min_value=5, max_value=300, value=60)
    debug_ws = st.checkbox("Debug WebSocket frames", value=False)
    dedupe_mode = st.selectbox(
        "De-duplicate",
        options=["by values", "off", "by task_id"],
        index=0,
        help="by values = unique by project+content+status+date+time; by task_id = unique per server task; off = keep all rows",
    )
    st.caption("QA host; server timezone assumed Africa/Harare.")

# ======================= Session State =======================

if "results_df" not in st.session_state:
    st.session_state["results_df"] = pd.DataFrame(columns=[
        "project","task_id","content","status","date","start","end",
        "availabilityTime","isFullDay","durationHours","spanIndex",
        "reasoning","raw_json"
    ])

# ======================= Tabs =======================

tab1, tab2, tab3, tab4 = st.tabs(["🧪 Generator", "📂 Strategic Batches", "🧰 Test Pack", "📈 Results"])

# ----------------------- Tab 1: Single -----------------------

with tab1:
    st.markdown("#### Generation Mode")
    st.selectbox("Mode", ["Strategic Testing Batches", "Ad-hoc Prompt"], index=0, key="mode_select")

    st.markdown('<div class="fxz-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Enter availability text")

    content = st.text_area(
        "Availability text",
        placeholder="e.g. i will be available next week",
        height=100,
        label_visibility="collapsed",
        key="single_text",
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    if c1.button("Tomorrow", key="chip1", width="stretch"):
        st.session_state["single_text"] = "i will be available tomorrow"
        st.rerun()
    if c2.button("Tomorrow morning", key="chip2", width="stretch"):
        st.session_state["single_text"] = "free tomorrow morning"
        st.rerun()
    if c3.button("After 3pm", key="chip3", width="stretch"):
        st.session_state["single_text"] = "tomorrow after 3pm"
        st.rerun()
    if c4.button("Full time", key="chip4", width="stretch"):
        st.session_state["single_text"] = "I am full time"
        st.rerun()
    if c5.button("Next week", key="chip5", width="stretch"):
        st.session_state["single_text"] = "i will be available next week"
        st.rerun()

    b1, b2, _ = st.columns([1, 1, 6])
    if b1.button("Parse availability", type="primary", width="stretch"):
        text = (st.session_state.get("single_text") or "").strip()
        if not text:
            st.warning("Please enter some text")
        else:
            with st.spinner("Submitting request …"):
                try:
                    task_id = _post_request(project, text, callback_uri)
                except requests.HTTPError as e:
                    st.error(f"HTTP error: {e.response.status_code} — {e.response.text}")
                    st.stop()
                except Exception as e:
                    st.error(f"Request failed: {e}")
                    st.stop()
            with st.spinner("Waiting for WebSocket result …"):
                result = _fetch_ws(project, task_id, timeout=int(ws_timeout), debug=debug_ws)
            df = _flatten_result(result, project, text, task_id)
            expected_cols = list(st.session_state["results_df"].columns)
            add_df = _concat_nonempty([df], columns=expected_cols)
            st.session_state["results_df"] = _concat_nonempty(
                [st.session_state["results_df"], add_df],
                columns=expected_cols
            )
            st.session_state["results_df"] = _dedupe(st.session_state["results_df"], dedupe_mode)
            st.success("Done" if "error" not in result else f"WS error: {result['error']}")

    if b2.button("Clear results", width="stretch"):
        st.session_state["results_df"] = st.session_state["results_df"].iloc[0:0]
        st.success("Cleared")

# ----------------------- Tab 2: Bulk -----------------------

with tab2:
    st.markdown("### Strategic Batch File Manager")
    left, right = st.columns([2,1])
    with right:
        st.download_button(
            "Download CSV template",
            data=_sample_csv(),
            file_name="flexzo_availability_template.csv",
            mime="text/csv",
            width="stretch",
        )

    uploaded = st.file_uploader(
        "Upload CSV/XLSX with a 'content' column",
        type=["csv","xlsx","xls"],
        accept_multiple_files=False,
    )
    clear_before_bulk = st.checkbox("Clear results before bulk run", value=False)
    if uploaded is not None:
        try:
            udf = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()
        if "content" not in udf.columns:
            st.error("Missing required 'content' column")
            st.stop()

        st.markdown(f"Filesize OK · Rows to process: **{len(udf)}**")
        if st.button("Run bulk parsing", type="primary", width="stretch"):
            if clear_before_bulk:
                st.session_state["results_df"] = st.session_state["results_df"].iloc[0:0]
            prog = st.progress(0.0)
            bulk_rows = []
            for i, row in udf.iterrows():
                text = str(row["content"]).strip()
                if not text:
                    continue
                try:
                    task_id = _post_request(project, text, callback_uri)
                    result = _fetch_ws(project, task_id, timeout=int(ws_timeout), debug=debug_ws)
                    df_row = _flatten_result(result, project, text, task_id)
                    bulk_rows.append(df_row)
                except Exception as e:
                    err_df = pd.DataFrame([{
                        "project": project, "task_id": None, "content": text,
                        "status": "error", "date": None, "start": None, "end": None,
                        "availabilityTime": None, "isFullDay": None, "durationHours": None,
                        "spanIndex": None, "reasoning": str(e), "raw_json": None
                    }])
                    bulk_rows.append(err_df)
                prog.progress((i + 1) / max(1, len(udf)))
            expected_cols = list(st.session_state["results_df"].columns)
            add_df = _concat_nonempty(bulk_rows, columns=expected_cols)
            st.session_state["results_df"] = _concat_nonempty(
                [st.session_state["results_df"], add_df],
                columns=expected_cols
            )
            st.session_state["results_df"] = _dedupe(st.session_state["results_df"], dedupe_mode)
            st.success("Bulk parsing finished")

# ----------------------- Tab 3: Test Pack -----------------------

with tab3:
    st.markdown("### Built-in Test Pack")
    tests = [
        "i will be available tomorrow",
        "i will be unavailable tomorrow",
        "i will be available next week",
        "free tomorrow morning",
        "tomorrow after 3pm",
    ]
    st.write("Runs a small standardized set of prompts and appends results.")
    clear_before_pack = st.checkbox("Clear results before test pack", value=False)
    if st.button("Run Test Pack", type="primary", width="stretch"):
        if clear_before_pack:
            st.session_state["results_df"] = st.session_state["results_df"].iloc[0:0]
        prog = st.progress(0.0)
        pack_rows = []
        for i, text in enumerate(tests):
            try:
                task_id = _post_request(project, text, callback_uri)
                result = _fetch_ws(project, task_id, timeout=int(ws_timeout), debug=debug_ws)
                df_row = _flatten_result(result, project, text, task_id)
                pack_rows.append(df_row)
            except Exception as e:
                err_df = pd.DataFrame([{
                    "project": project, "task_id": None, "content": text,
                    "status": "error", "date": None, "start": None, "end": None,
                    "availabilityTime": None, "isFullDay": None, "durationHours": None,
                    "spanIndex": None, "reasoning": str(e), "raw_json": None
                }])
                pack_rows.append(err_df)
            prog.progress((i + 1) / len(tests))
        expected_cols = list(st.session_state["results_df"].columns)
        add_df = _concat_nonempty(pack_rows, columns=expected_cols)
        st.session_state["results_df"] = _concat_nonempty(
            [st.session_state["results_df"], add_df],
            columns=expected_cols
        )
        st.session_state["results_df"] = _dedupe(st.session_state["results_df"], dedupe_mode)
        st.success("Test Pack finished")

# ----------------------- Tab 4: Results -----------------------

with tab4:
    st.markdown("### Results")
    res_df = st.session_state["results_df"].copy()

    # Quick filters
    c1, c2, c3 = st.columns([1,1,2])
    proj_filter = c1.selectbox("Project", options=["all", "health", "teach"], index=0)
    status_filter = c2.selectbox("Status", options=["all", "available", "not_available", "no_parse", "error"], index=0)
    text_search = c3.text_input("Search content/reasoning", placeholder="type to filter…")

    if proj_filter != "all":
        res_df = res_df[res_df["project"] == proj_filter]
    if status_filter != "all":
        res_df = res_df[res_df["status"] == status_filter]
    if text_search.strip():
        q = text_search.strip().lower()
        res_df = res_df[
            res_df["content"].str.lower().str.contains(q) |
            res_df["reasoning"].fillna("").str.lower().str.contains(q)
        ]

    # Friendly status icon
    def _status_icon(s):
        return {
            "available": "✅ available",
            "not_available": "⛔ not_available",
            "no_parse": "⚠️ no_parse",
            "error": "❌ error"
        }.get(s, s)

    if not res_df.empty:
        res_df = res_df.assign(statusIcon=res_df["status"].map(_status_icon))

    st.dataframe(res_df, hide_index=True, width="stretch")

    st.markdown("#### Export")
    csv_bytes = res_df.to_csv(index=False).encode("utf-8") if not res_df.empty else b""
    xlsx_bytes = _to_excel(res_df) if not res_df.empty else b""
    d1, d2, _ = st.columns([1,1,4])
    with d1:
        st.download_button("Download CSV", data=csv_bytes, file_name="flexzo_availability_results.csv",
                           mime="text/csv", width="stretch", disabled=res_df.empty)
    with d2:
        st.download_button("Download Excel", data=xlsx_bytes, file_name="flexzo_availability_results.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           width="stretch", disabled=res_df.empty)

st.caption("Flexzo QA · This tool focuses on availability correctness, not performance.")

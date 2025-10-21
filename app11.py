# app.py — Flexzo Availability Lab (Health / Teach)
# -------------------------------------------------
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
    page_title="Flexzo Availability Lab",
    page_icon="💜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Elegant Flexzo purple UI (soft gradients, glass cards)
st.markdown(
    """
    <style>
      :root {
        --bg-1: #0f0a1f;
        --bg-2: #1a0f3a;
        --card: rgba(255,255,255,0.06);
        --stroke: rgba(255,255,255,0.12);
        --text: #f5edff;
        --muted: #cbb8ff;
        --chip: rgba(110, 66, 193, 0.45);
        --ok: #22c55e; --warn: #f59e0b; --bad: #ef4444;
      }
      html, body, .block-container {
        background:
          radial-gradient(1200px 600px at 10% -10%, rgba(139,92,246,0.25), transparent 60%),
          radial-gradient(900px 500px at 100% 0%, rgba(34,211,238,0.25), transparent 50%),
          linear-gradient(180deg, var(--bg-2), var(--bg-1) 40%, #0b0717 100%);
      }
      .block-container { padding-top: 1rem; }
      h1,h2,h3,h4,h5,h6,p, label, span, div { color: var(--text); }
      .fx-hero {
        background: linear-gradient(135deg, rgba(139,92,246,0.22), rgba(34,211,238,0.12));
        border: 1px solid var(--stroke);
        border-radius: 18px;
        padding: 20px 22px;
        margin-bottom: 14px;
        box-shadow: 0 24px 50px rgba(0,0,0,0.30), 0 0 0 1px rgba(255,255,255,0.03) inset;
      }
      .fx-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
        border: 1px solid var(--stroke);
        border-radius: 14px;
        padding: 14px;
        box-shadow: 0 14px 28px rgba(0,0,0,0.25), 0 0 0 1px rgba(255,255,255,0.02) inset;
      }
      .fx-chip {
        display:inline-block; padding: 4px 10px; border-radius: 999px;
        border: 1px solid var(--stroke); background: var(--chip); color: #f0eaff; font-size: .85rem;
      }
      .fx-divider { height:1px; background: var(--stroke); margin: 10px 0 12px; }
      .stDownloadButton > button, .stButton > button {
        border-radius: 999px !important;
        border: 1px solid var(--stroke) !important;
        background: linear-gradient(180deg, rgba(255,255,255,0.12), rgba(255,255,255,0.05)) !important;
        color: var(--text) !important;
      }
      .stTextArea textarea { min-height: 90px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ======================= Config =======================

API_HOST = "35.197.253.66"          # QA host
FLOWER_URL = "http://34.142.119.139/"

# ======================= Helpers =======================

def _fix_timestr(s: str | None) -> str | None:
    """Normalize spaced time strings like '08: 00: 00' → '08:00:00'."""
    if not s:
        return s
    return re.sub(r"(\d{2}):\s*(\d{2}):\s*(\d{2})", r"\1:\2:\3", s.strip())

def _parse_ws_payload(text: str | bytes) -> dict:
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

def _post_request(project: str, content: str, callback_uri: str | None, timeout: int = 30) -> str:
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

def _fetch_ws(project: str, task_id: str, timeout: int = 60, debug: bool = False) -> dict:
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

def _duration_hours(start: str | None, end: str | None) -> float | None:
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

def _flatten_result(result: dict, project: str, content: str, task_id: str) -> pd.DataFrame:
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

def _to_excel(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="results")
    buffer.seek(0)
    return buffer.read()

def _sample_csv() -> bytes:
    df = pd.DataFrame({"content": [
        "i will be available tomorrow",
        "i will be unavailable tomorrow",
        "i will be available next week",
        "free tomorrow morning",
        "tomorrow after 3pm",
    ]})
    return df.to_csv(index=False).encode("utf-8")

def _dedupe(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if df is None or df.empty or mode == "off":
        return df
    if mode == "by task_id":
        subset = ["task_id", "status", "date", "start", "end"]
    else:  # by values
        subset = ["project", "content", "status", "date", "start", "end", "availabilityTime"]
    return df.drop_duplicates(subset=subset, keep="last").reset_index(drop=True)

# ======================= Safe Session Defaults =======================

if "results_df" not in st.session_state:
    st.session_state["results_df"] = pd.DataFrame(columns=[
        "project","task_id","content","status","date","start","end",
        "availabilityTime","isFullDay","durationHours","spanIndex",
        "reasoning","raw_json"
    ])

if "single_text" not in st.session_state:
    st.session_state["single_text"] = ""

# ======================= Header =======================

st.markdown(
    """
    <div class="fx-hero">
      <h1>Flexzo Availability Lab</h1>
      <p>Type natural-language availability, parse via QA APIs, and export clean results. Works for <b>health</b> and <b>teach</b>.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

a, b, c = st.columns(3)
with a:
    st.markdown(f"<div class='fx-card'>🌐 <b>API Host:</b> <span class='fx-chip'>{API_HOST}</span></div>", unsafe_allow_html=True)
with b:
    st.markdown(f"<div class='fx-card'>📈 <b>Flower:</b> <a href='{FLOWER_URL}' target='_blank' class='fx-chip'>{FLOWER_URL}</a></div>", unsafe_allow_html=True)
with c:
    st.markdown("<div class='fx-card'>⏱️ <b>WS Timeout (default):</b> <span class='fx-chip'>60s</span></div>", unsafe_allow_html=True)

# ======================= Sidebar (stable keys) =======================

with st.sidebar:
    st.header("Settings")
    project = st.selectbox("Project", options=["health", "teach"], index=0, key="sb_project")
    callback_uri = st.text_input("Optional callbackUri", key="sb_callback", placeholder="https://... (optional)")
    ws_timeout = st.number_input("WebSocket timeout (s)", min_value=5, max_value=300, value=60, key="sb_ws_timeout")
    debug_ws = st.checkbox("Debug WebSocket frames", value=False, key="sb_debug")
    dedupe_mode = st.selectbox(
        "De-duplicate rows",
        options=["by values", "off", "by task_id"],
        index=0,
        help="by values = unique by project+content+status+date+time; by task_id = unique per server task; off = keep all",
        key="sb_dedupe",
    )
    st.caption("QA host; server timezone assumed Africa/Harare.")

# ======================= Tabs =======================

tab1, tab2, tab3, tab4 = st.tabs(["Single", "Bulk Upload", "Quick Tests", "Results"])

# ----------------------- Tab 1: Single -----------------------

with tab1:
    st.markdown("#### Input")
    content = st.text_area(
        "Availability text",
        placeholder="e.g. i will be available next week",
        height=110,
        label_visibility="collapsed",
        key="single_text",
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    if c1.button("Tomorrow", key="chip1", width="stretch"):
        st.session_state["single_text"] = "i will be available tomorrow"; st.rerun()
    if c2.button("Tomorrow morning", key="chip2", width="stretch"):
        st.session_state["single_text"] = "free tomorrow morning"; st.rerun()
    if c3.button("After 3pm", key="chip3", width="stretch"):
        st.session_state["single_text"] = "tomorrow after 3pm"; st.rerun()
    if c4.button("Full time", key="chip4", width="stretch"):
        st.session_state["single_text"] = "I am full time"; st.rerun()
    if c5.button("Next week", key="chip5", width="stretch"):
        st.session_state["single_text"] = "i will be available next week"; st.rerun()

    st.markdown('<div class="fx-divider"></div>', unsafe_allow_html=True)

    b1, b2, _ = st.columns([1, 1, 6])
    if b1.button("Parse", type="primary", width="stretch", key="btn_parse_single"):
        text = (st.session_state.get("single_text") or "").strip()
        if not text:
            st.warning("Please enter some text")
        else:
            with st.spinner("Submitting request …"):
                try:
                    task_id = _post_request(st.session_state["sb_project"], text, st.session_state["sb_callback"])
                except requests.HTTPError as e:
                    st.error(f"HTTP error: {e.response.status_code} — {e.response.text}")
                    st.stop()
                except Exception as e:
                    st.error(f"Request failed: {e}")
                    st.stop()
            with st.spinner("Waiting for WebSocket result …"):
                result = _fetch_ws(st.session_state["sb_project"], task_id, timeout=int(st.session_state["sb_ws_timeout"]), debug=st.session_state["sb_debug"])
            df = _flatten_result(result, st.session_state["sb_project"], text, task_id)
            if st.session_state["results_df"].empty:
                st.session_state["results_df"] = df.copy()
            else:
                st.session_state["results_df"] = pd.concat([st.session_state["results_df"], df], ignore_index=True)
            st.session_state["results_df"] = _dedupe(st.session_state["results_df"], st.session_state["sb_dedupe"])
            st.success("Done" if "error" not in result else f"WS error: {result['error']}")

    if b2.button("Clear table", width="stretch", key="btn_clear_single"):
        st.session_state["results_df"] = st.session_state["results_df"].iloc[0:0]
        st.success("Cleared")

# ----------------------- Tab 2: Bulk -----------------------

with tab2:
    st.markdown("#### Upload a CSV or Excel file with a `content` column")
    left, right = st.columns([2,1])
    with right:
        st.download_button(
            "Download CSV template",
            data=_sample_csv(),
            file_name="flexzo_availability_template.csv",
            mime="text/csv",
            width="stretch",
            key="dl_template",
        )

    uploaded = st.file_uploader(
        "Choose file",
        type=["csv","xlsx","xls"],
        accept_multiple_files=False,
        key="fu_bulk",
    )
    clear_before_bulk = st.checkbox("Clear table before bulk run", value=False, key="cb_clear_bulk")

    if uploaded is not None:
        try:
            udf = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()
        if "content" not in udf.columns:
            st.error("Missing required 'content' column")
            st.stop()

        st.markdown(f"Rows to process: **{len(udf)}**")
        if st.button("Run", type="primary", width="stretch", key="btn_run_bulk"):
            if clear_before_bulk:
                st.session_state["results_df"] = st.session_state["results_df"].iloc[0:0]
            prog = st.progress(0.0)
            bulk_rows = []
            for i, row in udf.iterrows():
                text = str(row["content"]).strip()
                if not text:
                    continue
                try:
                    task_id = _post_request(st.session_state["sb_project"], text, st.session_state["sb_callback"])
                    result = _fetch_ws(st.session_state["sb_project"], task_id, timeout=int(st.session_state["sb_ws_timeout"]), debug=st.session_state["sb_debug"])
                    df_row = _flatten_result(result, st.session_state["sb_project"], text, task_id)
                    bulk_rows.append(df_row)
                except Exception as e:
                    err_df = pd.DataFrame([{
                        "project": st.session_state["sb_project"], "task_id": None, "content": text,
                        "status": "error", "date": None, "start": None, "end": None,
                        "availabilityTime": None, "isFullDay": None, "durationHours": None,
                        "spanIndex": None, "reasoning": str(e), "raw_json": None
                    }])
                    bulk_rows.append(err_df)
                prog.progress((i + 1) / max(1, len(udf)))
            if bulk_rows:
                add_df = pd.concat(bulk_rows, ignore_index=True)
                if st.session_state["results_df"].empty:
                    st.session_state["results_df"] = add_df.copy()
                else:
                    st.session_state["results_df"] = pd.concat([st.session_state["results_df"], add_df], ignore_index=True)
                st.session_state["results_df"] = _dedupe(st.session_state["results_df"], st.session_state["sb_dedupe"])
            st.success("Bulk complete")

# ----------------------- Tab 3: Quick Tests -----------------------

with tab3:
    st.markdown("#### Run a small, built-in test set")
    tests = [
        "i will be available tomorrow",
        "i will be unavailable tomorrow",
        "i will be available next week",
        "free tomorrow morning",
        "tomorrow after 3pm",
    ]
    clear_before_pack = st.checkbox("Clear table before quick tests", value=False, key="cb_clear_pack")
    if st.button("Run tests", type="primary", width="stretch", key="btn_run_tests"):
        if clear_before_pack:
            st.session_state["results_df"] = st.session_state["results_df"].iloc[0:0]
        prog = st.progress(0.0)
        pack_rows = []
        for i, text in enumerate(tests):
            try:
                task_id = _post_request(st.session_state["sb_project"], text, st.session_state["sb_callback"])
                result = _fetch_ws(st.session_state["sb_project"], task_id, timeout=int(st.session_state["sb_ws_timeout"]), debug=st.session_state["sb_debug"])
                df_row = _flatten_result(result, st.session_state["sb_project"], text, task_id)
                pack_rows.append(df_row)
            except Exception as e:
                err_df = pd.DataFrame([{
                    "project": st.session_state["sb_project"], "task_id": None, "content": text,
                    "status": "error", "date": None, "start": None, "end": None,
                    "availabilityTime": None, "isFullDay": None, "durationHours": None,
                    "spanIndex": None, "reasoning": str(e), "raw_json": None
                }])
                pack_rows.append(err_df)
            prog.progress((i + 1) / len(tests))
        if pack_rows:
            add_df = pd.concat(pack_rows, ignore_index=True)
            if st.session_state["results_df"].empty:
                st.session_state["results_df"] = add_df.copy()
            else:
                st.session_state["results_df"] = pd.concat([st.session_state["results_df"], add_df], ignore_index=True)
            st.session_state["results_df"] = _dedupe(st.session_state["results_df"], st.session_state["sb_dedupe"])
        st.success("Done")

# ----------------------- Tab 4: Results -----------------------

with tab4:
    st.markdown("#### Table")
    res_df = st.session_state["results_df"].copy()

    # Quick filters (stable keys)
    c1, c2, c3 = st.columns([1,1,2])
    proj_filter = c1.selectbox("Project", options=["all", "health", "teach"], index=0, key="flt_project")
    status_filter = c2.selectbox("Status", options=["all", "available", "not_available", "no_parse", "error"], index=0, key="flt_status")
    text_search = c3.text_input("Search content or reasoning", placeholder="type to filter…", key="flt_text")

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

    # Status icon
    def _status_icon(s: str) -> str:
        return {"available": "✅ available", "not_available": "⛔ not_available", "no_parse": "⚠️ no_parse", "error": "❌ error"}.get(s, s)

    if not res_df.empty:
        res_df = res_df.assign(statusIcon=res_df["status"].map(_status_icon))

    # IMPORTANT: use data_editor (disabled) with a stable key to avoid the 'all_edit_mode' KeyError
    st.data_editor(
        res_df,
        hide_index=True,
        disabled=True,
        key="results_table",  # stable key fixes the widget-state crash
    )

    st.markdown("#### Export")
    csv_bytes = res_df.to_csv(index=False).encode("utf-8") if not res_df.empty else b""
    xlsx_bytes = _to_excel(res_df) if not res_df.empty else b""
    d1, d2, _ = st.columns([1,1,4])
    with d1:
        st.download_button("Download CSV", data=csv_bytes, file_name="flexzo_availability_results.csv", mime="text/csv", width="stretch", disabled=res_df.empty, key="dl_csv")
    with d2:
        st.download_button("Download Excel", data=xlsx_bytes, file_name="flexzo_availability_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch", disabled=res_df.empty, key="dl_xlsx")

st.caption("Flexzo QA · Availability extraction focus (not performance).")

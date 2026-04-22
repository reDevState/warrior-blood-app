"""
chw_dashboard.py — Warrior Blood Community Health Worker Triage Dashboard.

Streamlit app connecting to the FastAPI backend. CHWs use this to:
  - View all patients sorted by VOC risk tier (HIGH first)
  - Inspect 30-day pain and risk timelines per patient
  - Send manual SMS alerts
  - See hydration and SHAP factor breakdowns

Run:
    streamlit run dashboard/chw_dashboard.py
"""

from __future__ import annotations

import os
from datetime import datetime

import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Warrior Blood — CHW Dashboard",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .risk-HIGH   { color: #A32D2D; font-weight: 600; }
    .risk-MODERATE { color: #854F0B; font-weight: 600; }
    .risk-LOW    { color: #3B6D11; font-weight: 600; }
    .metric-card { background: #f8f8f8; border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _headers() -> dict:
    return {"Authorization": f"Bearer {st.session_state.get('token', '')}"}


def _api(method: str, path: str, **kwargs):
    try:
        resp = getattr(requests, method)(f"{API_URL}{path}", **kwargs, timeout=10)
        return resp
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach the Warrior Blood API. Is the backend running on port 8000?")
        return None


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------

def login_page():
    st.title("Warrior Blood")
    st.subheader("CHW Login")

    with st.form("login"):
        username = st.text_input("Username", value="test_chw")
        password = st.text_input("Password", type="password", value="chwpassword")
        submitted = st.form_submit_button("Sign in")

    if submitted:
        resp = _api(
            "post",
            "/auth/token",
            data={"username": username, "password": password},
        )
        if resp and resp.status_code == 200:
            st.session_state["token"] = resp.json()["access_token"]
            st.session_state["username"] = username
            st.rerun()
        else:
            st.error("Login failed — check username and password.")

    st.caption("MVP demo credentials: test_chw / chwpassword")


# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------

def dashboard():
    # Sidebar
    with st.sidebar:
        st.markdown("### Warrior Blood")
        st.caption(f"Signed in as: **{st.session_state.get('username', '')}**")
        st.divider()
        page = st.radio("View", ["Patient triage", "Register patient", "About"])
        if st.button("Sign out"):
            st.session_state.clear()
            st.rerun()

    if page == "Patient triage":
        triage_page()
    elif page == "Register patient":
        register_page()
    else:
        about_page()


# ---------------------------------------------------------------------------
# Triage page
# ---------------------------------------------------------------------------

def triage_page():
    st.header("Patient triage")

    resp = _api("get", "/patients", headers=_headers())
    if not resp or resp.status_code != 200:
        st.warning("Could not load patient list.")
        return

    patients: list[dict] = resp.json()

    if not patients:
        st.info("No patients registered yet. Use 'Register patient' to add one.")
        return

    # Summary metrics
    high = sum(1 for p in patients if p.get("latest_risk_tier") == "HIGH")
    mod  = sum(1 for p in patients if p.get("latest_risk_tier") == "MODERATE")
    low  = sum(1 for p in patients if p.get("latest_risk_tier") == "LOW")
    none_ = sum(1 for p in patients if not p.get("latest_risk_tier"))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total patients", len(patients))
    c2.metric("🔴 HIGH risk", high)
    c3.metric("🟡 MODERATE", mod)
    c4.metric("🟢 LOW / none", low + none_)

    st.divider()

    # Patient cards
    for p in patients:
        tier = p.get("latest_risk_tier") or "—"
        tier_colour = {"HIGH": "🔴", "MODERATE": "🟡", "LOW": "🟢"}.get(tier, "⚪")
        enrolled = p.get("enrolled_at", "")[:10]

        with st.expander(f"{tier_colour} Patient {p['id'][:8]}...  |  {tier}  |  enrolled {enrolled}"):
            col_left, col_right = st.columns([2, 1])

            with col_left:
                st.markdown(f"**Diagnosis:** {p.get('diagnosis_type') or 'Not specified'}")
                history_resp = _api(
                    "get",
                    f"/patients/{p['id']}/history",
                    headers=_headers(),
                    params={"days": 30},
                )

                if history_resp and history_resp.status_code == 200:
                    history = history_resp.json()
                    if history:
                        timeline_chart(history)
                    else:
                        st.caption("No diary entries in the last 30 days.")
                else:
                    st.caption("Could not load history.")

            with col_right:
                st.markdown("**Send alert**")
                msg = st.text_area(
                    "Message",
                    value=f"Please check in — your risk level is {tier}.",
                    key=f"msg_{p['id']}",
                    height=80,
                )
                if st.button("Send SMS", key=f"sms_{p['id']}"):
                    alert_resp = _api(
                        "post",
                        "/alerts",
                        headers=_headers(),
                        json={"patient_id": p["id"], "message": msg, "channel": "sms"},
                    )
                    if alert_resp and alert_resp.status_code == 200:
                        st.success("Alert queued.")
                    else:
                        st.error("Alert failed.")


def timeline_chart(history: list[dict]):
    """Render 30-day pain score and risk score timeline with Plotly."""
    dates      = [h["entry_date"] for h in history]
    pain       = [h["pain_score"] for h in history]
    risk_score = [h.get("risk_score") or 0 for h in history]
    hydration  = [h.get("hydration_status") or "" for h in history]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates, y=pain, name="Pain score",
        line=dict(color="#A32D2D", width=2),
        yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=risk_score, name="VOC risk",
        line=dict(color="#378ADD", width=2, dash="dot"),
        yaxis="y2",
    ))

    # Shade HIGH-risk days
    for i, h in enumerate(history):
        if h.get("risk_tier") == "HIGH":
            fig.add_vrect(
                x0=dates[i], x1=dates[i],
                fillcolor="rgba(163,45,45,0.15)",
                line_width=0,
            )

    fig.update_layout(
        height=200,
        margin=dict(l=0, r=0, t=20, b=0),
        showlegend=True,
        legend=dict(orientation="h", y=1.15),
        yaxis=dict(title="Pain (0-10)", range=[0, 10], side="left"),
        yaxis2=dict(title="Risk (0-1)", range=[0, 1], side="right", overlaying="y"),
        xaxis=dict(title=""),
        plot_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Register patient
# ---------------------------------------------------------------------------

def register_page():
    st.header("Register new patient")

    with st.form("register"):
        name = st.text_input("Full name")
        phone = st.text_input("Phone number (optional)")
        dob = st.date_input("Date of birth", value=None)
        diagnosis = st.selectbox(
            "Diagnosis type",
            ["HbSS", "HbSC", "HbS/beta-thalassaemia", "Other"],
        )
        submitted = st.form_submit_button("Register patient")

    if submitted:
        if not name:
            st.error("Name is required.")
            return
        payload = {
            "name": name,
            "phone": phone or None,
            "dob": str(dob) if dob else None,
            "diagnosis_type": diagnosis,
        }
        resp = _api("post", "/patients", headers=_headers(), json=payload)
        if resp and resp.status_code == 201:
            pid = resp.json()["id"]
            st.success(f"Patient registered. ID: {pid}")
        else:
            detail = resp.json().get("detail", "Unknown error") if resp else "No response"
            st.error(f"Registration failed: {detail}")


# ---------------------------------------------------------------------------
# About
# ---------------------------------------------------------------------------

def about_page():
    st.header("About Warrior Blood")
    st.markdown("""
**Warrior Blood** is a Sickle Cell Disease (SCD) patient monitoring system
developed as an MSc Software Engineering project at the University of Greater Manchester.

**Renee Tucker · Supervisor: Aamir Abbas · 2025–26**

---

### MVP components
- **FastAPI backend** — patient check-in, VOC risk scoring, CHW alerts
- **Rule-based ML predictor** — heuristic model (replace with LightGBM ONNX after training)
- **Hydration module** — WHO oral rehydration guideline thresholds (Yallop et al. 2007)
- **Streamlit CHW dashboard** — triage interface with 30-day timelines
- **SQLite** — zero-infrastructure database for MVP (swap to PostgreSQL for production)

### Key references
- Machado et al. (2024) — LightGBM on SCD prediction
- Yallop et al. (2007) — dehydration and SCD hospitalisation
- Wahl et al. (2018) — interpretability in LMIC health AI
- Lundberg & Lee (2017) — SHAP values

### MVP limitations
- SMS alerts are stubbed (logged only — integrate Africa's Talking for real SMS)
- ML predictor is rule-based (train LightGBM on real data for production)
- Authentication is in-memory (add persistent user table for production)
    """)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if "token" not in st.session_state:
    login_page()
else:
    dashboard()

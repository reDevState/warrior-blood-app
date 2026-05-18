"""
chw_dashboard.py — Warrior Blood unified dashboard.

Decodes the JWT role after login and routes to the correct view:
  - CHW / admin → CHW triage dashboard
  - patient      → Patient self-service dashboard

Patient data logged through the patient dashboard is immediately visible
in the CHW triage view — both share the same FastAPI backend.

Run:
    streamlit run dashboard/chw_dashboard.py
"""

from __future__ import annotations

import base64
import json
import os
from datetime import datetime

import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Warrior Blood",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .risk-HIGH     { color: #A32D2D; font-weight: 700; }
    .risk-MODERATE { color: #854F0B; font-weight: 700; }
    .risk-LOW      { color: #3B6D11; font-weight: 700; }
    .alert-box     { background: #fff4e5; border-left: 4px solid #e07b00;
                     padding: 10px 14px; border-radius: 4px; margin: 6px 0; }
    .breakthrough  { background: #fff0f0; border-left: 4px solid #A32D2D;
                     padding: 10px 14px; border-radius: 4px; margin: 6px 0; }
    .info-card     { background: #f0f4ff; border-left: 4px solid #378ADD;
                     padding: 10px 14px; border-radius: 4px; margin: 6px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# JWT / session helpers
# ---------------------------------------------------------------------------

def _decode_role(token: str) -> str:
    """Extract role from JWT payload without signature verification."""
    try:
        payload_b64 = token.split(".")[1]
        payload_b64 += "=" * (4 - len(payload_b64) % 4)
        payload = json.loads(base64.b64decode(payload_b64))
        return payload.get("role", "patient")
    except Exception:
        return "patient"


def _headers() -> dict:
    return {"Authorization": f"Bearer {st.session_state.get('token', '')}"}


def _api(method: str, path: str, **kwargs):
    try:
        resp = getattr(requests, method)(f"{API_URL}{path}", **kwargs, timeout=10)
        return resp
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach the Warrior Blood API. Is the backend running on port 8000?")
        return None


def _patient_options() -> list[dict]:
    resp = _api("get", "/patients", headers=_headers())
    return resp.json() if resp and resp.status_code == 200 else []


def _patient_label(p: dict) -> str:
    name = p.get("name") or f"Patient {p['id'][:8]}…"
    tier = p.get("latest_risk_tier") or "—"
    icon = {"HIGH": "🔴", "MODERATE": "🟡", "LOW": "🟢"}.get(tier, "⚪")
    return f"{icon} {name}"


# ---------------------------------------------------------------------------
# Login page
# ---------------------------------------------------------------------------

def login_page() -> None:
    col_left, col_mid, col_right = st.columns([1, 2, 1])
    with col_mid:
        st.markdown("## 🩸 Warrior Blood")

        tab_signin, tab_register, tab_forgot = st.tabs(
            ["Sign in", "Register patient", "Forgot password"]
        )

        # ── Sign in ──────────────────────────────────────────────────────────
        with tab_signin:
            with st.form("login"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Sign in", use_container_width=True)

            if submitted:
                resp = _api("post", "/auth/token",
                            data={"username": username, "password": password})
                if resp and resp.status_code == 200:
                    token = resp.json()["access_token"]
                    st.session_state["token"]    = token
                    st.session_state["username"] = username
                    st.session_state["role"]     = _decode_role(token)
                    st.rerun()
                else:
                    st.error("Login failed — check username and password.")

            st.divider()
            st.caption("**CHW demo:** test_chw / chwpassword")
            st.caption("**Patient demo:** test_patient / testpassword")

        # ── Register patient (CHW credentials required) ───────────────────
        with tab_register:
            st.caption("CHW credentials are required to register a new patient.")
            with st.form("login_register"):
                chw_user  = st.text_input("CHW username", key="reg_chw_user")
                chw_pass  = st.text_input("CHW password", type="password", key="reg_chw_pass")
                st.divider()
                reg_name  = st.text_input("Patient full name", key="reg_name")
                reg_email = st.text_input("Patient email", placeholder="patient@example.com", key="reg_email")
                reg_phone = st.text_input("Phone number (optional)", key="reg_phone")
                reg_dob   = st.date_input("Date of birth", value=None, key="reg_dob")
                reg_diag  = st.selectbox(
                    "Diagnosis type",
                    ["HbSS", "HbSC", "HbS/beta-thalassaemia", "Other"],
                    key="reg_diag",
                )
                reg_submitted = st.form_submit_button("Register patient", use_container_width=True)

            if reg_submitted:
                if not reg_name:
                    st.error("Patient name is required.")
                else:
                    auth_resp = _api("post", "/auth/token",
                                     data={"username": chw_user, "password": chw_pass})
                    if not auth_resp or auth_resp.status_code != 200:
                        st.error("CHW login failed — check your username and password.")
                    else:
                        chw_token = auth_resp.json()["access_token"]
                        if _decode_role(chw_token) not in ("chw", "admin"):
                            st.error("Only CHW or admin accounts can register patients.")
                        else:
                            pat_resp = _api("post", "/patients",
                                            headers={"Authorization": f"Bearer {chw_token}"},
                                            json={
                                                "name": reg_name,
                                                "email": reg_email.strip() or None,
                                                "phone": reg_phone or None,
                                                "dob": str(reg_dob) if reg_dob else None,
                                                "diagnosis_type": reg_diag,
                                            })
                            if pat_resp and pat_resp.status_code == 201:
                                data = pat_resp.json()
                                pid  = data["id"]
                                st.success(f"Patient **{reg_name}** registered successfully.")
                                if reg_email.strip():
                                    st.info(
                                        f"The patient can sign in and link their record using email: "
                                        f"**{reg_email.strip()}**"
                                    )
                                else:
                                    st.info(f"Patient ID: `{pid}` — share this so the patient can link their account.")
                            else:
                                detail = pat_resp.json().get("detail", "Unknown error") if pat_resp else "No response"
                                st.error(f"Registration failed: {detail}")

        # ── Forgot password ───────────────────────────────────────────────
        with tab_forgot:
            st.caption("Enter your username to receive a temporary password.")
            with st.form("forgot_password"):
                fp_username  = st.text_input("Username", key="fp_user")
                fp_submitted = st.form_submit_button("Reset password", use_container_width=True)

            if fp_submitted:
                if not fp_username:
                    st.error("Please enter your username.")
                else:
                    fp_resp = _api("post", "/auth/forgot-password",
                                   json={"username": fp_username})
                    if fp_resp and fp_resp.status_code == 200:
                        data = fp_resp.json()
                        st.success("Password reset.")
                        st.markdown(
                            f"**Temporary password:** `{data['temp_password']}`\n\n"
                            "Sign in with this password, then contact your administrator to set a permanent one."
                        )
                    elif fp_resp and fp_resp.status_code == 404:
                        st.error("Username not found.")
                    else:
                        st.error("Password reset failed. Please try again.")


# ---------------------------------------------------------------------------
# Entry router
# ---------------------------------------------------------------------------

def main() -> None:
    if "token" not in st.session_state:
        login_page()
        return

    role = st.session_state.get("role", "patient")
    if role in ("chw", "admin"):
        chw_dashboard()
    else:
        patient_dashboard()


# ============================================================================
# CHW DASHBOARD
# ============================================================================

def chw_dashboard() -> None:
    with st.sidebar:
        st.markdown("### 🩸 Warrior Blood")
        st.caption("Community Health Worker")
        st.caption(f"Signed in as **{st.session_state.get('username', '')}**")
        st.divider()
        page = st.radio("Navigation", [
            "Patient triage",
            "Log for patient",
            "About",
        ])
        if st.button("Sign out", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    if page == "Patient triage":
        chw_triage_page()
    elif page == "Log for patient":
        chw_log_page()
    else:
        about_page()


# ---------------------------------------------------------------------------
# CHW — Triage
# ---------------------------------------------------------------------------

def chw_triage_page() -> None:
    st.header("Patient triage")

    col_refresh, _ = st.columns([1, 5])
    if col_refresh.button("🔄 Refresh"):
        st.rerun()

    patients = _patient_options()
    if not patients:
        st.info("No patients registered yet. Use 'Register patient' to add one.")
        return

    high  = sum(1 for p in patients if p.get("latest_risk_tier") == "HIGH")
    mod   = sum(1 for p in patients if p.get("latest_risk_tier") == "MODERATE")
    low   = sum(1 for p in patients if p.get("latest_risk_tier") == "LOW")
    none_ = sum(1 for p in patients if not p.get("latest_risk_tier"))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total patients", len(patients))
    c2.metric("🔴 HIGH risk", high)
    c3.metric("🟡 MODERATE", mod)
    c4.metric("🟢 LOW / none", low + none_)

    st.divider()

    for p in patients:
        tier        = p.get("latest_risk_tier") or "—"
        tier_icon   = {"HIGH": "🔴", "MODERATE": "🟡", "LOW": "🟢"}.get(tier, "⚪")
        enrolled    = p.get("enrolled_at", "")[:10]
        name        = p.get("name") or f"Patient {p['id'][:8]}…"
        pid         = p["id"]

        with st.expander(f"{tier_icon} {name}  ·  {tier}  ·  enrolled {enrolled}"):
            tab_overview, tab_pain, tab_hydration, tab_alert = st.tabs([
                "Overview", "Pain history", "Hydration", "Send alert"
            ])

            # --- Overview ---
            with tab_overview:
                st.markdown(f"**Diagnosis:** {p.get('diagnosis_type') or 'Not specified'}")
                st.caption(f"Patient ID: `{pid}`")
                history_resp = _api("get", f"/patients/{pid}/history",
                                    headers=_headers(), params={"days": 30})
                if history_resp and history_resp.status_code == 200:
                    history = history_resp.json()
                    if history:
                        _timeline_chart(history)
                    else:
                        st.caption("No diary entries in the last 30 days.")
                else:
                    st.caption("Could not load history.")

            # --- Pain history ---
            with tab_pain:
                pain_resp = _api("get", f"/patients/{pid}/pain",
                                 headers=_headers(), params={"days": 30})
                if pain_resp and pain_resp.status_code == 200:
                    entries = pain_resp.json()
                    if entries:
                        _pain_history_chart(entries)
                        for e in reversed(entries[-5:]):
                            ts     = e.get("recorded_at", "")[:16].replace("T", " ")
                            score  = e.get("pain_score", 0)
                            locs   = ", ".join(
                                _LOCATION_LABELS.get(l, l)
                                for l in (e.get("pain_locations") or [])
                            )
                            flags  = []
                            if e.get("is_breakthrough"):
                                flags.append("⭐ Breakthrough")
                            if e.get("chest_pain_alert"):
                                flags.append("🚨 Chest pain")
                            st.markdown(
                                f"`{ts}` — Score **{score}/10**"
                                + (f" · {locs}" if locs else "")
                                + (f" · {' · '.join(flags)}" if flags else "")
                            )
                    else:
                        st.caption("No pain entries in the last 30 days.")
                else:
                    st.caption("Could not load pain history.")

            # --- Hydration ---
            with tab_hydration:
                diary_resp = _api("get", f"/patients/{pid}/history",
                                  headers=_headers(), params={"days": 7})
                if diary_resp and diary_resp.status_code == 200:
                    entries = diary_resp.json()
                    if entries:
                        recent = entries[-1]
                        status = recent.get("hydration_status") or "—"
                        icon   = "✅" if status == "WELL_HYDRATED" else "⚠️"
                        st.markdown(f"**Latest hydration status:** {icon} {status.replace('_', ' ').title()}")
                        fluids = [e.get("fluid_intake_glasses", 0) for e in entries]
                        dates  = [e.get("entry_date", "") for e in entries]
                        fig = go.Figure(go.Bar(
                            x=dates, y=fluids,
                            marker_color=["#3B6D11" if g >= 8 else "#854F0B" if g >= 5 else "#A32D2D" for g in fluids],
                            name="Glasses/day",
                        ))
                        fig.add_hline(y=8, line_dash="dash", line_color="green",
                                      annotation_text="Target (8 glasses)")
                        fig.update_layout(
                            height=180, margin=dict(l=0, r=0, t=10, b=0),
                            yaxis=dict(title="Glasses"), plot_bgcolor="white",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.caption("No diary data for the last 7 days.")

            # --- Alert ---
            with tab_alert:
                msg = st.text_area(
                    "Message",
                    value=f"Please check in — your risk level is {tier}.",
                    key=f"msg_{pid}",
                    height=80,
                )
                if st.button("Send SMS", key=f"sms_{pid}"):
                    alert_resp = _api("post", "/alerts", headers=_headers(),
                                      json={"patient_id": pid, "message": msg, "channel": "sms"})
                    if alert_resp and alert_resp.status_code == 200:
                        st.success("Alert queued.")
                    else:
                        st.error("Alert failed.")


def _timeline_chart(history: list[dict]) -> None:
    dates      = [h["entry_date"] for h in history]
    pain       = [h["pain_score"] for h in history]
    risk_score = [h.get("risk_score") or 0 for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=pain, name="Pain score",
        line=dict(color="#A32D2D", width=2), yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=risk_score, name="VOC risk",
        line=dict(color="#378ADD", width=2, dash="dot"), yaxis="y2",
    ))
    for i, h in enumerate(history):
        if h.get("risk_tier") == "HIGH":
            fig.add_vrect(
                x0=dates[i], x1=dates[i],
                fillcolor="rgba(163,45,45,0.12)", line_width=0,
            )
    fig.update_layout(
        height=200, margin=dict(l=0, r=0, t=20, b=0),
        showlegend=True, legend=dict(orientation="h", y=1.15),
        yaxis=dict(title="Pain (0-10)", range=[0, 10], side="left"),
        yaxis2=dict(title="Risk (0-1)", range=[0, 1], side="right", overlaying="y"),
        xaxis=dict(title=""), plot_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# CHW — Log for patient (manual data entry on behalf)
# ---------------------------------------------------------------------------

def chw_log_page() -> None:
    st.header("Log for patient")
    st.caption("Use this to record check-in, pain or hydration data on behalf of a patient.")

    patients = _patient_options()
    if not patients:
        st.warning("No patients registered yet.")
        return

    patient_map = {_patient_label(p): p for p in patients}
    selected    = st.selectbox("Select patient", list(patient_map.keys()))
    patient     = patient_map[selected]
    pid         = patient["id"]

    tab_ci, tab_pain, tab_hydration = st.tabs(["Check-in & Weather", "Pain diary", "Hydration diary"])

    with tab_ci:
        _checkin_form(pid, key_prefix="chw")

    with tab_pain:
        _pain_log_form(pid, key_prefix="chw")

    with tab_hydration:
        _hydration_log_form(pid, key_prefix="chw")


# ---------------------------------------------------------------------------
# CHW — Register patient
# ---------------------------------------------------------------------------

def register_page() -> None:
    st.header("Register new patient")

    with st.form("register"):
        name      = st.text_input("Full name")
        email     = st.text_input("Email address", placeholder="patient@example.com")
        phone     = st.text_input("Phone number (optional)")
        dob       = st.date_input("Date of birth", value=None)
        diagnosis = st.selectbox(
            "Diagnosis type",
            ["HbSS", "HbSC", "HbS/beta-thalassaemia", "Other"],
        )
        submitted = st.form_submit_button("Register patient", use_container_width=True)

    if submitted:
        if not name:
            st.error("Name is required.")
            return
        resp = _api("post", "/patients", headers=_headers(), json={
            "name": name,
            "email": email.strip() or None,
            "phone": phone or None,
            "dob": str(dob) if dob else None,
            "diagnosis_type": diagnosis,
        })
        if resp and resp.status_code == 201:
            data = resp.json()
            pid  = data["id"]
            st.success("Patient registered successfully.")
            if email.strip():
                st.markdown(
                    f'<div class="info-card">✅ The patient can now log in to the patient dashboard '
                    f'and enter their email address <strong>{email.strip()}</strong> to link their record.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="info-card">📋 <strong>Patient ID:</strong> <code>{pid}</code><br>'
                    f"No email registered — give the patient this ID to link their account.</div>",
                    unsafe_allow_html=True,
                )
        else:
            detail = resp.json().get("detail", "Unknown error") if resp else "No response"
            st.error(f"Registration failed: {detail}")


# ============================================================================
# PATIENT DASHBOARD
# ============================================================================

def patient_dashboard() -> None:
    with st.sidebar:
        st.markdown("### 🩸 Warrior Blood")
        st.caption("Patient Portal")
        st.caption(f"Signed in as **{st.session_state.get('username', '')}**")

        pid = st.session_state.get("patient_id")
        if pid:
            st.caption(f"Patient ID: `{pid[:8]}…`")
            if st.button("Change record", use_container_width=True):
                del st.session_state["patient_id"]
                st.rerun()

        st.divider()
        page = st.radio("Navigation", [
            "Check-in & Weather",
            "Pain diary",
            "Hydration diary",
            "My history",
            "About",
        ])
        if st.button("Sign out", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    # Patient must link their record before using the app
    if "patient_id" not in st.session_state:
        _patient_link_page()
        return

    pid = st.session_state["patient_id"]

    if page == "Check-in & Weather":
        patient_checkin_page(pid)
    elif page == "Pain diary":
        patient_pain_page(pid)
    elif page == "Hydration diary":
        patient_hydration_page(pid)
    elif page == "My history":
        patient_history_page(pid)
    else:
        about_page()


def _patient_link_page() -> None:
    """One-time setup: patient links their record via email or Patient ID."""
    st.header("Link your patient record")

    tab_email, tab_id = st.tabs(["Link by email", "Link by Patient ID"])

    with tab_email:
        st.markdown(
            "Enter the **email address** your Community Health Worker registered for you."
        )
        with st.form("link_by_email"):
            email_input = st.text_input("Email address", placeholder="patient@example.com")
            submitted_email = st.form_submit_button("Find my record", use_container_width=True)

        if submitted_email:
            email_input = email_input.strip()
            if not email_input:
                st.error("Please enter your email address.")
            else:
                resp = _api("get", "/patients/lookup",
                            headers=_headers(), params={"email": email_input})
                if resp and resp.status_code == 200:
                    pid = resp.json()["patient_id"]
                    st.session_state["patient_id"] = pid
                    st.success("Record linked successfully!")
                    st.rerun()
                elif resp and resp.status_code == 404:
                    st.error("No record found for that email. Check with your health worker.")
                else:
                    st.error("Could not search records. Please try again.")

    with tab_id:
        st.markdown(
            "If you don't have an email registered, enter the **Patient ID** "
            "your Community Health Worker gave you."
        )
        with st.form("link_by_id"):
            pid_input = st.text_input("Patient ID", placeholder="e.g. a1b2c3d4-…")
            submitted_id = st.form_submit_button("Link record", use_container_width=True)

        if submitted_id:
            pid_input = pid_input.strip()
            if not pid_input:
                st.error("Please enter your Patient ID.")
            else:
                resp = _api("get", f"/patients/{pid_input}/history",
                            headers=_headers(), params={"days": 1})
                if resp and resp.status_code == 200:
                    st.session_state["patient_id"] = pid_input
                    st.success("Record linked successfully!")
                    st.rerun()
                elif resp and resp.status_code == 404:
                    st.error("Patient ID not found. Check with your health worker.")
                else:
                    st.error("Could not verify Patient ID. Please try again.")

    st.divider()
    st.caption("Don't have a record? Ask your Community Health Worker to register you.")


# ---------------------------------------------------------------------------
# Patient — Check-in & Weather
# ---------------------------------------------------------------------------

def patient_checkin_page(pid: str) -> None:
    st.header("Daily check-in")
    st.caption(
        "Log your daily symptoms. Your location enables weather-based VOC risk alerts "
        "(Nolan et al. 2008)."
    )
    _checkin_form(pid, key_prefix="pat")


# ---------------------------------------------------------------------------
# Patient — Pain diary
# ---------------------------------------------------------------------------

def patient_pain_page(pid: str) -> None:
    st.header("Pain diary")

    tab_log, tab_history = st.tabs(["Log pain", "My pain history"])

    with tab_log:
        _pain_log_form(pid, key_prefix="pat")

    with tab_history:
        _show_pain_history(pid)


# ---------------------------------------------------------------------------
# Patient — Hydration diary
# ---------------------------------------------------------------------------

def patient_hydration_page(pid: str) -> None:
    st.header("Hydration diary")

    tab_log, tab_guide = st.tabs(["Log a drink", "Hydration guide"])

    with tab_log:
        _hydration_log_form(pid, key_prefix="pat")

    with tab_guide:
        _hydration_guide()


# ---------------------------------------------------------------------------
# Patient — My history
# ---------------------------------------------------------------------------

def patient_history_page(pid: str) -> None:
    st.header("My health history")

    days = st.slider("Days to show", 7, 90, 30)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Diary entries")
        resp = _api("get", f"/patients/{pid}/history",
                    headers=_headers(), params={"days": days})
        if resp and resp.status_code == 200:
            history = resp.json()
            if history:
                _timeline_chart(history)
                st.caption(f"{len(history)} entries in the last {days} days")
            else:
                st.info("No diary entries yet.")
        else:
            st.warning("Could not load diary history.")

    with col2:
        st.subheader("Pain entries")
        resp = _api("get", f"/patients/{pid}/pain",
                    headers=_headers(), params={"days": days})
        if resp and resp.status_code == 200:
            pain_entries = resp.json()
            if pain_entries:
                _pain_history_chart(pain_entries)
                breaks = sum(1 for e in pain_entries if e.get("is_breakthrough"))
                st.caption(f"{len(pain_entries)} entries · {breaks} breakthrough events")
            else:
                st.info("No pain entries yet.")
        else:
            st.warning("Could not load pain history.")


# ============================================================================
# SHARED FORM COMPONENTS  (used by both CHW and Patient dashboards)
# ============================================================================

# City → (latitude, longitude).  Covers regions with highest SCD burden.
_LOCATIONS: dict[str, tuple[float, float]] = {
    # --- West Africa ---
    "Abidjan, Côte d'Ivoire":       (5.3544,   -4.0017),
    "Abuja, Nigeria":               (9.0765,    7.3986),
    "Accra, Ghana":                 (5.6037,   -0.1870),
    "Bamako, Mali":                 (12.6392,  -8.0029),
    "Conakry, Guinea":              (9.6412,  -13.5784),
    "Cotonou, Benin":               (6.3703,    2.3912),
    "Dakar, Senegal":               (14.7167,  -17.4677),
    "Freetown, Sierra Leone":       (8.4657,  -13.2317),
    "Kano, Nigeria":                (12.0022,   8.5920),
    "Kumasi, Ghana":                (6.6884,   -1.6244),
    "Lagos, Nigeria":               (6.5244,    3.3792),
    "Lomé, Togo":                   (6.1375,    1.2123),
    "Monrovia, Liberia":            (6.2907,  -10.7605),
    "Niamey, Niger":                (13.5137,   2.1098),
    "Ouagadougou, Burkina Faso":    (12.3569,  -1.5353),
    "Yamoussoukro, Côte d'Ivoire":  (6.8276,   -5.2893),
    # --- Central Africa ---
    "Brazzaville, Republic of Congo": (4.2634, 15.2429),
    "Douala, Cameroon":             (4.0511,    9.7679),
    "Kinshasa, DR Congo":           (-4.4419,  15.2663),
    "Libreville, Gabon":            (0.3901,    9.4544),
    "Luanda, Angola":               (-8.8368,  13.2343),
    "Yaoundé, Cameroon":            (3.8480,   11.5021),
    # --- East Africa ---
    "Addis Ababa, Ethiopia":        (9.1450,   40.4897),
    "Dar es Salaam, Tanzania":      (-6.7924,  39.2083),
    "Kampala, Uganda":              (0.3476,   32.5825),
    "Khartoum, Sudan":              (15.5007,  32.5599),
    "Nairobi, Kenya":               (-1.2921,  36.8219),
    # --- Southern Africa ---
    "Blantyre, Malawi":             (-15.7861, 35.0058),
    "Harare, Zimbabwe":             (-17.8252, 31.0335),
    "Lusaka, Zambia":               (-15.4166, 28.2832),
    "Maputo, Mozambique":           (-25.9692, 32.5732),
    # --- United Kingdom ---
    "Birmingham, UK":               (52.4862,  -1.8904),
    "Bristol, UK":                  (51.4545,  -2.5879),
    "Leeds, UK":                    (53.8008,  -1.5491),
    "Leicester, UK":                (52.6369,  -1.1398),
    "London, UK":                   (51.5074,  -0.1278),
    "Manchester, UK":               (53.4808,  -2.2426),
    "Nottingham, UK":               (52.9548,  -1.1581),
    # --- United States ---
    "Atlanta, USA":                 (33.7490,  -84.3880),
    "Baltimore, USA":               (39.2904,  -76.6122),
    "Chicago, USA":                 (41.8781,  -87.6298),
    "Detroit, USA":                 (42.3314,  -83.0458),
    "Houston, USA":                 (29.7604,  -95.3698),
    "Los Angeles, USA":             (34.0522, -118.2437),
    "Memphis, USA":                 (35.1495,  -90.0490),
    "New York, USA":                (40.7128,  -74.0060),
    "Philadelphia, USA":            (39.9526,  -75.1652),
    "Washington DC, USA":           (38.9072,  -77.0369),
    # --- Caribbean ---
    "Bridgetown, Barbados":         (13.1132,  -59.5988),
    "Kingston, Jamaica":            (17.9970,  -76.7936),
    "Nassau, Bahamas":              (25.0480,  -77.3554),
    "Port of Spain, Trinidad":      (10.6596,  -61.5086),
    # --- South America ---
    "Rio de Janeiro, Brazil":       (-22.9068, -43.1729),
    "São Paulo, Brazil":            (-23.5505, -46.6333),
    # --- Europe ---
    "Amsterdam, Netherlands":       (52.3676,    4.9041),
    "Brussels, Belgium":            (50.8503,    4.3517),
    "Paris, France":                (48.8566,    2.3522),
    # --- Custom ---
    "Custom coordinates…":          (0.0, 0.0),
}

_DEFAULT_LOCATION = "Lagos, Nigeria"


def _checkin_form(pid: str, key_prefix: str) -> None:
    """Full check-in form with weather, shared between CHW and Patient views."""
    with st.form(f"{key_prefix}_checkin_form"):
        st.subheader("Symptoms")

        r1c1, r1c2, r1c3 = st.columns(3)
        pain_score           = r1c1.slider("Pain score (0–10)", 0, 10, 0,
                                           key=f"{key_prefix}_pain_ci")
        fluid_intake_glasses = r1c2.number_input("Fluid intake (glasses)", 0, 30, 6,
                                                  key=f"{key_prefix}_fluid_ci")
        urine_colour         = r1c3.select_slider(
            "Urine colour",
            options=list(range(1, 9)),
            format_func=lambda x: f"{x} — " + [
                "Pale straw", "Straw", "Yellow", "Dark yellow",
                "Amber", "Dark amber", "Honey", "Brown"
            ][x - 1],
            value=3,
            key=f"{key_prefix}_urine_ci",
        )

        r2c1, r2c2, r2c3 = st.columns(3)
        body_temp_c   = r2c1.number_input("Body temp (°C)", 35.0, 43.0, 36.8, 0.1,
                                           key=f"{key_prefix}_temp_ci")
        fever_present = r2c2.checkbox("Fever", key=f"{key_prefix}_fever_ci")
        med_taken     = r2c3.checkbox("Medication taken today", value=True,
                                      key=f"{key_prefix}_med_ci")

        st.markdown("**Sleep last night**")
        sl1, sl2, sl3 = st.columns([2, 2, 1])
        sleep_hour = sl1.selectbox(
            "Hours", list(range(0, 13)), index=7,
            key=f"{key_prefix}_sleep_h",
        )
        sleep_min = sl2.selectbox(
            "Minutes", [0, 15, 30, 45],
            format_func=lambda x: f"{x:02d}",
            key=f"{key_prefix}_sleep_m",
        )
        sleep_ampm = sl3.radio(
            "AM / PM", ["AM", "PM"], index=0,
            key=f"{key_prefix}_sleep_ap",
        )
        # AM = under 12 h duration; PM = over 12 h (e.g. illness recovery)
        sleep_hours = float(
            sleep_hour + (12 if sleep_ampm == "PM" else 0) + sleep_min / 60
        )

        st.subheader("Location (for weather data)")
        use_location = st.checkbox("Include location", value=True,
                                   key=f"{key_prefix}_useloc_ci")
        city = st.selectbox(
            "Search city",
            options=list(_LOCATIONS.keys()),
            index=list(_LOCATIONS.keys()).index(_DEFAULT_LOCATION),
            key=f"{key_prefix}_city_ci",
            disabled=not use_location,
        )
        if city == "Custom coordinates…" and use_location:
            lc1, lc2 = st.columns(2)
            latitude  = lc1.number_input("Latitude",  -90.0,  90.0, 0.0, 0.0001,
                                         format="%.4f", key=f"{key_prefix}_lat_ci")
            longitude = lc2.number_input("Longitude", -180.0, 180.0, 0.0, 0.0001,
                                         format="%.4f", key=f"{key_prefix}_lon_ci")
        else:
            lat_lon   = _LOCATIONS.get(city, (0.0, 0.0))
            latitude, longitude = lat_lon
            if use_location:
                st.caption(f"📍 {city}  ·  {latitude:.4f}, {longitude:.4f}")

        submitted = st.form_submit_button("Submit check-in", type="primary",
                                          use_container_width=True)

    if submitted:
        payload: dict = {
            "patient_id": pid,
            "pain_score": int(pain_score),
            "fluid_intake_glasses": int(fluid_intake_glasses),
            "urine_colour": int(urine_colour),
            "body_temp_c": round(float(body_temp_c), 1),
            "fever_present": fever_present,
            "med_taken": med_taken,
            "sleep_hours": float(sleep_hours),
        }
        if use_location and city != "Custom coordinates…":
            payload["latitude"]  = round(float(latitude), 4)
            payload["longitude"] = round(float(longitude), 4)
        elif use_location and city == "Custom coordinates…":
            payload["latitude"]  = round(float(latitude), 4)
            payload["longitude"] = round(float(longitude), 4)

        resp = _api("post", "/checkin", headers=_headers(), json=payload)
        if resp and resp.status_code == 200:
            _show_checkin_result(resp.json())
        else:
            detail = resp.json().get("detail", "Unknown error") if resp else "No response"
            st.error(f"Check-in failed: {detail}")


def _show_checkin_result(data: dict) -> None:
    tier   = data.get("risk_tier", "LOW")
    score  = data.get("risk_score", 0.0)
    colour = {"HIGH": "#A32D2D", "MODERATE": "#854F0B", "LOW": "#3B6D11"}.get(tier, "#378ADD")

    st.divider()

    # Risk gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score * 100, 1),
        number={"suffix": "%", "font": {"color": colour}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": colour},
            "steps": [
                {"range": [0, 30],   "color": "#dfd"},
                {"range": [30, 60],  "color": "#ffd"},
                {"range": [60, 100], "color": "#fdd"},
            ],
        },
        title={"text": f"VOC Risk — {tier}"},
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # SHAP factors
    shap_factors = data.get("shap_factors", [])
    if shap_factors:
        st.markdown("**Top risk factors**")
        for sf in shap_factors:
            icon   = "📈" if sf.get("direction") == "increases" else "📉"
            name   = sf["factor"].replace("_", " ").title()
            contrib = sf.get("contribution")
            suffix  = f"  `{contrib:+.3f}`" if contrib is not None else ""
            st.markdown(f"{icon} **{name}** — {sf['direction']} risk{suffix}")

    if data.get("suggestion"):
        st.info(f"💬 {data['suggestion']}")

    # Weather + hydration row
    weather_alerts: list[str] = data.get("weather_alerts", [])
    col_w, col_h = st.columns(2)

    with col_w:
        st.markdown("**🌤 Weather alerts**")
        if weather_alerts:
            for alert in weather_alerts:
                icon = "🥵" if "heat" in alert.lower() else ("🥶" if "cold" in alert.lower() else "💨")
                st.markdown(
                    f'<div class="alert-box">{icon} {alert}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("✅ No weather risk alerts.")

    with col_h:
        st.markdown("**💧 Hydration**")
        hstatus = data.get("hydration_status", "—")
        hicon   = "✅" if hstatus == "WELL_HYDRATED" else "⚠️"
        st.markdown(f"{hicon} **{hstatus.replace('_', ' ').title()}**")
        if data.get("hydration_message"):
            st.caption(data["hydration_message"])
        if data.get("hydration_advice"):
            st.markdown(
                f'<div class="alert-box">📋 {data["hydration_advice"]}</div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Shared — Pain forms and charts
# ---------------------------------------------------------------------------

_PAIN_LOCATIONS = ["CHEST", "BACK", "ABDOMEN", "L_ARM", "R_ARM", "L_LEG", "R_LEG", "HEAD", "OTHER"]
_LOCATION_LABELS = {
    "CHEST": "Chest", "BACK": "Back", "ABDOMEN": "Abdomen",
    "L_ARM": "Left arm", "R_ARM": "Right arm",
    "L_LEG": "Left leg", "R_LEG": "Right leg",
    "HEAD": "Head", "OTHER": "Other",
}


def _pain_log_form(pid: str, key_prefix: str) -> None:
    with st.form(f"{key_prefix}_pain_form"):
        locations = st.multiselect(
            "Where does it hurt?",
            options=_PAIN_LOCATIONS,
            format_func=lambda x: _LOCATION_LABELS.get(x, x),
            key=f"{key_prefix}_locations",
        )

        pain_score = st.slider("Pain score (0 = none, 10 = worst)", 0, 10, 0,
                               key=f"{key_prefix}_pain_score")

        st.markdown("**Triggers**")
        tc1, tc2, tc3 = st.columns(3)
        trigger_cold        = tc1.checkbox("Cold exposure",  key=f"{key_prefix}_tcold")
        trigger_stress      = tc1.checkbox("Stress",         key=f"{key_prefix}_tstress")
        trigger_exercise    = tc2.checkbox("Exercise",       key=f"{key_prefix}_texercise")
        trigger_infection   = tc2.checkbox("Infection",      key=f"{key_prefix}_tinfect")
        trigger_dehydration = tc3.checkbox("Dehydration",    key=f"{key_prefix}_tdehydr")
        trigger_other       = st.text_input("Other trigger", key=f"{key_prefix}_tother")

        st.markdown("**Medication taken**")
        mc1, mc2, mc3 = st.columns(3)
        took_paracetamol = mc1.checkbox("Paracetamol", key=f"{key_prefix}_para")
        took_ibuprofen   = mc2.checkbox("Ibuprofen",   key=f"{key_prefix}_ibu")
        took_opioid      = mc3.checkbox("Opioid",      key=f"{key_prefix}_opi")

        any_med = took_paracetamol or took_ibuprofen or took_opioid
        relief  = st.select_slider(
            "Pain relief effectiveness",
            [0, 1, 2, 3],
            format_func=lambda x: ["No medication / no effect", "Mild", "Moderate", "Good"][x],
            value=0,
            key=f"{key_prefix}_relief",
        )
        notes = st.text_area("Notes (optional)", height=70, key=f"{key_prefix}_notes")

        submitted = st.form_submit_button("Log pain entry", type="primary",
                                          use_container_width=True)

    if submitted:
        resp = _api("post", "/pain", headers=_headers(), json={
            "patient_id": pid,
            "pain_score": int(pain_score),
            "pain_locations": locations or None,
            "trigger_cold": trigger_cold,
            "trigger_stress": trigger_stress,
            "trigger_exercise": trigger_exercise,
            "trigger_infection": trigger_infection,
            "trigger_dehydration": trigger_dehydration,
            "trigger_other": trigger_other or None,
            "took_paracetamol": took_paracetamol,
            "took_ibuprofen": took_ibuprofen,
            "took_opioid": took_opioid,
            "pain_relief_rating": int(relief) if any_med else None,
            "notes": notes or None,
        })
        if resp is not None and resp.status_code == 200:
            _show_pain_result(resp.json())
        else:
            detail = resp.json().get("detail", "Unknown error") if resp is not None else "No response"
            st.error(f"Failed: {detail}")


def _show_pain_result(data: dict) -> None:
    if data.get("chest_pain_alert"):
        st.markdown(
            '<div class="breakthrough">🚨 <strong>Chest pain detected — CHW alert queued.</strong></div>',
            unsafe_allow_html=True,
        )
    elif data.get("is_breakthrough"):
        st.markdown(
            '<div class="breakthrough">⚠️ <strong>Breakthrough pain event — CHW alert queued.</strong></div>',
            unsafe_allow_html=True,
        )
    else:
        st.success(f"Pain entry logged — score {data['pain_score']}/10")

    c1, c2, c3 = st.columns(3)
    c1.metric("Pain score", f"{data['pain_score']}/10")

    slope = data.get("pain_slope_3d")
    if slope is not None:
        trend = "Rising ▲" if slope > 0.3 else ("Falling ▼" if slope < -0.3 else "Stable →")
        c2.metric("3-day trend", trend, f"{slope:+.1f}/day")
    else:
        c2.metric("3-day trend", "—")

    locs = data.get("pain_locations") or []
    c3.metric("Locations", ", ".join(_LOCATION_LABELS.get(l, l) for l in locs) or "—")

    if data.get("suggestion"):
        st.info(f"💬 {data['suggestion']}")


def _show_pain_history(pid: str) -> None:
    days = st.slider("Days of history", 7, 90, 30, key="pain_hist_slider")
    resp = _api("get", f"/patients/{pid}/pain", headers=_headers(), params={"days": days})
    if not resp or resp.status_code != 200:
        st.warning("Could not load pain history.")
        return
    entries: list[dict] = resp.json()
    if not entries:
        st.info("No pain diary entries in this period.")
        return

    _pain_history_chart(entries)
    st.caption(f"{len(entries)} entries in the last {days} days")

    with st.expander("View all entries"):
        for e in reversed(entries):
            ts    = e.get("recorded_at", "")[:16].replace("T", " ")
            score = e.get("pain_score", 0)
            locs  = ", ".join(_LOCATION_LABELS.get(l, l) for l in (e.get("pain_locations") or []))
            flags = []
            if e.get("is_breakthrough"):
                flags.append("⭐ Breakthrough")
            if e.get("chest_pain_alert"):
                flags.append("🚨 Chest pain")
            st.markdown(
                f"`{ts}` — **{score}/10**"
                + (f" · {locs}" if locs else "")
                + (f" · {' · '.join(flags)}" if flags else "")
            )


def _pain_history_chart(entries: list[dict]) -> None:
    dates  = [e.get("recorded_at", "")[:10] for e in entries]
    scores = [e.get("pain_score", 0) for e in entries]
    breaks = [e.get("is_breakthrough", False) for e in entries]

    fig = go.Figure(go.Scatter(
        x=dates, y=scores, mode="lines+markers", name="Pain score",
        line=dict(color="#A32D2D", width=2),
        marker=dict(
            size=[12 if b else 7 for b in breaks],
            color=["#A32D2D" if b else "#E07070" for b in breaks],
            symbol=["star" if b else "circle" for b in breaks],
        ),
    ))
    fig.add_hline(y=7, line_dash="dash", line_color="orange",
                  annotation_text="Breakthrough threshold")
    fig.update_layout(
        height=240, margin=dict(l=0, r=0, t=30, b=0),
        yaxis=dict(title="Pain score", range=[0, 10]),
        plot_bgcolor="white",
        title="Pain history — ⭐ breakthrough events",
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Shared — Hydration forms and charts
# ---------------------------------------------------------------------------

_DRINK_TYPES  = ["WATER", "JUICE", "MILK", "TEA", "COFFEE", "SODA", "OTHER"]
_DRINK_LABELS = {
    "WATER": "💧 Water", "JUICE": "🥤 Juice", "MILK": "🥛 Milk",
    "TEA":   "🍵 Tea",   "COFFEE": "☕ Coffee", "SODA": "🫧 Soda",
    "OTHER": "Other",
}
_HYDRATION_COLOUR = {
    "WELL_HYDRATED": "#3B6D11", "MILD_RISK": "#854F0B",
    "MODERATE_RISK": "#854F0B", "SEVERE_RISK": "#A32D2D", "CRITICAL": "#A32D2D",
}
_WHO_TARGET_ML = 2000


def _hydration_log_form(pid: str, key_prefix: str) -> None:
    with st.form(f"{key_prefix}_hydration_form"):
        col1, col2 = st.columns(2)
        with col1:
            drink_type = st.selectbox(
                "Drink type", _DRINK_TYPES,
                format_func=lambda x: _DRINK_LABELS[x],
                key=f"{key_prefix}_dtype",
            )
            drink_vol = st.number_input(
                "Volume (ml)", min_value=50, max_value=2000, value=250, step=50,
                key=f"{key_prefix}_dvol",
            )
            if drink_type in ("TEA", "COFFEE"):
                st.caption("⚠️ Stored at 80% effective volume (mild diuretic).")

        with col2:
            urine_colour = st.select_slider(
                "Urine colour (Armstrong scale)",
                options=list(range(1, 9)),
                format_func=lambda x: {
                    1: "1 — Pale straw", 2: "2 — Straw", 3: "3 — Yellow",
                    4: "4 — Dark yellow", 5: "5 — Amber", 6: "6 — Dark amber",
                    7: "7 — Honey",      8: "8 — Brown",
                }[x],
                value=3,
                key=f"{key_prefix}_uc",
            )
            thirst = st.select_slider(
                "Thirst level", [1, 2, 3, 4],
                format_func=lambda x: ["1 — Not thirsty", "2 — Slightly", "3 — Moderately", "4 — Very thirsty"][x - 1],
                value=1,
                key=f"{key_prefix}_thirst",
            )

        s1, s2, s3 = st.columns(3)
        dry_mouth = s1.checkbox("Dry mouth",  key=f"{key_prefix}_dry")
        dizziness = s2.checkbox("Dizziness",  key=f"{key_prefix}_dizzy")
        headache  = s3.checkbox("Headache",   key=f"{key_prefix}_head")

        submitted = st.form_submit_button("Log drink", type="primary",
                                          use_container_width=True)

    if submitted:
        resp = _api("post", "/hydration", headers=_headers(), json={
            "patient_id": pid,
            "drink_type": drink_type,
            "drink_volume_ml": int(drink_vol),
            "urine_colour": int(urine_colour),
            "thirst_level": int(thirst),
            "dry_mouth": dry_mouth,
            "dizziness": dizziness,
            "headache": headache,
        })
        if resp and resp.status_code == 200:
            _show_hydration_result(resp.json())
        else:
            detail = resp.json().get("detail", "Unknown error") if resp else "No response"
            st.error(f"Failed: {detail}")


def _show_hydration_result(data: dict) -> None:
    status    = data.get("hydration_status", "")
    colour    = _HYDRATION_COLOUR.get(status, "#378ADD")
    daily_ml  = data.get("daily_total_ml", 0)
    daily_gl  = data.get("daily_total_glasses", 0.0)
    pct       = min(daily_ml / _WHO_TARGET_ML * 100, 100)

    st.success(f"{_DRINK_LABELS.get(data['drink_type'], data['drink_type'])} — {data['drink_volume_ml']} ml logged")

    c1, c2, c3 = st.columns(3)
    c1.metric("Daily total", f"{daily_ml} ml",  f"{daily_gl:.1f} glasses")
    c2.metric("WHO target",  f"{_WHO_TARGET_ML} ml", f"{pct:.0f}% reached")
    c3.metric("Status", status.replace("_", " ").title())

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=daily_ml,
        number={"suffix": " ml"},
        gauge={
            "axis": {"range": [0, 3000]},
            "bar": {"color": colour},
            "steps": [
                {"range": [0, 1000],  "color": "#fdd"},
                {"range": [1000, 1500], "color": "#ffd"},
                {"range": [1500, 2000], "color": "#dfd"},
                {"range": [2000, 3000], "color": "#cfc"},
            ],
            "threshold": {
                "line": {"color": "green", "width": 3},
                "thickness": 0.75,
                "value": _WHO_TARGET_ML,
            },
        },
        title={"text": "Daily fluid intake"},
    ))
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

    if data.get("hydration_message"):
        icon = "✅" if status == "WELL_HYDRATED" else "⚠️"
        st.markdown(
            f'<div class="alert-box">{icon} <strong>{data["hydration_message"]}</strong></div>',
            unsafe_allow_html=True,
        )
    if data.get("hydration_advice"):
        st.info(f"📋 {data['hydration_advice']}")
    if data.get("suggestion"):
        st.caption(f"💬 {data['suggestion']}")


def _hydration_guide() -> None:
    st.subheader("Hydration targets for SCD patients")
    st.markdown("""
**WHO oral rehydration guidelines** (Yallop et al. 2007) recommend **≥ 2 000 ml/day**.
Dehydration increases VOC risk (OR ≈ 2.1).

| Urine colour | Meaning | Action |
|---|---|---|
| 1–2 Pale straw | Well hydrated ✅ | Maintain intake |
| 3–4 Yellow | Adequate | Drink more water |
| 5–6 Amber | Mild dehydration ⚠️ | 250 ml water now |
| 7–8 Dark/brown | Severe dehydration 🚨 | Seek medical attention |
    """)
    colours_hex = ["#FFF8DC", "#F5DEB3", "#FFD700", "#DAA520",
                   "#CD853F", "#8B6914", "#8B4513", "#4B2F1A"]
    fig = go.Figure(go.Bar(
        x=[f"Colour {i}" for i in range(1, 9)],
        y=[1] * 8,
        marker_color=colours_hex,
        text=["Pale straw", "Straw", "Yellow", "Dark yellow",
              "Amber", "Dark amber", "Honey", "Brown"],
        textposition="inside",
        hovertemplate="%{text}<extra></extra>",
    ))
    fig.update_layout(
        height=110, margin=dict(l=0, r=0, t=10, b=30),
        showlegend=False, yaxis=dict(visible=False),
        title="Armstrong (1994) urine colour chart",
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# About
# ---------------------------------------------------------------------------

def about_page() -> None:
    st.header("About Warrior Blood")
    role = st.session_state.get("role", "patient")
    st.markdown(f"""
**Warrior Blood** is a Sickle Cell Disease (SCD) patient monitoring system developed as an
MSc Software Engineering dissertation at the University of Greater Manchester.

**Renee Tucker · Supervisor: Aamir Abbas · 2025–26**

---

### How it works
{"**CHW view:** Register patients, monitor triage list sorted by VOC risk, view patient-logged pain and hydration data in real time, send manual SMS alerts." if role in ("chw","admin") else "**Patient view:** Log daily symptoms, pain diary entries and hydration intake. Your data is immediately visible to your Community Health Worker."}

### Clinical references
- Machado et al. (2024) — LightGBM on SCD prediction (AUROC 0.808)
- Brandow et al. (2020) — pain trajectories in SCD
- Yallop et al. (2007) — dehydration, AQI and SCD hospitalisation
- Nolan et al. (2008) — temperature and SCD hospitalisation
- Wahl et al. (2018) — interpretability in LMIC health AI
- Lundberg & Lee (2017) — SHAP values
- Armstrong (1994) — urine colour chart (1–8 scale)
- WHO (2005) — oral rehydration guidelines
    """)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

main()

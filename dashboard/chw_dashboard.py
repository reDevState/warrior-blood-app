"""
chw_dashboard.py — Warrior Blood Community Health Worker Triage Dashboard.

Streamlit app connecting to the FastAPI backend. CHWs use this to:
  - View all patients sorted by VOC risk tier (HIGH first)
  - Log pain diary entries with trend detection
  - Log hydration entries with daily progress tracking
  - Submit check-ins with weather and environmental risk data
  - Register patients and send manual SMS alerts

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
    .risk-HIGH     { color: #A32D2D; font-weight: 600; }
    .risk-MODERATE { color: #854F0B; font-weight: 600; }
    .risk-LOW      { color: #3B6D11; font-weight: 600; }
    .metric-card   { background: #f8f8f8; border-radius: 8px;
                     padding: 12px 16px; margin-bottom: 8px; }
    .alert-box     { background: #fff4e5; border-left: 4px solid #e07b00;
                     padding: 10px 14px; border-radius: 4px; margin: 6px 0; }
    .breakthrough  { background: #fff0f0; border-left: 4px solid #A32D2D;
                     padding: 10px 14px; border-radius: 4px; }
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


def _patient_options() -> list[dict]:
    """Fetch all patients and return list for selectbox."""
    resp = _api("get", "/patients", headers=_headers())
    if resp and resp.status_code == 200:
        return resp.json()
    return []


def _patient_label(p: dict) -> str:
    name = p.get("name") or f"Patient {p['id'][:8]}…"
    tier = p.get("latest_risk_tier") or "—"
    icon = {"HIGH": "🔴", "MODERATE": "🟡", "LOW": "🟢"}.get(tier, "⚪")
    return f"{icon} {name}"


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------

def login_page() -> None:
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

def dashboard() -> None:
    with st.sidebar:
        st.markdown("### 🩸 Warrior Blood")
        st.caption(f"Signed in as: **{st.session_state.get('username', '')}**")
        st.divider()
        page = st.radio(
            "View",
            [
                "Patient triage",
                "Pain diary",
                "Hydration diary",
                "Weather & check-in",
                "Register patient",
                "About",
            ],
        )
        if st.button("Sign out"):
            st.session_state.clear()
            st.rerun()

    if page == "Patient triage":
        triage_page()
    elif page == "Pain diary":
        pain_diary_page()
    elif page == "Hydration diary":
        hydration_diary_page()
    elif page == "Weather & check-in":
        weather_checkin_page()
    elif page == "Register patient":
        register_page()
    else:
        about_page()


# ---------------------------------------------------------------------------
# Triage page
# ---------------------------------------------------------------------------

def triage_page() -> None:
    st.header("Patient triage")

    resp = _api("get", "/patients", headers=_headers())
    if not resp or resp.status_code != 200:
        st.warning("Could not load patient list.")
        return

    patients: list[dict] = resp.json()

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
        tier = p.get("latest_risk_tier") or "—"
        tier_colour = {"HIGH": "🔴", "MODERATE": "🟡", "LOW": "🟢"}.get(tier, "⚪")
        enrolled = p.get("enrolled_at", "")[:10]
        display_name = p.get("name") or f"Patient {p['id'][:8]}..."

        with st.expander(f"{tier_colour} {display_name}  |  {tier}  |  enrolled {enrolled}"):
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


def timeline_chart(history: list[dict]) -> None:
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
                fillcolor="rgba(163,45,45,0.15)", line_width=0,
            )
    fig.update_layout(
        height=200, margin=dict(l=0, r=0, t=20, b=0), showlegend=True,
        legend=dict(orientation="h", y=1.15),
        yaxis=dict(title="Pain (0-10)", range=[0, 10], side="left"),
        yaxis2=dict(title="Risk (0-1)", range=[0, 1], side="right", overlaying="y"),
        xaxis=dict(title=""), plot_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Pain diary page
# ---------------------------------------------------------------------------

_PAIN_LOCATIONS = ["CHEST", "BACK", "ABDOMEN", "L_ARM", "R_ARM", "L_LEG", "R_LEG", "HEAD", "OTHER"]
_LOCATION_LABELS = {
    "CHEST": "Chest", "BACK": "Back", "ABDOMEN": "Abdomen",
    "L_ARM": "Left arm", "R_ARM": "Right arm",
    "L_LEG": "Left leg", "R_LEG": "Right leg",
    "HEAD": "Head", "OTHER": "Other",
}


def pain_diary_page() -> None:
    st.header("Pain diary")

    patients = _patient_options()
    if not patients:
        st.warning("No patients found. Register a patient first.")
        return

    patient_map = {_patient_label(p): p for p in patients}
    selected_label = st.selectbox("Select patient", list(patient_map.keys()))
    patient = patient_map[selected_label]
    patient_id = patient["id"]

    tab_log, tab_history = st.tabs(["Log entry", "Pain history"])

    # --- Log entry ---
    with tab_log:
        with st.form("pain_entry"):
            st.subheader("New pain entry")

            pain_score = st.slider("Pain score (0 = none, 10 = worst imaginable)", 0, 10, 0)

            col1, col2 = st.columns(2)
            with col1:
                selected_locations = st.multiselect(
                    "Pain locations",
                    options=_PAIN_LOCATIONS,
                    format_func=lambda x: _LOCATION_LABELS[x],
                )

            with col2:
                st.markdown("**Triggers**")
                trigger_cold        = st.checkbox("Cold exposure")
                trigger_stress      = st.checkbox("Stress")
                trigger_exercise    = st.checkbox("Exercise")
                trigger_infection   = st.checkbox("Infection / illness")
                trigger_dehydration = st.checkbox("Dehydration")
                trigger_other = st.text_input("Other trigger (describe)")

            st.markdown("**Medication taken**")
            mc1, mc2, mc3 = st.columns(3)
            took_paracetamol = mc1.checkbox("Paracetamol")
            took_ibuprofen   = mc2.checkbox("Ibuprofen")
            took_opioid      = mc3.checkbox("Opioid")

            pain_relief_rating = st.select_slider(
                "Pain relief effectiveness",
                options=[0, 1, 2, 3],
                format_func=lambda x: ["None taken / no effect", "Mild", "Moderate", "Good"][x],
                value=0,
            )

            notes = st.text_area("Notes (optional)", height=80)

            submitted = st.form_submit_button("Log pain entry", type="primary")

        if submitted:
            payload = {
                "patient_id": patient_id,
                "pain_score": pain_score,
                "pain_locations": selected_locations or None,
                "trigger_cold": trigger_cold,
                "trigger_stress": trigger_stress,
                "trigger_exercise": trigger_exercise,
                "trigger_infection": trigger_infection,
                "trigger_dehydration": trigger_dehydration,
                "trigger_other": trigger_other or None,
                "took_paracetamol": took_paracetamol,
                "took_ibuprofen": took_ibuprofen,
                "took_opioid": took_opioid,
                "pain_relief_rating": pain_relief_rating if (took_paracetamol or took_ibuprofen or took_opioid) else None,
                "notes": notes or None,
            }
            resp = _api("post", "/pain", headers=_headers(), json=payload)
            if resp and resp.status_code == 200:
                data = resp.json()
                _show_pain_result(data)
            else:
                detail = resp.json().get("detail", "Unknown error") if resp else "No response"
                st.error(f"Failed to log pain entry: {detail}")

    # --- History ---
    with tab_history:
        _show_pain_history(patient_id)


def _show_pain_result(data: dict) -> None:
    if data.get("chest_pain_alert"):
        st.markdown(
            '<div class="breakthrough">🚨 <strong>Chest pain detected — CHW alert has been queued.</strong></div>',
            unsafe_allow_html=True,
        )
    elif data.get("is_breakthrough"):
        st.markdown(
            '<div class="breakthrough">⚠️ <strong>Breakthrough pain event detected — CHW alert queued.</strong></div>',
            unsafe_allow_html=True,
        )
    else:
        st.success(f"Pain entry logged (score: {data['pain_score']}/10)")

    col1, col2, col3 = st.columns(3)
    col1.metric("Pain score", f"{data['pain_score']}/10")

    slope = data.get("pain_slope_3d")
    if slope is not None:
        trend_label = "Rising ▲" if slope > 0.3 else ("Falling ▼" if slope < -0.3 else "Stable →")
        col2.metric("3-day trend", trend_label, delta=f"{slope:+.1f}/day")
    else:
        col2.metric("3-day trend", "Insufficient data")

    locs = data.get("pain_locations") or []
    col3.metric("Locations", ", ".join(_LOCATION_LABELS.get(l, l) for l in locs) or "—")

    if data.get("suggestion"):
        st.info(f"💬 {data['suggestion']}")


def _show_pain_history(patient_id: str) -> None:
    days = st.slider("Days of history", 7, 90, 30, key="pain_hist_days")
    resp = _api("get", f"/patients/{patient_id}/pain", headers=_headers(), params={"days": days})

    if not resp or resp.status_code != 200:
        st.warning("Could not load pain history.")
        return

    entries: list[dict] = resp.json()
    if not entries:
        st.info("No pain diary entries in this period.")
        return

    st.caption(f"{len(entries)} entries in the last {days} days")

    dates  = [e.get("recorded_at", "")[:10] for e in entries]
    scores = [e.get("pain_score", 0) for e in entries]
    breaks = [e.get("is_breakthrough", False) for e in entries]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=scores, name="Pain score",
        mode="lines+markers",
        line=dict(color="#A32D2D", width=2),
        marker=dict(
            size=[12 if b else 7 for b in breaks],
            color=["#A32D2D" if b else "#E07070" for b in breaks],
            symbol=["star" if b else "circle" for b in breaks],
        ),
    ))
    fig.add_hline(y=7, line_dash="dash", line_color="orange",
                  annotation_text="Breakthrough threshold (7)")
    fig.update_layout(
        height=280, margin=dict(l=0, r=0, t=30, b=0),
        yaxis=dict(title="Pain score", range=[0, 10]),
        xaxis=dict(title=""),
        plot_bgcolor="white",
        title="Pain score history — ⭐ = breakthrough event",
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View raw entries"):
        for e in reversed(entries):
            ts = e.get("recorded_at", "")[:16].replace("T", " ")
            score = e.get("pain_score", 0)
            locs = ", ".join(_LOCATION_LABELS.get(l, l) for l in (e.get("pain_locations") or []))
            flag = "⭐ Breakthrough" if e.get("is_breakthrough") else ""
            chest = "🚨 Chest pain" if e.get("chest_pain_alert") else ""
            st.markdown(f"**{ts}** — Score {score}/10 {locs and f'| {locs}'} {flag} {chest}")


# ---------------------------------------------------------------------------
# Hydration diary page
# ---------------------------------------------------------------------------

_DRINK_TYPES = ["WATER", "JUICE", "MILK", "TEA", "COFFEE", "SODA", "OTHER"]
_DRINK_LABELS = {
    "WATER": "💧 Water", "JUICE": "🥤 Juice", "MILK": "🥛 Milk",
    "TEA": "🍵 Tea", "COFFEE": "☕ Coffee", "SODA": "🥤 Soda", "OTHER": "Other",
}
_HYDRATION_COLOURS = {
    "WELL_HYDRATED": "#3B6D11",
    "MILD_RISK": "#854F0B",
    "MODERATE_RISK": "#854F0B",
    "SEVERE_RISK": "#A32D2D",
    "CRITICAL": "#A32D2D",
}
_WHO_TARGET_ML = 2000  # WHO oral rehydration target (Yallop et al. 2007)


def hydration_diary_page() -> None:
    st.header("Hydration diary")

    patients = _patient_options()
    if not patients:
        st.warning("No patients found. Register a patient first.")
        return

    patient_map = {_patient_label(p): p for p in patients}
    selected_label = st.selectbox("Select patient", list(patient_map.keys()), key="hydration_patient")
    patient_id = patient_map[selected_label]["id"]

    tab_log, tab_guide = st.tabs(["Log drink", "Hydration guide"])

    with tab_log:
        with st.form("hydration_entry"):
            st.subheader("Log a drink")

            col1, col2 = st.columns(2)
            with col1:
                drink_type = st.selectbox(
                    "Drink type",
                    _DRINK_TYPES,
                    format_func=lambda x: _DRINK_LABELS[x],
                )
                drink_volume_ml = st.number_input(
                    "Volume (ml)", min_value=50, max_value=2000, value=250, step=50,
                )
                if drink_type in ("TEA", "COFFEE"):
                    st.caption("⚠️ Tea/coffee stored at 80% effective volume due to mild diuretic effect.")

            with col2:
                urine_colour = st.select_slider(
                    "Urine colour (Armstrong scale)",
                    options=list(range(1, 9)),
                    format_func=lambda x: {
                        1: "1 — Pale straw", 2: "2 — Straw", 3: "3 — Yellow",
                        4: "4 — Dark yellow", 5: "5 — Amber", 6: "6 — Dark amber",
                        7: "7 — Honey", 8: "8 — Brown",
                    }[x],
                    value=3,
                )
                thirst_level = st.select_slider(
                    "Thirst level",
                    options=[1, 2, 3, 4],
                    format_func=lambda x: ["1 — Not thirsty", "2 — Slightly", "3 — Moderately", "4 — Very thirsty"][x - 1],
                    value=1,
                )

            st.markdown("**Symptoms**")
            sc1, sc2, sc3 = st.columns(3)
            dry_mouth  = sc1.checkbox("Dry mouth")
            dizziness  = sc2.checkbox("Dizziness")
            headache   = sc3.checkbox("Headache")

            submitted = st.form_submit_button("Log drink", type="primary")

        if submitted:
            payload = {
                "patient_id": patient_id,
                "drink_type": drink_type,
                "drink_volume_ml": int(drink_volume_ml),
                "urine_colour": urine_colour,
                "thirst_level": thirst_level,
                "dry_mouth": dry_mouth,
                "dizziness": dizziness,
                "headache": headache,
            }
            resp = _api("post", "/hydration", headers=_headers(), json=payload)
            if resp and resp.status_code == 200:
                _show_hydration_result(resp.json())
            else:
                detail = resp.json().get("detail", "Unknown error") if resp else "No response"
                st.error(f"Failed to log drink: {detail}")

    with tab_guide:
        _show_hydration_guide()


def _show_hydration_result(data: dict) -> None:
    status = data.get("hydration_status", "")
    colour = _HYDRATION_COLOURS.get(status, "#378ADD")

    daily_ml  = data.get("daily_total_ml", 0)
    daily_gl  = data.get("daily_total_glasses", 0.0)
    pct       = min(daily_ml / _WHO_TARGET_ML * 100, 100)

    st.success(f"Drink logged: {_DRINK_LABELS.get(data['drink_type'], data['drink_type'])} — {data['drink_volume_ml']} ml")

    col1, col2, col3 = st.columns(3)
    col1.metric("Daily total", f"{daily_ml} ml", f"{daily_gl:.1f} glasses")
    col2.metric("WHO target", f"{_WHO_TARGET_ML} ml", f"{pct:.0f}% reached")
    col3.metric("Status", status.replace("_", " ").title())

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
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=0))
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


def _show_hydration_guide() -> None:
    st.subheader("Hydration targets for SCD patients")
    st.markdown("""
**WHO oral rehydration guidelines** (Yallop et al. 2007) recommend **≥ 2 000 ml/day**
for SCD patients. Dehydration increases VOC risk (OR ≈ 2.1).

| Urine colour | Meaning | Action |
|---|---|---|
| 1–2 (pale straw) | Well hydrated ✅ | Maintain intake |
| 3–4 (yellow) | Adequate | Drink more water |
| 5–6 (amber) | Mild dehydration ⚠️ | 250 ml water now |
| 7–8 (dark/brown) | Severe dehydration 🚨 | Seek medical attention |

**Diuretic note:** Tea and coffee are stored at 80% effective volume due to
mild diuretic effect. Encourage water as primary fluid.
    """)

    urine_colours = list(range(1, 9))
    colours_hex   = ["#FFF8DC", "#F5DEB3", "#FFD700", "#DAA520",
                     "#CD853F", "#8B6914", "#8B4513", "#4B2F1A"]

    fig = go.Figure(go.Bar(
        x=[f"Colour {i}" for i in urine_colours],
        y=[1] * 8,
        marker_color=colours_hex,
        text=["Pale straw", "Straw", "Yellow", "Dark yellow",
              "Amber", "Dark amber", "Honey", "Brown"],
        textposition="inside",
        hovertemplate="%{text}<extra></extra>",
    ))
    fig.update_layout(
        height=120, margin=dict(l=0, r=0, t=20, b=30),
        showlegend=False, yaxis=dict(visible=False),
        title="Armstrong (1994) urine colour chart",
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Weather & check-in page
# ---------------------------------------------------------------------------

_RISK_COLOURS = {"HIGH": "#A32D2D", "MODERATE": "#854F0B", "LOW": "#3B6D11"}

# Lagos, Nigeria as default (central SCD-prevalent region)
_DEFAULT_LAT = 6.5244
_DEFAULT_LON = 3.3792


def weather_checkin_page() -> None:
    st.header("Weather & check-in")
    st.caption(
        "Submit a full symptom check-in with GPS coordinates. "
        "Weather and air quality data are retrieved automatically and used in VOC risk scoring "
        "(Nolan et al. 2008; Yallop et al. 2007)."
    )

    patients = _patient_options()
    if not patients:
        st.warning("No patients found. Register a patient first.")
        return

    patient_map = {_patient_label(p): p for p in patients}
    selected_label = st.selectbox("Select patient", list(patient_map.keys()), key="weather_patient")
    patient_id = patient_map[selected_label]["id"]

    with st.form("weather_checkin"):
        st.subheader("Patient symptoms")

        sc1, sc2, sc3 = st.columns(3)
        pain_score          = sc1.slider("Pain score", 0, 10, 0)
        fluid_intake_glasses = sc2.number_input("Fluid intake (glasses)", min_value=0, max_value=30, value=6)
        urine_colour        = sc3.select_slider(
            "Urine colour", options=list(range(1, 9)),
            format_func=lambda x: f"{x} — " + ["Pale straw","Straw","Yellow","Dark yellow","Amber","Dark amber","Honey","Brown"][x-1],
            value=3,
        )

        sc4, sc5, sc6 = st.columns(3)
        body_temp_c = sc4.number_input("Body temp (°C)", min_value=35.0, max_value=43.0, value=36.8, step=0.1)
        fever_present = sc5.checkbox("Fever present")
        med_taken = sc6.checkbox("Medication taken today", value=True)

        sleep_hours = st.slider("Sleep last night (hours)", 0.0, 12.0, 7.0, 0.5)

        st.subheader("Location (for weather data)")
        lc1, lc2 = st.columns(2)
        latitude  = lc1.number_input("Latitude",  min_value=-90.0,  max_value=90.0,  value=_DEFAULT_LAT, step=0.0001, format="%.4f")
        longitude = lc2.number_input("Longitude", min_value=-180.0, max_value=180.0, value=_DEFAULT_LON, step=0.0001, format="%.4f")

        use_location = st.checkbox("Include location in check-in (enables weather data)", value=True)

        submitted = st.form_submit_button("Submit check-in", type="primary")

    if submitted:
        payload: dict = {
            "patient_id": patient_id,
            "pain_score": int(pain_score),
            "fluid_intake_glasses": int(fluid_intake_glasses),
            "urine_colour": int(urine_colour),
            "body_temp_c": round(float(body_temp_c), 1),
            "fever_present": fever_present,
            "med_taken": med_taken,
            "sleep_hours": float(sleep_hours),
        }
        if use_location:
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
    colour = _RISK_COLOURS.get(tier, "#378ADD")

    st.divider()
    st.subheader("Check-in result")

    # Risk gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(score * 100, 1),
        number={"suffix": "%", "font": {"color": colour}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": colour},
            "steps": [
                {"range": [0, 30],  "color": "#dfd"},
                {"range": [30, 60], "color": "#ffd"},
                {"range": [60, 100], "color": "#fdd"},
            ],
            "threshold": {"line": {"color": colour, "width": 4}, "thickness": 0.8, "value": score * 100},
        },
        title={"text": f"VOC Risk — {tier}"},
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # SHAP factors
    shap_factors: list[dict] = data.get("shap_factors", [])
    if shap_factors:
        st.markdown("**Top risk factors (SHAP)**")
        for sf in shap_factors:
            icon = "📈" if sf.get("direction") == "increases" else "📉"
            factor_name = sf["factor"].replace("_", " ").title()
            contrib = sf.get("contribution")
            contrib_str = f"  `{contrib:+.3f}`" if contrib is not None else ""
            st.markdown(f"{icon} **{factor_name}** — {sf['direction']} risk{contrib_str}")

    # Suggestion
    if data.get("suggestion"):
        st.info(f"💬 {data['suggestion']}")

    st.divider()

    # Weather alerts
    weather_alerts: list[str] = data.get("weather_alerts", [])
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🌤 Weather alerts**")
        if weather_alerts:
            for alert in weather_alerts:
                icon = "🥵" if "heat" in alert.lower() else ("🥶" if "cold" in alert.lower() else "💨")
                st.markdown(
                    f'<div class="alert-box">{icon} {alert}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("✅ No weather-related VOC risk alerts.")
            st.caption("Ensure location was provided to enable weather data.")

    with col2:
        st.markdown("**💧 Hydration assessment**")
        hydration_status  = data.get("hydration_status", "—")
        hydration_message = data.get("hydration_message", "")
        hydration_advice  = data.get("hydration_advice", "")

        status_icon = "✅" if hydration_status == "WELL_HYDRATED" else "⚠️"
        st.markdown(f"{status_icon} **{hydration_status.replace('_', ' ').title()}**")
        if hydration_message:
            st.caption(hydration_message)
        if hydration_advice:
            st.markdown(
                f'<div class="alert-box">📋 {hydration_advice}</div>',
                unsafe_allow_html=True,
            )

    # Weather reference
    if weather_alerts:
        with st.expander("About weather risk factors"):
            st.markdown("""
**Temperature thresholds** (Nolan et al. 2008):
- Heat stress (> 35 °C) — vasoconstriction and dehydration risk
- Cold stress (< 15 °C) — vasospasm triggering VOC

**Air quality** (Yallop et al. 2007):
- AQI ≥ 3 — significantly associated with SCD hospitalisation
            """)


# ---------------------------------------------------------------------------
# Register patient
# ---------------------------------------------------------------------------

def register_page() -> None:
    st.header("Register new patient")

    with st.form("register"):
        name      = st.text_input("Full name")
        phone     = st.text_input("Phone number (optional)")
        dob       = st.date_input("Date of birth", value=None)
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
            st.success(f"Patient registered. ID: `{pid}`")
        else:
            detail = resp.json().get("detail", "Unknown error") if resp else "No response"
            st.error(f"Registration failed: {detail}")


# ---------------------------------------------------------------------------
# About
# ---------------------------------------------------------------------------

def about_page() -> None:
    st.header("About Warrior Blood")
    st.markdown("""
**Warrior Blood** is a Sickle Cell Disease (SCD) patient monitoring system
developed as an MSc Software Engineering project at the University of Greater Manchester.

**Renee Tucker · Supervisor: Aamir Abbas · 2025–26**

---

### MVP components
- **FastAPI backend** — patient check-in, VOC risk scoring, CHW alerts
- **LightGBM ONNX predictor** — trained on synthetic SCD diary data (AUROC 0.808)
- **SHAP explanations** — top-3 plain-English risk factors per check-in (Lundberg & Lee 2017)
- **Hydration module** — WHO oral rehydration guideline thresholds (Yallop et al. 2007)
- **Pain diary** — trend detection, breakthrough events, chest pain alerts (Brandow et al. 2020)
- **Weather integration** — OpenWeatherMap temperature and AQI risk flags (Nolan et al. 2008)
- **MySQL 8.0** — production-ready database with Fernet-encrypted PHI
- **Streamlit CHW dashboard** — triage, pain diary, hydration diary, weather check-in

### Key references
- Machado et al. (2024) — LightGBM on SCD prediction
- Brandow et al. (2020) — pain trajectories in SCD
- Yallop et al. (2007) — dehydration, AQI and SCD hospitalisation
- Nolan et al. (2008) — temperature and SCD hospitalisation
- Wahl et al. (2018) — interpretability in LMIC health AI
- Lundberg & Lee (2017) — SHAP values
- Armstrong (1994) — urine colour chart (1–8 scale)
- WHO (2005) — oral rehydration guidelines

### MVP limitations
- SMS alerts are stubbed (logged only — integrate Africa's Talking for production)
- No patient-facing mobile app yet (planned: React Native)
    """)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if "token" not in st.session_state:
    login_page()
else:
    dashboard()

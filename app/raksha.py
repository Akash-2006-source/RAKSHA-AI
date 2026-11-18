"""RAKSHA AI platform — Hardcoded Research Landing Page, OAuth 2.0 + OIDC Auth, Clean Navbar & Interactive Modules."""
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import altair as alt
import folium
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import report_content as rc  # noqa: E402

ROOT = APP_DIR.parent
P = ROOT / "data" / "processed"
R = ROOT / "reports"
FIG = R / "report_figures"

st.set_page_config(
    page_title="RAKSHA AI | Road Risk Intelligence & Near-Miss Prediction",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Plain & Clean Minimalist Styling
# ---------------------------------------------------------------------------
CLEAN_CSS = """
<style>
.block-container {
    padding-top: 1.1rem;
    padding-bottom: 2.5rem;
    max-width: 1360px;
}
.topbar {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 0.75rem 1.15rem;
    margin-bottom: 0.6rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.6rem;
}
.brand-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #0F172A;
    margin: 0;
}
.brand-sub {
    font-size: 0.82rem;
    color: #64748B;
    margin-top: 2px;
}
.user-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #F8FAFC;
    border: 1px solid #CBD5E1;
    border-radius: 999px;
    padding: 5px 12px;
    font-size: 0.82rem;
    color: #0F172A;
}
.user-dot {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background: #2563EB;
    color: #FFFFFF;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.72rem;
}
.hero-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-left: 4px solid #2563EB;
    border-radius: 10px;
    padding: 1.5rem 1.65rem;
    margin-bottom: 1rem;
}
.hero-title {
    font-size: 1.85rem;
    font-weight: 700;
    color: #0F172A;
    line-height: 1.25;
    margin: 0 0 0.5rem 0;
}
.hero-lead {
    font-size: 0.98rem;
    line-height: 1.58;
    color: #334155;
    margin: 0;
}
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 0.85rem 1rem;
    height: 100%;
}
.kpi-val {
    font-size: 1.45rem;
    font-weight: 700;
    color: #0F172A;
}
.kpi-lbl {
    font-size: 0.82rem;
    font-weight: 600;
    color: #2563EB;
    margin-top: 2px;
}
.kpi-sub {
    font-size: 0.75rem;
    color: #64748B;
    margin-top: 3px;
}
.step-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 0.95rem 1rem;
    height: 100%;
}
.step-title {
    font-size: 0.94rem;
    font-weight: 700;
    color: #2563EB;
}
.step-sub {
    font-size: 0.88rem;
    font-weight: 600;
    color: #0F172A;
    margin: 2px 0 6px 0;
}
.step-body {
    font-size: 0.83rem;
    color: #475569;
    line-height: 1.48;
}
.google-box {
    background: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 12px;
    padding: 1.5rem 1.5rem 1.1rem 1.5rem;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
}
.sec-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #0F172A;
    border-bottom: 1px solid #E2E8F0;
    padding-bottom: 0.35rem;
    margin: 1.25rem 0 0.75rem 0;
}
div.stButton > button {
    border-radius: 8px;
    font-weight: 600;
}
</style>
"""
st.markdown(CLEAN_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached Data & Model Loaders
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    d = {k: pd.read_csv(P / f"{k}.csv") for k in ["crash_cells", "ml_observations", "risk_zones", "scenarios"]}
    d["hazards"] = pd.read_csv(P / "hazard_reports.csv", parse_dates=["date"])
    d["obs"] = pd.read_csv(P / "ml_observations.csv", parse_dates=["date"])
    d["crashes"] = pd.read_csv(P / "crashes.csv", parse_dates=["crash_date"])
    return d


@st.cache_data
def load_sim_results():
    return json.loads((R / "results.json").read_text(encoding="utf-8")) if (R / "results.json").exists() else {}


@st.cache_resource
def load_model_bundle():
    model_path = P / "xgb_model.joblib"
    if not model_path.exists():
        return None, None
    bundle = joblib.load(model_path)
    explainer = shap.TreeExplainer(bundle["model"])
    return bundle, explainer


def show_fig(filename: str, caption: str = "", fallback_report_dir: bool = False):
    p = FIG / filename
    if not p.exists() and fallback_report_dir:
        p = R / filename
    if p.exists():
        st.image(str(p), caption=caption, width="stretch")


D = load_data()
SIM_RES = load_sim_results()
DOCX_PATH = R / "RAKSHA_AI_Project_Report_FINAL.docx"
DOCX_BYTES = DOCX_PATH.read_bytes() if DOCX_PATH.exists() else None


# ---------------------------------------------------------------------------
# OAuth 2.0 + OpenID Connect (OIDC) Helpers
# ---------------------------------------------------------------------------
st.session_state.setdefault("auth_user", None)
st.session_state.setdefault("nav_page", "Overview")
st.session_state.setdefault("landing_section", "Overview")


def _oauth_configured() -> bool:
    try:
        return "auth" in st.secrets
    except Exception:
        return False


def _oidc_metadata_url() -> str:
    try:
        return str(st.secrets["auth"]["google"].get("server_metadata_url", ""))
    except Exception:
        return ""


@st.cache_resource
def ensure_local_oidc_provider() -> bool:
    """Ensure the local OpenID Connect IdP (src/oidc_provider.py on port 8502) is running when configured."""
    meta_url = _oidc_metadata_url()
    if "localhost:8502" not in meta_url and "127.0.0.1:8502" not in meta_url:
        return False

    health_url = "http://127.0.0.1:8502/health"
    try:
        with urllib.request.urlopen(health_url, timeout=0.6) as resp:
            if resp.status == 200:
                return True
    except Exception:
        pass

    if os.environ.get("RAKSHA_SKIP_OIDC_SPAWN"):
        return False

    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
    subprocess.Popen(
        [sys.executable, str(ROOT / "src" / "oidc_provider.py")],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )
    for _ in range(25):
        time.sleep(0.2)
        try:
            with urllib.request.urlopen(health_url, timeout=0.5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
    return False


if _oauth_configured():
    ensure_local_oidc_provider()


def _get_active_auth_user() -> dict | None:
    if _oauth_configured():
        try:
            u = getattr(st, "user", None)
            if u is not None and getattr(u, "is_logged_in", False):
                u_dict = dict(u)
                email = str(u_dict.get("email", "") or "")
                name = str(u_dict.get("name", "") or email.split("@")[0] or "OIDC User")
                initials = "".join(w[0].upper() for w in name.split()[:2]) or "O"
                tokens_obj = u_dict.get("tokens")
                tokens = dict(tokens_obj) if tokens_obj is not None else {}
                return {
                    "name": name,
                    "email": email,
                    "initials": initials,
                    "provider": "OAuth 2.0 + OpenID Connect",
                    "claims": {k: v for k, v in u_dict.items() if k != "tokens"},
                    "tokens": tokens,
                }
        except Exception:
            pass
    return st.session_state.get("auth_user")


def sign_out():
    st.session_state["auth_user"] = None
    st.session_state["nav_page"] = "Overview"
    st.session_state["landing_section"] = "Overview"
    if _oauth_configured():
        try:
            u = getattr(st, "user", None)
            if u is not None and getattr(u, "is_logged_in", False):
                st.logout()
                return
        except Exception:
            pass
    st.rerun()


def render_google_auth_card(key_prefix: str = "gauth"):
    _, mid, _ = st.columns([1.0, 1.45, 1.0])
    with mid:
        meta_url = _oidc_metadata_url() or "Not configured"
        is_google_cloud = "accounts.google.com" in meta_url
        provider_label = "Google Identity Platform (accounts.google.com)" if is_google_cloud else "Local OIDC Identity Provider (localhost:8502)"

        st.markdown(
            f"""
            <div class="google-box">
              <div style="display:flex;align-items:center;margin-bottom:6px;">
                {rc.GOOGLE_G_SVG}
                <span style="font-size:1.18rem;font-weight:700;color:#0F172A;">Sign in with OAuth 2.0 &amp; OpenID Connect</span>
              </div>
              <p style="font-size:0.86rem;color:#475569;margin-bottom:0.75rem;">
                Authenticate via Authorization Code Flow with <b>PKCE (S256)</b> and cryptographically verified
                <b>RS256 ID Tokens</b> to access the interactive <b>RAKSHA AI</b> workspace.
              </p>
              <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:0.65rem 0.85rem;font-size:0.79rem;color:#334155;line-height:1.5;">
                <b>Provider:</b> {provider_label}<br/>
                <b>Discovery:</b> <code>{meta_url}</code><br/>
                <b>Scopes:</b> <code>openid email profile</code> &bull; <b>Callback:</b> <code>/oauth2callback</code>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        if _oauth_configured():
            if st.button("Continue with OAuth 2.0 / OpenID Connect", key=f"{key_prefix}_oauth_btn", type="primary", width="stretch"):
                st.login("google")
        else:
            st.error("OAuth 2.0 / OIDC is not configured. Add `[auth]` and `[auth.google]` to `.streamlit/secrets.toml`.")

        with st.expander("Switch between Local OIDC IdP and Google Cloud OAuth"):
            st.markdown(
                """
                Edit `.streamlit/secrets.toml` `[auth.google]` to point to Google's production OpenID Connect endpoint:
                ```toml
                [auth.google]
                client_id = "YOUR_GOOGLE_CLIENT_ID.apps.googleusercontent.com"
                client_secret = "YOUR_GOOGLE_CLIENT_SECRET"
                server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
                ```
                """
            )


# ---------------------------------------------------------------------------
# Top Header & Clean Horizontal Navbar
# ---------------------------------------------------------------------------
AUTH_USER = _get_active_auth_user()

if AUTH_USER:
    chip_html = (
        f'<div class="user-chip">'
        f'  <span class="user-dot">{AUTH_USER["initials"]}</span>'
        f'  <span><b>{AUTH_USER["email"]}</b></span>'
        f'</div>'
    )
else:
    chip_html = '<span style="font-size:0.82rem;color:#64748B;">Research Showcase • Sign in with Google to open live platform</span>'

h_left, h_right = st.columns([4.8, 1.2])
with h_left:
    st.markdown(
        f"""
        <div class="topbar">
          <div>
            <div class="brand-title">🛡️ RAKSHA AI</div>
            <div class="brand-sub">GIS-Based Road Risk Intelligence &amp; Machine Learning Near-Miss Prediction</div>
          </div>
          <div>{chip_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with h_right:
    st.write("")
    if AUTH_USER:
        if st.button("Sign out", key="top_signout", type="secondary", width="stretch"):
            sign_out()
    else:
        if st.button("Sign in with Google", key="top_signin", type="primary", width="stretch"):
            if _oauth_configured():
                st.login("google")
            else:
                st.session_state["landing_section"] = "Sign in with Google"
                st.rerun()

if AUTH_USER:
    NAV_PAGES = [
        "Landing Page",
        "Overview",
        "GIS Map",
        "Model Workbench",
        "Research Hub",
        "Community",
        "Authority Triage",
        "Analytics",
    ]
    cols = st.columns(len(NAV_PAGES))
    for i, pname in enumerate(NAV_PAGES):
        active = st.session_state["nav_page"] == pname
        if cols[i].button(pname, key=f"nav_{i}", type="primary" if active else "secondary", width="stretch"):
            st.session_state["nav_page"] = pname
            st.rerun()
else:
    LANDING_TABS = [
        "Overview",
        "How It Works & Architecture",
        "ML Benchmarks",
        "GIS Coverage",
        "Sign in with Google",
    ]
    cols = st.columns(len(LANDING_TABS))
    for i, sname in enumerate(LANDING_TABS):
        active = st.session_state["landing_section"] == sname
        if cols[i].button(sname, key=f"lnav_{i}", type="primary" if active else "secondary", width="stretch"):
            st.session_state["landing_section"] = sname
            st.rerun()


# ---------------------------------------------------------------------------
# Hardcoded Research Landing Page Sections
# ---------------------------------------------------------------------------
def render_landing_hero_and_kpis(is_authenticated: bool):
    c1, c2 = st.columns([1.65, 1.0])
    with c1:
        st.markdown(
            """
            <div class="hero-card">
              <div class="hero-title">GIS Road Risk Intelligence &amp; ML Near-Miss Prediction</div>
              <p class="hero-lead">
                Conventional crash maps are retrospective—a road segment appears dangerous only after injuries or
                fatalities accumulate. <b>RAKSHA AI</b> combines retrospective 150 m GIS crash-blackspot mapping
                (<b>217,160 crash records</b> across New York, Chicago, Great Britain, and 8 Indian demo cities) with
                a causal machine learning model trained on <b>304,988 real Argoverse 2 road-user pairs</b> to
                predict near-miss onset up to <b>3 seconds early</b> (<b>0.966 ROC-AUC</b> vs <b>0.745</b> for a TTC-only rule).
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        b1, b2, b3 = st.columns(3)
        if is_authenticated:
            if b1.button("Open GIS Map", key="lp_go_gis", type="primary", width="stretch"):
                st.session_state["nav_page"] = "GIS Map"
                st.rerun()
            if b2.button("Open Model Workbench", key="lp_go_wb", type="secondary", width="stretch"):
                st.session_state["nav_page"] = "Model Workbench"
                st.rerun()
        else:
            if b1.button("Sign in with Google", key="lp_go_auth", type="primary", width="stretch"):
                st.session_state["landing_section"] = "Sign in with Google"
                st.rerun()
            if b2.button("View ML Benchmarks", key="lp_go_ml", type="secondary", width="stretch"):
                st.session_state["landing_section"] = "ML Benchmarks"
                st.rerun()
        if DOCX_BYTES:
            b3.download_button(
                "Download Report (.docx)",
                data=DOCX_BYTES,
                file_name="RAKSHA_AI_Project_Report_FINAL.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="lp_dl_docx",
                width="stretch",
            )
    with c2:
        show_fig("fig4_1_architecture.jpeg", "Layered architecture of RAKSHA AI")

    kcols = st.columns(len(rc.HEADLINE_METRICS))
    for col, (val, lbl, sub) in zip(kcols, rc.HEADLINE_METRICS):
        col.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-val">{val}</div>
              <div class="kpi-lbl">{lbl}</div>
              <div class="kpi-sub">{sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_landing_workflow_and_architecture():
    st.markdown('<div class="sec-title">How RAKSHA AI Works — Four-Step Workflow</div>', unsafe_allow_html=True)
    scols = st.columns(4)
    for col, (title, sub, body) in zip(scols, rc.STORY_STEPS):
        col.markdown(
            f"""
            <div class="step-card">
              <div class="step-title">{title}</div>
              <div class="step-sub">{sub}</div>
              <div class="step-body">{body}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown("**Dataset Inventory & Honest Provenance**")
    st.dataframe(pd.DataFrame(rc.DATASET_PROVENANCE), width="stretch", hide_index=True)

    st.markdown('<div class="sec-title">System Architecture, Causal Formulation & Leakage Control</div>', unsafe_allow_html=True)
    a1, a2 = st.columns(2)
    with a1:
        show_fig("fig4_2_dataflow.jpeg", "Near-miss prediction pipeline data flow")
    with a2:
        show_fig("fig4_3_anchor_window.jpeg", "Causal 1.5 s history window, 0.5 s anchor t, and 3 s look-ahead window")

    b1, b2 = st.columns([1.05, 1.0])
    with b1:
        show_fig("fig5_1_crossing_conflict.jpeg", "Crossing conflict trajectory, clearance, TTC and projected PET")
    with b2:
        st.markdown("**Mathematical Definitions**")
        st.markdown(
            """
            - **Outcome-Based Near-Miss:** First frame where boundary clearance $c(\\tau) < 1.0\\text{ m}$
              *(on Argoverse 2, also requires relative speed $> 1.5\\text{ m/s}$ and swept-path overlap to remove queued/adjacent passes)*.
            - **Closing Velocity:** $v_c = \\frac{(\\mathbf{p}_B - \\mathbf{p}_A) \\cdot (\\mathbf{v}_A - \\mathbf{v}_B)}{\\|\\mathbf{p}_B - \\mathbf{p}_A\\|}$ (approaching when $v_c > 0$).
            - **TTC (30 s cap):** earliest contact time assuming constant velocities.
            - **Projected PET:** arrival-time gap at path intersection (crossing) or headway (following).
            - **DRAC:** $\\text{DRAC} = \\frac{v_c^2}{2 \\cdot c}$ when $v_c > 0$, else $0$.
            - **GIS Grid Severity Score:** $\\text{Score} = \\sum (1 + 2 \\cdot \\text{injured} + 10 \\cdot \\text{killed})$ per $0.0015^\\circ$ (~150 m) cell.
            """
        )
        st.markdown("**Anti-Leakage Guarantees**")
        for item in rc.LEAKAGE_GUARANTEES:
            st.markdown(f"- {item}")

    with st.expander("27 Model Features, Argoverse 2 Class Dimensions & Functional Requirements"):
        t1, t2, t3 = st.tabs(["27 Features", "AV2 Class Sizes", "Functional Requirements"])
        with t1:
            st.dataframe(pd.DataFrame(rc.FEATURE_DICTIONARY), width="stretch", hide_index=True)
        with t2:
            st.dataframe(pd.DataFrame(rc.AV2_CLASSES), width="stretch", hide_index=True)
        with t3:
            st.dataframe(pd.DataFrame(rc.FUNCTIONAL_REQS), width="stretch", hide_index=True)


def render_landing_ml_benchmarks():
    st.markdown('<div class="sec-title">ML Benchmarks — Argoverse 2 Real Traffic vs Simulated Mixed Traffic</div>', unsafe_allow_html=True)
    st.info(
        "**Key Finding:** On **1,956,954 causal anchors across 304,988 real Argoverse 2 road-user pairs** "
        "(0.12% positive rate, 652 test positives), **XGBoost reaches 0.966 ROC-AUC and 0.067 AP (57x random)** "
        "versus **0.745 AUC and 0.016 AP** for a TTC threshold rule."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Argoverse 2 Real Traffic (Test Set)**")
        st.dataframe(pd.DataFrame(rc.AV2_LEADERBOARD), width="stretch", hide_index=True)
        show_fig("fig6_2_av2_auc.png", "Argoverse 2: TTC-only vs full-feature models")
    with c2:
        st.markdown("**Simulated Mixed Traffic (6,000 Scenarios, Test Set)**")
        st.dataframe(pd.DataFrame(rc.SIM_LEADERBOARD), width="stretch", hide_index=True)
        show_fig("fig6_1_sim_auc.png", "Simulated traffic: TTC-only vs full-feature models")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**XGBoost Confusion Matrix on Argoverse 2 (Tuned Threshold)**")
        st.dataframe(pd.DataFrame(rc.AV2_CONFUSION), width="stretch", hide_index=True)
        show_fig("fig6_3_av2_confusion.png", "Argoverse 2 confusion matrices")
    with c4:
        st.markdown("**Lead-Time AUC Before Near-Miss Onset (Real Traffic)**")
        st.dataframe(pd.DataFrame(rc.LEAD_TIME_ONSET), width="stretch", hide_index=True)
        show_fig("fig6_4_av2_leadtime.jpeg", "Look-ahead window L (left) and time to onset (right)")

    c5, c6 = st.columns(2)
    with c5:
        st.markdown("**Feature-Group Ablation (XGBoost)**")
        st.dataframe(pd.DataFrame(rc.ABLATION_TABLE), width="stretch", hide_index=True)
        show_fig("fig6_5_av2_ablation.png", "Incremental gain from TTC to full feature set")
    with c6:
        st.markdown("**Top 8 SHAP Feature Attributions (Mean |SHAP|)**")
        st.dataframe(pd.DataFrame(rc.SHAP_TOP8), width="stretch", hide_index=True)
        show_fig("fig6_6_av2_shap.jpeg", "Argoverse 2 TreeSHAP beeswarm summary")

    p1, p2 = st.columns([1.0, 1.1])
    with p1:
        st.markdown("**Near-Miss Rate by Road-User Pair (Argoverse 2)**")
        st.dataframe(pd.DataFrame(rc.PAIR_RATES_AV2), width="stretch", hide_index=True)
    with p2:
        show_fig("fig6_14_av2_pairs.png", "Near-miss rate across road-user combinations")


def render_landing_gis_and_scope():
    st.markdown('<div class="sec-title">Multi-Country GIS Coverage, Blackspots & Emerging Risk Zones</div>', unsafe_allow_html=True)
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**Crash Sources Harmonised**")
        st.dataframe(pd.DataFrame(rc.CRASH_SOURCES_TABLE), width="stretch", hide_index=True)
    with g2:
        st.markdown("**Highest-Scoring Real Blackspots**")
        st.dataframe(pd.DataFrame(rc.TOP_REAL_BLACKSPOTS), width="stretch", hide_index=True)

    st.markdown("**150 m Grid Cells & Blackspots Across 11 Cities**")
    st.dataframe(pd.DataFrame(rc.CITY_GRID_SUMMARY), width="stretch", hide_index=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        show_fig("fig6_7_map_nyc.jpeg", "New York: 98 blackspots & emerging zones")
    with m2:
        show_fig("fig6_8_map_chicago.jpeg", "Chicago: 77 blackspots & emerging zones")
    with m3:
        show_fig("fig6_9_map_india.jpeg", "India demo: OSM junctions + simulated crash counts")

    st.markdown('<div class="sec-title">Documented Limitations & Future Scope</div>', unsafe_allow_html=True)
    lcol, fcol = st.columns(2)
    with lcol:
        st.markdown("**Limitations**")
        for title, desc in rc.LIMITATIONS:
            st.markdown(f"- **{title}:** {desc}")
    with fcol:
        st.markdown("**Future Scope**")
        for title, desc in rc.FUTURE_SCOPE:
            st.markdown(f"- **{title}:** {desc}")


# ---------------------------------------------------------------------------
# Sidebar (Clean Auth Status & Provenance)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛡️ RAKSHA AI")
    if AUTH_USER:
        st.success(f"Signed in: **{AUTH_USER['email']}**")
        if st.button("Sign out", key="sb_signout", width="stretch"):
            sign_out()
    else:
        st.info("Sign in via OAuth 2.0 / OpenID Connect to access the interactive platform modules.")
        if st.button("Sign in with Google", key="sb_signin", type="primary", width="stretch"):
            if _oauth_configured():
                st.login("google")
            else:
                st.session_state["landing_section"] = "Sign in with Google"
                st.rerun()

    st.caption(
        "**Data provenance:** Real crash records (NYC, Chicago, UK STATS19) and real Argoverse 2 trajectories. "
        "Indian crash counts, simulated traffic trajectories, map alerts, emerging zones, and seed reports are simulated."
    )


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
if not AUTH_USER:
    sec = st.session_state["landing_section"]
    if sec == "Overview":
        render_landing_hero_and_kpis(is_authenticated=False)
        render_landing_workflow_and_architecture()
        render_landing_ml_benchmarks()
        render_landing_gis_and_scope()
        st.markdown("---")
        render_google_auth_card(key_prefix="bot_auth")
    elif sec == "How It Works & Architecture":
        render_landing_workflow_and_architecture()
    elif sec == "ML Benchmarks":
        render_landing_ml_benchmarks()
    elif sec == "GIS Coverage":
        render_landing_gis_and_scope()
    else:
        render_google_auth_card(key_prefix="main_auth")

else:
    page = st.session_state["nav_page"]

    # ====================================================================
    # 1. LANDING PAGE (authenticated view)
    # ====================================================================
    if page == "Landing Page":
        render_landing_hero_and_kpis(is_authenticated=True)
        render_landing_workflow_and_architecture()
        render_landing_ml_benchmarks()
        render_landing_gis_and_scope()

    # ====================================================================
    # 2. OVERVIEW (live workspace metrics)
    # ====================================================================
    elif page == "Overview":
        st.title("RAKSHA AI — Platform Overview")
        st.caption("Live workspace metrics combining multi-country GIS crash records with near-miss ML predictions.")

        cells = D["crash_cells"]
        cr = D["crashes"]
        c = st.columns(5)
        c[0].metric("Crash records", f"{len(cr):,}", f"{int(cr.real.sum()):,} real")
        c[1].metric("Blackspots", int(cells.blackspot.sum()), f"{cells.city.nunique()} places")
        c[2].metric("Emerging zones", int(D["risk_zones"].emerging.sum()))
        c[3].metric("ML observations", len(D["obs"]))
        c[4].metric("Open hazard reports", int(D["hazards"].status.isin(["Submitted", "Under review"]).sum()))

        a, b = st.columns(2)
        a.metric("TTC Threshold Rule (Argoverse 2 Real)", "0.745 AUC", "0.016 AP")
        b.metric("XGBoost Full Model (Argoverse 2 Real)", "0.966 AUC", "+0.221 over TTC rule")

        st.subheader("City Coverage")
        cov = cr.groupby(["country", "city"]).agg(
            crashes=("collision_id", "count"),
            injured=("injured", "sum"),
            killed=("killed", "sum"),
            real_data=("real", "first"),
        ).reset_index()
        cov["blackspots"] = cov.city.map(cells.groupby("city").blackspot.sum())
        st.dataframe(cov, width="stretch", hide_index=True)

        q1, q2, q3, q4 = st.columns(4)
        if q1.button("Open GIS Map", key="ov_gis", type="primary", width="stretch"):
            st.session_state["nav_page"] = "GIS Map"; st.rerun()
        if q2.button("Open Model Workbench", key="ov_wb", width="stretch"):
            st.session_state["nav_page"] = "Model Workbench"; st.rerun()
        if q3.button("Report a Hazard", key="ov_rep", width="stretch"):
            st.session_state["nav_page"] = "Community"; st.rerun()
        if q4.button("Authority Triage", key="ov_tri", width="stretch"):
            st.session_state["nav_page"] = "Authority Triage"; st.rerun()

        if AUTH_USER.get("claims") or AUTH_USER.get("tokens"):
            with st.expander("OpenID Connect (OIDC) Session & Token Claims"):
                st.markdown(f"**Provider:** `{AUTH_USER.get('provider', 'OAuth 2.0 + OIDC')}`")
                if AUTH_USER.get("claims"):
                    st.markdown("**Verified ID Token Claims (`st.user`):**")
                    st.json(AUTH_USER["claims"])
                toks = AUTH_USER.get("tokens") or {}
                if toks:
                    id_t = str(toks.get("id", "") or "")
                    acc_t = str(toks.get("access", "") or "")
                    st.markdown("**Issued Tokens:**")
                    st.code(
                        f"id_token (RS256 JWT): {id_t[:48]}...{id_t[-24:] if len(id_t) > 72 else ''}\n"
                        f"access_token (Bearer): {acc_t[:28]}...",
                        language="text",
                    )

    # ====================================================================
    # 3. GIS MAP
    # ====================================================================
    elif page == "GIS Map":
        st.title("GIS Risk Map")
        side = st.sidebar
        side.markdown("---")
        side.markdown("#### Map Filters")
        places = D["crash_cells"].groupby(["country", "city"]).agg(lat=("lat", "median"), lon=("lon", "median")).reset_index()
        pl = side.selectbox("Place", ["All places"] + (places.country + " - " + places.city).tolist(), index=0)
        layers = side.multiselect(
            "Layers",
            ["Blackspots (crash history)", "Emerging risk zones", "ML risk layer (unvalidated)", "Verified hazard reports", "Heatmap"],
            default=["Blackspots (crash history)", "Emerging risk zones", "Heatmap"],
        )
        sev = side.multiselect("Hazard severity", ["Low", "Medium", "High"], default=["Low", "Medium", "High"])
        cat = side.multiselect("Hazard category", sorted(D["hazards"].category.unique()), default=list(D["hazards"].category.unique()))
        vt = side.multiselect("Vehicle type (ML obs, vehicle A)", sorted(D["obs"].type_a.unique()), default=list(D["obs"].type_a.unique()))
        dmin, dmax = D["obs"].date.min().date(), D["obs"].date.max().date()
        dr = side.slider("Date range", dmin, dmax, (dmin, dmax))
        show_status = side.multiselect("ML observation status", ["verified", "unverified"], default=["verified"])

        obs = D["obs"][D["obs"].type_a.isin(vt) & D["obs"].date.dt.date.between(*dr) & D["obs"].status.isin(show_status)]
        cells_v = D["crash_cells"]
        zones_v = D["risk_zones"]
        hz_v = D["hazards"]
        if pl == "All places":
            loc, zoom, sel = [28.0, 10.0], 2, None
        else:
            row = places[(places.country + " - " + places.city) == pl].iloc[0]
            loc, zoom, sel = [row.lat, row.lon], (6 if row.city == "Great Britain" else 11), row.city
            obs = obs[obs.city == sel]
            cells_v = cells_v[cells_v.city == sel]
            zones_v = zones_v[zones_v.city == sel]
            hz_v = hz_v[hz_v.city == sel]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Blackspots shown", int(cells_v.blackspot.sum()))
        m2.metric("Emerging zones shown", int(zones_v.emerging.sum()))
        m3.metric("ML alerts matching filters", len(obs))
        m4.metric("Verified hazard reports", int((hz_v.status == "Verified").sum()))

        m = folium.Map(location=loc, zoom_start=zoom, tiles="OpenStreetMap")
        if "Heatmap" in layers:
            hv = obs[obs.status == "verified"] if "verified" in show_status else obs.iloc[0:0]
            if len(hv):
                HeatMap(
                    hv[["lat", "lon"]].values.tolist(),
                    radius=18, blur=22, min_opacity=0.3,
                    gradient={0.3: "yellow", 0.6: "orange", 1.0: "red"},
                ).add_to(m)
        if "Blackspots (crash history)" in layers:
            for _, r in cells_v[cells_v.blackspot].iterrows():
                tag = "" if r.real else " [SIMULATED counts]"
                folium.CircleMarker(
                    [r.lat, r.lon],
                    radius=4 + r.crashes ** 0.5 / (2 if sel is None else 1),
                    color="#7b0000", fill=True, fill_opacity=0.72,
                    tooltip=f"{r.city}: {r.crashes} crashes, {int(r.injured)} injured, {int(r.killed)} killed{tag}",
                ).add_to(m)
        if "Emerging risk zones" in layers:
            for _, r in zones_v[zones_v.emerging].iterrows():
                folium.Circle(
                    [r.lat, r.lon],
                    radius=140, color="#e67e22", fill=True, fill_opacity=0.25, weight=2,
                    tooltip=f"Emerging zone ({r.city}): {r.obs} observations, mean risk {r.mean_p:.2f} "
                            f"(historic crashes: {int(r.crashes) if pd.notna(r.crashes) else 0})",
                ).add_to(m)
        if "ML risk layer (unvalidated)" in layers:
            for _, r in obs.head(600).iterrows():
                folium.CircleMarker(
                    [r.lat, r.lon],
                    radius=3, color="#2980b9", fill=True,
                    tooltip=f"{r.type_a} vs {r.type_b} p={r.p:.2f} [{r.status}] (simulated)",
                ).add_to(m)
        if "Verified hazard reports" in layers:
            h = hz_v[hz_v.status.isin(["Verified"]) & hz_v.severity.isin(sev) & hz_v.category.isin(cat)]
            for _, r in h.iterrows():
                folium.Marker(
                    [r.lat, r.lon],
                    icon=folium.Icon(color="green", icon="warning-sign"),
                    tooltip=f"{r.city}: {r.category} ({r.severity})",
                ).add_to(m)

        st_folium(m, height=580, use_container_width=True, returned_objects=[])
        st.caption("A dense heatmap indicates where to investigate—it does not by itself establish an official blackspot.")

        st.subheader("Top Places to Inspect")
        cz = zones_v.sort_values("obs", ascending=False).head(12)[["city", "lat", "lon", "obs", "mean_p", "crashes", "blackspot", "emerging"]]
        st.dataframe(cz, width="stretch", hide_index=True)

        with st.expander("Blackspot Reference Tables (City Grid Summary & Top Blackspots)"):
            t1, t2 = st.tabs(["Top Real Blackspots", "City Grid Summary"])
            with t1:
                st.dataframe(pd.DataFrame(rc.TOP_REAL_BLACKSPOTS), width="stretch", hide_index=True)
            with t2:
                st.dataframe(pd.DataFrame(rc.CITY_GRID_SUMMARY), width="stretch", hide_index=True)

    # ====================================================================
    # 4. MODEL WORKBENCH
    # ====================================================================
    elif page == "Model Workbench":
        st.title("Model Workbench & Near-Miss Prediction")
        ds_choice = st.radio(
            "Dataset",
            ["Argoverse 2 Real Traffic (4,988 scenarios • 304,988 pairs)", "Simulated Mixed Traffic (6,000 scenarios)"],
            horizontal=True,
        )
        is_real = ds_choice.startswith("Argoverse 2")

        tabs = st.tabs([
            "Comparison vs TTC Rule",
            "Lead-Time & Look-Ahead",
            "Ablation & SHAP",
            "Vehicle Pairs",
            "Live Conflict Simulator & CSV Scorer",
        ])

        with tabs[0]:
            if is_real:
                c1, c2 = st.columns(2)
                with c1:
                    st.dataframe(pd.DataFrame(rc.AV2_LEADERBOARD), width="stretch", hide_index=True)
                    st.dataframe(pd.DataFrame(rc.AV2_CONFUSION), width="stretch", hide_index=True)
                with c2:
                    show_fig("fig6_2_av2_auc.png", "Argoverse 2 ROC-AUC comparison")
                    show_fig("fig6_3_av2_confusion.png", "Argoverse 2 confusion matrices")
            else:
                c1, c2 = st.columns(2)
                with c1:
                    if SIM_RES:
                        st.markdown("**Workspace Simulated Run**")
                        st.dataframe(pd.DataFrame(SIM_RES["exp1"]).T.round(3), width="stretch")
                    st.markdown("**3 s Look-Ahead Simulated Benchmark**")
                    st.dataframe(pd.DataFrame(rc.SIM_LEADERBOARD), width="stretch", hide_index=True)
                with c2:
                    show_fig("exp1.png", "Simulated ROC-AUC", fallback_report_dir=True)
                    show_fig("confusion.png", "Simulated confusion matrices", fallback_report_dir=True)

        with tabs[1]:
            if is_real:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**AUC by Time Remaining Until Onset**")
                    st.dataframe(pd.DataFrame(rc.LEAD_TIME_ONSET), width="stretch", hide_index=True)
                    st.markdown("**AUC by Look-Ahead Window L (1–5 s)**")
                    st.dataframe(pd.DataFrame(rc.LOOKAHEAD_WINDOW), width="stretch", hide_index=True)
                with c2:
                    show_fig("fig6_4_av2_leadtime.jpeg", "Look-ahead window (left) and time to onset (right)")
            else:
                c1, c2 = st.columns(2)
                with c1:
                    if SIM_RES:
                        df_e3 = pd.DataFrame(SIM_RES["exp3"]).T.reset_index().rename(columns={"index": "Horizon (s)"})
                        st.dataframe(df_e3, width="stretch", hide_index=True)
                        ch = alt.Chart(df_e3.melt("Horizon (s)", var_name="Model", value_name="AUC")).mark_line(point=True).encode(
                            x="Horizon (s):O", y=alt.Y("AUC:Q", scale=alt.Scale(domain=[0.7, 1.01])), color="Model:N"
                        )
                        st.altair_chart(ch, width="stretch")
                with c2:
                    show_fig("figC_1_sim_leadtime.jpeg", "Simulated lead-time and look-ahead analysis")

        with tabs[2]:
            if is_real:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Feature-Group Ablation**")
                    st.dataframe(pd.DataFrame(rc.ABLATION_TABLE), width="stretch", hide_index=True)
                    show_fig("fig6_5_av2_ablation.png", "Argoverse 2 feature ablation")
                with c2:
                    st.markdown("**Top 8 SHAP Features**")
                    st.dataframe(pd.DataFrame(rc.SHAP_TOP8), width="stretch", hide_index=True)
                    show_fig("fig6_6_av2_shap.jpeg", "Argoverse 2 TreeSHAP summary")
            else:
                c1, c2 = st.columns(2)
                with c1:
                    if SIM_RES:
                        st.dataframe(pd.DataFrame(SIM_RES["exp2"]).T.round(3), width="stretch")
                        st.dataframe(pd.Series(SIM_RES["top_shap"], name="Mean |SHAP|").to_frame(), width="stretch")
                with c2:
                    show_fig("shap.png", "Simulated TreeSHAP summary", fallback_report_dir=True)

        with tabs[3]:
            if is_real:
                c1, c2 = st.columns(2)
                with c1:
                    st.dataframe(pd.DataFrame(rc.PAIR_RATES_AV2), width="stretch", hide_index=True)
                with c2:
                    show_fig("fig6_14_av2_pairs.png", "Argoverse 2 near-miss rate by pair")
            else:
                c1, c2 = st.columns(2)
                with c1:
                    if SIM_RES:
                        st.dataframe(pd.DataFrame(SIM_RES["exp4"]).T, width="stretch")
                with c2:
                    show_fig("exp4.png", "Simulated near-miss rate by pair", fallback_report_dir=True)

        with tabs[4]:
            bundle, explainer = load_model_bundle()
            if bundle is None:
                st.error("Model bundle `data/processed/xgb_model.joblib` not found.")
            else:
                st.subheader("Live Near-Miss Conflict Simulator & TreeSHAP Explainer")
                preset_name = st.selectbox("Load Conflict Preset", list(rc.CONFLICT_PRESETS.keys()))
                pr = rc.CONFLICT_PRESETS[preset_name]
                veh_a_opts = ["car", "motorcycle", "bus", "truck"]
                veh_b_opts = ["car", "motorcycle", "bus", "truck", "pedestrian"]

                sc1, sc2, sc3, sc4 = st.columns(4)
                type_a = sc1.selectbox("Road User A", veh_a_opts, index=veh_a_opts.index(pr["type_a"]), key=f"sim_ta_{preset_name}")
                type_b = sc2.selectbox("Road User B", veh_b_opts, index=veh_b_opts.index(pr["type_b"]), key=f"sim_tb_{preset_name}")
                crossing = sc3.selectbox("Geometry", ["Crossing (1)", "Following (0)"], index=0 if pr["crossing"] else 1, key=f"sim_cr_{preset_name}")
                clr_last = sc4.slider("Clearance clr_last (m)", 0.1, 30.0, float(pr["clr_last"]), 0.1, key=f"sim_clr_{preset_name}")

                sc5, sc6, sc7, sc8 = st.columns(4)
                ttc_last = sc5.slider("TTC at anchor (s)", 0.1, 30.0, float(pr["ttc_last"]), 0.1, key=f"sim_ttc_{preset_name}")
                ttc_min = sc6.slider("Min TTC (s)", 0.1, 30.0, float(pr["ttc_min"]), 0.1, key=f"sim_ttcm_{preset_name}")
                drac_last = sc7.slider("DRAC (m/s²)", 0.0, 12.0, float(pr["drac_last"]), 0.1, key=f"sim_drac_{preset_name}")
                pet_last = sc8.slider("Projected PET (s)", 0.1, 10.0, float(pr["pet_last"]), 0.1, key=f"sim_pet_{preset_name}")

                sc9, sc10, sc11, sc12 = st.columns(4)
                closing_last = sc9.slider("Closing velocity (m/s)", -5.0, 20.0, float(pr["closing_last"]), 0.2, key=f"sim_clos_{preset_name}")
                speed_a = sc10.slider("Speed A (m/s)", 0.0, 25.0, float(pr["speed_a"]), 0.2, key=f"sim_sa_{preset_name}")
                speed_b = sc11.slider("Speed B (m/s)", 0.0, 25.0, float(pr["speed_b"]), 0.2, key=f"sim_sb_{preset_name}")
                acc_a = sc12.slider("Accel A (m/s²)", -8.0, 4.0, float(pr["acc_a"]), 0.1, key=f"sim_acca_{preset_name}")

                row_dict = {f: 0.0 for f in bundle["features"]}
                row_dict.update({
                    "ttc_min": ttc_min,
                    "ttc_last": ttc_last,
                    "pet_min": max(0.1, pet_last - 0.1),
                    "pet_last": pet_last,
                    "drac_max": round(drac_last * 1.1, 2),
                    "drac_last": drac_last,
                    "dist_last": clr_last + 3.0,
                    "clr_last": clr_last,
                    "closing_last": closing_last,
                    "speed_a": speed_a,
                    "speed_b": speed_b,
                    "rel_speed": abs(speed_a - speed_b),
                    "acc_a": acc_a,
                    "acc_b": float(pr["acc_b"]),
                    "acc_a_min": min(acc_a, acc_a - 0.5),
                    "acc_b_min": float(pr["acc_b_min"]),
                    "jerk_a": float(pr["jerk_a"]),
                    "crossing": 1 if "1" in crossing else 0,
                    f"ta_{type_a}": 1,
                    f"tb_{type_b}": 1,
                })
                X_one = pd.DataFrame([row_dict])[bundle["features"]]
                xgb_prob = float(bundle["model"].predict_proba(X_one)[0, 1])
                ttc_rule_score = float(1.0 - ttc_last / 30.0)

                r1, r2, r3 = st.columns(3)
                r1.metric("XGBoost Near-Miss Probability", f"{xgb_prob*100:.1f}%", "Alert" if xgb_prob >= 0.5 else "Low Risk")
                r2.metric("TTC Rule Score", f"{ttc_rule_score*100:.1f}%", f"TTC = {ttc_last:.1f} s")
                r3.metric("Recommended Action", "Escalate to GIS Cell" if xgb_prob >= 0.5 else "Routine Monitoring")

                sv_one = explainer.shap_values(X_one)
                sv_arr = sv_one[0] if isinstance(sv_one, np.ndarray) else sv_one
                shap_df = pd.DataFrame({"Feature": bundle["features"], "SHAP Contribution (log-odds)": sv_arr})
                shap_df["abs"] = shap_df["SHAP Contribution (log-odds)"].abs()
                top10 = shap_df.sort_values("abs", ascending=False).head(10)
                shap_chart = alt.Chart(top10).mark_bar().encode(
                    x=alt.X("SHAP Contribution (log-odds):Q"),
                    y=alt.Y("Feature:N", sort="-x"),
                    color=alt.condition(
                        alt.datum["SHAP Contribution (log-odds)"] > 0,
                        alt.value("#2563EB"),
                        alt.value("#94A3B8"),
                    ),
                    tooltip=["Feature", "SHAP Contribution (log-odds)"],
                ).properties(height=270, title="Live TreeSHAP Explanation")
                st.altair_chart(shap_chart, width="stretch")

                st.markdown("---")
                st.subheader("Batch CSV Scorer")
                sample_df = pd.concat([X_one, X_one.assign(clr_last=18.0, ttc_last=28.0, ttc_min=24.0, drac_last=0.0, drac_max=0.1)], ignore_index=True)
                st.download_button(
                    "Download 2-Row Sample CSV Template",
                    data=sample_df.to_csv(index=False),
                    file_name="raksha_sample_features.csv",
                    mime="text/csv",
                )
                up = st.file_uploader("Upload CSV with the 27 model features", type="csv")
                if up:
                    df_up = pd.read_csv(up)
                    miss = [c for c in bundle["features"] if c not in df_up]
                    if miss:
                        st.error(f"Missing columns: {miss}")
                    else:
                        df_up["risk_probability"] = bundle["model"].predict_proba(df_up[bundle["features"]])[:, 1]
                        st.dataframe(df_up, width="stretch")
                        st.download_button("Download Predictions", df_up.to_csv(index=False), "predictions.csv")

    # ====================================================================
    # 5. RESEARCH HUB
    # ====================================================================
    elif page == "Research Hub":
        st.title("Research Hub — Methodology & Documentation")
        render_landing_workflow_and_architecture()
        render_landing_gis_and_scope()
        if DOCX_BYTES:
            st.download_button(
                "Download Full Report (.docx)",
                data=DOCX_BYTES,
                file_name="RAKSHA_AI_Project_Report_FINAL.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="rh_dl_docx",
                type="primary",
            )

    # ====================================================================
    # 6. COMMUNITY HAZARD REPORTING
    # ====================================================================
    elif page == "Community":
        st.title("Community Hazard Reporting")
        st.caption(f"Signed in as **{AUTH_USER['email']}**. Reports require authority verification before affecting risk layers.")

        city_coords = (
            D["crash_cells"].groupby(["country", "city"])
            .agg(lat=("lat", "median"), lon=("lon", "median"))
            .reset_index()
        )
        place_labels = (city_coords["country"] + " - " + city_coords["city"]).tolist()
        sel_place = st.selectbox("City / Region", place_labels, index=0)
        sel_row = city_coords[(city_coords["country"] + " - " + city_coords["city"]) == sel_place].iloc[0]

        with st.form("hazard_report_form"):
            c1, c2 = st.columns(2)
            cat = c1.selectbox("Category", ["Pothole", "Damaged road", "Dangerous intersection", "Missing signage"])
            sev = c2.select_slider("Severity", ["Low", "Medium", "High"], value="Medium")
            c3, c4 = st.columns(2)
            lat = c3.number_input("Latitude", value=float(round(sel_row.lat, 5)), format="%.5f", key=f"rep_lat_{sel_place}")
            lon = c4.number_input("Longitude", value=float(round(sel_row.lon, 5)), format="%.5f", key=f"rep_lon_{sel_place}")
            desc = st.text_area("Description")
            img = st.file_uploader("Photo (optional)", type=["jpg", "jpeg", "png"])
            if st.form_submit_button("Submit Report", type="primary"):
                path = P / "hazard_reports.csv"
                h = pd.read_csv(path)
                new_id = int(h["id"].max()) + 1 if len(h) else 1
                if img:
                    (P / "uploads").mkdir(parents=True, exist_ok=True)
                    (P / "uploads" / f"{new_id}_{img.name}").write_bytes(img.read())
                new_row = {
                    "city": sel_row.city,
                    "country": sel_row.country,
                    "id": new_id,
                    "category": cat,
                    "lat": lat,
                    "lon": lon,
                    "severity": sev,
                    "status": "Submitted",
                    "date": pd.Timestamp.now().normalize().strftime("%Y-%m-%d"),
                    "simulated": False,
                }
                pd.concat([h, pd.DataFrame([new_row])], ignore_index=True).to_csv(path, index=False)
                load_data.clear()
                st.success(f"Report #{new_id} submitted for {sel_row.city}, {sel_row.country}.")

        st.subheader("Recent Submissions")
        st.dataframe(D["hazards"].sort_values("id", ascending=False).head(15), width="stretch", hide_index=True)

    # ====================================================================
    # 7. AUTHORITY TRIAGE
    # ====================================================================
    elif page == "Authority Triage":
        st.title("Authority Triage")
        st.caption(f"Reviewer: **{AUTH_USER['email']}**")

        t1, t2 = st.tabs(["Hazard Reports", "Emerging Risk Zones"])
        with t1:
            path = P / "hazard_reports.csv"
            h = pd.read_csv(path)
            f1, f2 = st.columns(2)
            flt_status = f1.multiselect("Status", sorted(h.status.unique()), default=[s for s in ["Submitted", "Under review"] if s in h.status.unique()])
            flt_city = f2.multiselect("City", sorted(h.city.dropna().unique()), default=[])
            q = h[h.status.isin(flt_status)] if flt_status else h
            if flt_city:
                q = q[q.city.isin(flt_city)]
            st.dataframe(q.sort_values("id", ascending=False), width="stretch", hide_index=True)

            c1, c2, c3 = st.columns([1, 1.3, 1])
            rid = int(c1.number_input("Report ID", int(h.id.min()), int(h.id.max()), int(q.id.iloc[0]) if len(q) else int(h.id.min())))
            new_status = c2.selectbox("New status", ["Under review", "Verified", "Resolved", "Rejected"])
            c3.write("")
            if c3.button("Apply", type="primary", width="stretch"):
                h.loc[h.id == rid, "status"] = new_status
                h.to_csv(path, index=False)
                load_data.clear()
                st.rerun()

            upload_dir = P / "uploads"
            if upload_dir.exists():
                matches = list(upload_dir.glob(f"{rid}_*"))
                if matches:
                    st.image(str(matches[0]), caption=f"Photo for Report #{rid}", width=340)

        with t2:
            z = D["risk_zones"][D["risk_zones"].emerging].sort_values("obs", ascending=False)
            st.dataframe(z[["city", "lat", "lon", "obs", "mean_p", "crashes"]], width="stretch", hide_index=True)

    # ====================================================================
    # 8. ANALYTICS
    # ====================================================================
    else:
        st.title("Crash & Near-Miss Analytics")
        c = D["crashes"]
        cities = sorted(c.city.unique())
        sel_cities = st.multiselect("Cities", cities, default=cities)
        cf = c[c.city.isin(sel_cities)] if sel_cities else c

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Crash Records", f"{len(cf):,}")
        k2.metric("Real Records", f"{int(cf.real.sum()):,}")
        k3.metric("Injured", f"{int(cf.injured.sum()):,}")
        k4.metric("Fatalities", f"{int(cf.killed.sum()):,}")

        a, b = st.columns(2)
        with a:
            st.subheader("Crashes by City")
            city_agg = cf.groupby(["city", "real"]).agg(crashes=("collision_id", "count"), injured=("injured", "sum"), killed=("killed", "sum")).reset_index()
            ch_city = alt.Chart(city_agg).mark_bar().encode(
                x=alt.X("crashes:Q", title="Crash records"),
                y=alt.Y("city:N", sort="-x", title=""),
                color=alt.Color("real:N", scale=alt.Scale(domain=[True, False], range=["#2563EB", "#94A3B8"]), title="Real data"),
                tooltip=["city", "crashes", "injured", "killed", "real"],
            ).properties(height=320)
            st.altair_chart(ch_city, width="stretch")

        with b:
            st.subheader("Monthly Crash Timeline")
            monthly = (
                cf.dropna(subset=["crash_date"])
                .assign(month=lambda x: pd.to_datetime(x.crash_date, errors="coerce").dt.to_period("M").dt.to_timestamp())
                .groupby(["month", "city"]).size().reset_index(name="crashes")
            )
            ch_mon = alt.Chart(monthly).mark_line().encode(
                x=alt.X("month:T", title="Month"),
                y=alt.Y("crashes:Q", title="Crashes"),
                color="city:N",
                tooltip=["month:T", "city", "crashes"],
            ).properties(height=320)
            st.altair_chart(ch_mon, width="stretch")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Severity Mix by City")
            sev_df = cf.assign(
                severity_tier=np.where(cf.killed > 0, "Fatal", np.where(cf.injured > 0, "Injury", "No injury recorded"))
            ).groupby(["city", "severity_tier"]).size().reset_index(name="count")
            ch_sev = alt.Chart(sev_df).mark_bar().encode(
                x=alt.X("count:Q", stack="normalize", title="Share of records"),
                y=alt.Y("city:N", title=""),
                color=alt.Color(
                    "severity_tier:N",
                    scale=alt.Scale(domain=["Fatal", "Injury", "No injury recorded"], range=["#0F172A", "#2563EB", "#CBD5E1"]),
                    title="Severity",
                ),
                tooltip=["city", "severity_tier", "count"],
            ).properties(height=320)
            st.altair_chart(ch_sev, width="stretch")
        with c2:
            st.subheader("Near-Miss Rate by Vehicle Pair (Simulated)")
            s = D["scenarios"].assign(pair=lambda x: x.type_a + " vs " + x.type_b)
            pair_df = s.groupby("pair").event.mean().sort_values(ascending=False).reset_index().rename(columns={"event": "near_miss_rate"})
            ch_pair = alt.Chart(pair_df.head(12)).mark_bar(color="#2563EB").encode(
                x=alt.X("near_miss_rate:Q", title="Near-miss rate"),
                y=alt.Y("pair:N", sort="-x", title=""),
                tooltip=["pair", alt.Tooltip("near_miss_rate:Q", format=".3f")],
            ).properties(height=320)
            st.altair_chart(ch_pair, width="stretch")

        st.subheader("Top-Scoring Real Blackspots")
        st.dataframe(pd.DataFrame(rc.TOP_REAL_BLACKSPOTS), width="stretch", hide_index=True)

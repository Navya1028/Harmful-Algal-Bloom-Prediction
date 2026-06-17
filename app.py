import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from hab_predictor import HABPredictor  # noqa: F401  (needed for unpickling)
# PAGE CONFIG
st.set_page_config(
    page_title="HAB Risk Monitor",
    layout="wide",
    initial_sidebar_state="expanded",
)

import pathlib

# Looks for hab_model.pkl in the same folder as this script — works both
# on your local machine and after deploying from GitHub. Override by
# setting the HAB_MODEL_PATH environment variable if you keep the file
# somewhere else (e.g. D:\algal4\hab_model.pkl on your own PC).
MODEL_PATH = os.environ.get(
    "HAB_MODEL_PATH",
    str(pathlib.Path(__file__).parent / "hab_model.pkl")
)

RISK_COLORS = {
    1: "#2F7D52",   # Safe      - matches palette's algae green
    2: "#C99A2E",   # Caution   - amber
    3: "#C97B2E",   # Warning   - burnt orange
    4: "#A8412F",   # Danger    - brick red
}
RISK_RGBA_FAINT = {   # translucent fills for gauge steps (fixes the ValueError)
    1: "rgba(46,139,87,0.22)",
    2: "rgba(212,172,13,0.22)",
    3: "rgba(230,126,34,0.22)",
    4: "rgba(192,57,43,0.22)",
}
RISK_NAMES = {1: "Safe", 2: "Caution", 3: "Warning", 4: "Danger"}
RISK_RANGE = {1: "< 10 µg/L", 2: "10–50 µg/L", 3: "50–100 µg/L", 4: "> 100 µg/L"}

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --deep-water: #082C24;
    --mid-water: #2F6B43;
    --algae: #1D4F45;
    --foam: #F6F7F5;
    --ink: #1A1A1A;
    --muted: #52645D;
    --danger: #B44A3A;
    --warning: #C9982D;
}

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
    color: var(--ink);
}

.stApp {
    background-color: var(--foam);
}

/* HEADER */

.dashboard-header {
    background: linear-gradient(
        120deg,
        #082C24 0%,
        #2F6B43 55%,
        #1D4F45 100%
    );
    padding: 2.2rem 2.5rem;
    border-radius: 20px;
    margin-bottom: 1.8rem;
    color: white;
    position: relative;
    overflow: hidden;
}

.dashboard-header::after {
    content: "";
    position: absolute;
    right: -60px;
    top: -60px;
    width: 260px;
    height: 260px;
    border-radius: 50%;
    background: radial-gradient(
        circle,
        rgba(255,255,255,0.08) 0%,
        transparent 70%
    );
}

.dashboard-title {
    font-family: 'Manrope', sans-serif;
    font-size: 2.3rem;
    font-weight: 800;
    margin: 0;
    color: white;
    letter-spacing: -0.03em;
}

.dashboard-subtitle {
    font-family: 'Manrope', sans-serif;
    font-size: 0.95rem;
    font-weight: 500;
    color: rgba(255,255,255,0.85);
    margin-top: 0.5rem;
    letter-spacing: 0.04em;
}

/* CARDS */

.card {
    background: white;
    border: 1px solid #E3E7E4;
    border-radius: 18px;
    padding: 1.6rem;
    color: var(--ink);
    box-shadow: 0 4px 16px rgba(0,0,0,0.04);
}

.card h4 {
    color: var(--deep-water);
    margin-top: 0;
    font-weight: 700;
}

.card p,
.card li {
    color: var(--ink);
    line-height: 1.65;
}

/* LABELS */

.field-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 700;
    color: var(--muted);
    margin-bottom: 0.2rem;
}

/* METRIC CARDS */

.metric-card {
    background: white;
    border: none;
    border-left: 5px solid var(--mid-water);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 4px 16px rgba(0,0,0,0.04);
}

.metric-card .value {
    font-family: 'Manrope', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: var(--deep-water);
}

.metric-card .label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 700;
    color: var(--muted);
}

/* RISK BANNER */

.risk-banner {
    border-radius: 18px;
    padding: 1.5rem 1.8rem;
    color: white;
    font-family: 'Manrope', sans-serif;
    box-shadow: 0 6px 20px rgba(0,0,0,0.08);
}

.risk-banner .field-label {
    color: rgba(255,255,255,0.85);
}

.risk-banner .level {
    font-size: 2.4rem;
    font-weight: 800;
    margin: 0;
    color: white;
}

.risk-banner .range {
    font-size: 0.95rem;
    color: rgba(255,255,255,0.95);
}

/* DIVIDER */

.section-divider {
    border-top: 1px solid #DCE5DE;
    margin: 2rem 0 1.5rem 0;
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #082C24 0%,
        #0D3A2D 100%
    );
}

[data-testid="stSidebar"] * {
    color: #F6F7F5 !important;
}

[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label {
    font-family: 'Manrope', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
}

[data-testid="stSidebar"] input {
    color: #1A1A1A !important;
    background: white !important;
    border-radius: 10px !important;
}

/* TABS */

.stTabs [data-baseweb="tab"] {
    font-family: 'Manrope', sans-serif;
    font-weight: 600;
    font-size: 0.92rem;
}

/* FOOTNOTE */

.footnote {
    font-size: 0.75rem;
    color: var(--muted);
    line-height: 1.6;
}

/* TABLES */

table {
    color: var(--ink);
}

[data-testid="stMarkdownContainer"] {
    color: var(--ink);
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# MODEL LOADING
# ----------------------------------------------------------------------
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


model = load_model(MODEL_PATH)

# ----------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------
st.markdown("""
<div class="dashboard-header">
    <p class="dashboard-title">Harmful ALgal Blooming Risk Monitor</p>
    <p class="dashboard-subtitle">CHLOROPHYLL-A · NUTRIENT LOADING · WHO BLOOM THRESHOLDS — CATBOOST PIPELINE</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error(
        f"Could not find a model file at **{MODEL_PATH}**. "
        "Make sure hab_model.pkl is in the same folder as app.py (or set "
        "the HAB_MODEL_PATH environment variable), and re-save the pickle "
        "using the standalone `hab_predictor.py` class if needed."
    )
    st.stop()

# ----------------------------------------------------------------------
# SIDEBAR — INPUT FORM (used by the Predict tab)
# ----------------------------------------------------------------------
st.sidebar.markdown("Lake Sample Inputs")
st.sidebar.markdown(
    '<p class="footnote">Enter raw field measurements. Engineered features '
    '(N:P ratio, log transforms, anomalies, interactions) are computed automatically.</p>',
    unsafe_allow_html=True
)

with st.sidebar.expander("Water Chemistry", expanded=True):
    NTL = st.number_input("Total Nitrogen — NTL (µg/L)", value=650.0, min_value=0.0, step=10.0)
    PTL = st.number_input("Total Phosphorus — PTL (µg/L)", value=45.0, min_value=0.0, step=1.0)

with st.sidebar.expander("Climate", expanded=True):
    Tmean = st.number_input("Mean Temperature — Tmean", value=22.0, step=0.5)
    Tmean_YrMean = st.number_input("Annual Mean Temp — Tmean_YrMean", value=18.0, step=0.5)
    Precip = st.number_input("Precipitation — Precip", value=85.0, step=1.0)
    Precip_YrMean = st.number_input("Annual Mean Precip — Precip_YrMean", value=90.0, step=1.0)
    LST = st.number_input("Land Surface Temp — LST", value=24.0, step=0.5)
    LST_YrMean = st.number_input("Annual Mean LST — LST_YrMean", value=20.0, step=0.5)
    NPP = st.number_input("Net Primary Productivity — NPP", value=1100.0, step=10.0)
    NPP_YrMean = st.number_input("Annual Mean NPP — NPP_YrMean", value=1000.0, step=10.0)

with st.sidebar.expander("Lake Physical Characteristics"):
    Month = st.slider("Sampling Month", 5, 10, 7)
    Year = st.number_input("Year", value=2023, step=1)
    lake_AREA_HA = st.number_input("Lake Area (hectares)", value=35.0, step=1.0)
    WSAREA_km2 = st.number_input("Watershed Area (km²)", value=120.0, step=1.0)
    Depth = st.number_input("Lake Depth (m)", value=4.5, step=0.1)
    LAT_DD = st.number_input("Latitude", value=42.5, step=0.1)
    LON_DD = st.number_input("Longitude", value=-85.2, step=0.1)

with st.sidebar.expander("Anthropogenic Nutrient Pressure"):
    nani = st.number_input("NANI — Net Anthro. N Inputs", value=4500.0, step=50.0)
    NAPI = st.number_input("NAPI — Net Anthro. P Inputs", value=320.0, step=10.0)
    Legacy = st.number_input("Legacy Phosphorus (soil)", value=850.0, step=10.0)
    Total_Input = st.number_input("Total P Input", value=410.0, step=10.0)

with st.sidebar.expander("Landscape / Watershed"):
    BFIWs = st.slider("Baseflow Index (BFIWs)", 0.0, 1.0, 0.55)
    wetlands = st.number_input("Wetland Cover (%)", value=8.0, step=0.5)
    RunoffWs = st.number_input("Runoff (RunoffWs)", value=35.0, step=1.0)

run_btn = st.sidebar.button("▶ Run Prediction", use_container_width=True)

# ----------------------------------------------------------------------
# BUILD INPUT ROW
# ----------------------------------------------------------------------
input_row = pd.DataFrame([{
    "NTL": NTL, "PTL": PTL,
    "Tmean": Tmean, "Tmean_YrMean": Tmean_YrMean,
    "Precip": Precip, "Precip_YrMean": Precip_YrMean,
    "LST": LST, "LST_YrMean": LST_YrMean,
    "NPP": NPP, "NPP_YrMean": NPP_YrMean,
    "Month": Month, "Year": Year,
    "lake_AREA_HA": lake_AREA_HA, "WSAREA_km2": WSAREA_km2,
    "Depth": Depth, "LAT_DD": LAT_DD, "LON_DD": LON_DD,
    "nani": nani, "NAPI": NAPI, "Legacy": Legacy, "Total Input": Total_Input,
    "BFIWs": BFIWs, "wetlands": wetlands, "RunoffWs": RunoffWs,
}])

# ----------------------------------------------------------------------
# TABS — Overview first, then live Predict, then Model Performance
# ----------------------------------------------------------------------
tab_about, tab_predict, tab_model = st.tabs(["Overview", "Predict", "Model Performance"])

# ========================================================================
# TAB 1 — OVERVIEW (explains the dashboard & model before anything else)
# ========================================================================
with tab_about:
    st.markdown("""
    <div class="card">
    <h4>What this dashboard does</h4>
    <p>This tool predicts <b>chlorophyll-a concentration</b> — the standard proxy for
    harmful algal bloom (HAB) severity — for a lake sample, using a trained
    <b>CatBoost</b> gradient-boosting model. Enter field measurements in the sidebar
    (nutrients, climate, lake geometry, watershed pressure), and the <b>Predict</b> tab
    will return a risk category aligned to WHO chlorophyll-a thresholds, along with
    supporting charts. The <b>Model Performance</b> tab shows how the model did on
    held-out test data.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="card">
        <h4>Purpose</h4>
        <p>Predict chlorophyll-a levels in US lakes to flag HAB risk and support
        public-health and water-resource decisions before a bloom becomes visible.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card">
        <h4> Data Source</h4>
        <p>EPA National Lakes Assessment (NLA) survey data, combined with PRISM
        climate records, NNI anthropogenic nutrient inputs, and LakeCat watershed
        characteristics.</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="card">
        <h4>Technology</h4>
        <p>CatBoost gradient boosting on engineered features capturing water
        chemistry, climate anomalies, land use, and decades of nutrient loading.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <h4>Risk Classification System</h4>
    <p>Predictions are converted from log-scale chlorophyll-a back to µg/L, then
    bucketed into four WHO-aligned severity levels:</p>
    </div>
    """, unsafe_allow_html=True)

    risk_table_cols = st.columns(4)
    risk_info = [
        (1, "Safe", "< 10 µg/L", "No action needed"),
        (2, "Caution", "10–50 µg/L", "Post signage, monitor weekly"),
        (3, "Warning", "50–100 µg/L", "Limit water contact"),
        (4, "Danger", "> 100 µg/L", "Close water body"),
    ]
    for col, (lvl, name, rng, action) in zip(risk_table_cols, risk_info):
        with col:
            st.markdown(f"""
            <div class="risk-banner" style="background:{RISK_COLORS[lvl]}; padding:1rem 1.1rem;">
                <p class="field-label">Level {lvl}</p>
                <p class="level" style="font-size:1.3rem;">{name}</p>
                <p class="range">{rng}</p>
                <p style="font-size:0.78rem; margin-top:0.4rem; color:#FFFFFF;">{action}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("""
    <p class="footnote">Built for exploratory/decision-support use. Not a substitute for
    laboratory chlorophyll-a measurement or regulatory water-quality testing.
    Head to the <b>Predict</b> tab to try it live.</p>
    """, unsafe_allow_html=True)


# ========================================================================
# TAB 2 — PREDICT (live interaction)
# ========================================================================
with tab_predict:
    left, right = st.columns([1, 1.3])

    if run_btn or "last_pred" in st.session_state:
        try:
            result = model.predict(input_row)
            st.session_state["last_pred"] = result
            st.session_state["last_inputs"] = input_row.copy()
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        r = st.session_state["last_pred"].iloc[0]
        risk_level = int(r["Risk_Level"])
        color = RISK_COLORS[risk_level]

        with left:
            st.markdown(f"""
            <div class="risk-banner" style="background:{color};">
                <p class="field-label">Predicted Risk Category</p>
                <p class="level">{RISK_NAMES[risk_level]}</p>
                <p class="range">{RISK_RANGE[risk_level]}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

            m1, m2 = st.columns(2)
            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">Predicted Chlorophyll-a</div>
                    <div class="value">{r['Predicted_chla_ugL']:.1f} <span style="font-size:1rem;">µg/L</span></div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">log(Chlorophyll-a)</div>
                    <div class="value">{r['Predicted_log_chla']:.3f}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            action_map = {
                1: "No action needed — conditions are within safe limits.",
                2: "Post advisory signage. Recommend weekly monitoring.",
                3: "Limit recreational water contact, especially for children and pets.",
                4: "Close the water body to recreational use until levels subside.",
            }
            st.info(f"**Recommended action:** {action_map[risk_level]}")

        with right:
            # --- WHO threshold gauge (FIXED: rgba instead of hex+alpha) ---
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=r["Predicted_chla_ugL"],
                number={"suffix": " µg/L", "font": {"size": 36, "color": "#14211A"}},
                gauge={
                    "axis": {"range": [0, 150], "tickwidth": 1, "tickcolor": "#14211A",
                              "tickfont": {"color": "#14211A", "size": 12}},
                    "bar": {"color": color, "thickness": 0.3},
                    "steps": [
                        {"range": [0, 10], "color": RISK_RGBA_FAINT[1]},
                        {"range": [10, 50], "color": RISK_RGBA_FAINT[2]},
                        {"range": [50, 100], "color": RISK_RGBA_FAINT[3]},
                        {"range": [100, 150], "color": RISK_RGBA_FAINT[4]},
                    ],
                    "threshold": {
                        "line": {"color": "#14211A", "width": 3},
                        "thickness": 0.8,
                        "value": r["Predicted_chla_ugL"],
                    },
                },
                title={"text": "WHO Bloom Severity Scale", "font": {"size": 14, "color": "#14211A"}},
            ))
            fig.update_layout(height=300, margin=dict(t=50, b=10, l=20, r=20),
                               paper_bgcolor="rgba(0,0,0,0)", font={"color": "#14211A"})
            st.plotly_chart(fig, use_container_width=True)

            # --- N:P ratio vs Redfield ---
            np_ratio = NTL / (PTL + 1e-6)
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=["This sample"], y=[np_ratio],
                marker_color="#14543B", width=0.4, name="N:P ratio"
            ))
            fig2.add_hline(y=16, line_dash="dash", line_color="#C99A2E",
                            annotation_text="Redfield Ratio (16:1)", annotation_position="top right",
                            annotation_font_color="#14211A")
            fig2.update_layout(
                title={"text": "Nitrogen : Phosphorus Balance", "font": {"color": "#14211A"}},
                height=240, margin=dict(t=40, b=10, l=20, r=20),
                yaxis_title="N:P molar ratio",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#14211A"},
                xaxis={"tickfont": {"color": "#14211A"}, "title": {"font": {"color": "#14211A"}}},
                yaxis={"tickfont": {"color": "#14211A"}, "title": {"font": {"color": "#14211A"}}},
            )
            st.plotly_chart(fig2, use_container_width=True)


        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        # --- Feature driver chart: which inputs are elevated vs. training medians ---
        st.markdown("#### What's driving this prediction")
        st.markdown(
            '<p class="footnote">Each bar shows how far your input sits from the '
            "training-set median, in standard-deviation units. Bars pointing right "
            "are above the typical lake in the dataset.</p>",
            unsafe_allow_html=True
        )

        driver_cols = ["NTL", "PTL", "Tmean", "Precip", "nani", "NAPI", "Legacy", "BFIWs"]
        driver_cols = [c for c in driver_cols if c in model.medians.index]
        z_scores = []
        for c in driver_cols:
            med = model.medians[c]
            spread = max(abs(med), 1e-6)  # rough normalizer since std isn't stored
            z_scores.append((input_row[c].iloc[0] - med) / spread)

        fig3 = go.Figure(go.Bar(
            x=z_scores, y=driver_cols, orientation="h",
            marker_color=["#A8412F" if z > 0 else "#2F7D52" for z in z_scores],
        ))
        fig3.add_vline(x=0, line_color="#4D5C53", line_width=1)
        fig3.update_layout(
            height=320, margin=dict(t=20, b=20, l=20, r=20),
            xaxis_title="Relative deviation from training median",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#14211A"},
            xaxis={"tickfont": {"color": "#14211A"}, "title": {"font": {"color": "#14211A"}}},
            yaxis={"tickfont": {"color": "#14211A"}, "title": {"font": {"color": "#14211A"}}},
        )
        st.plotly_chart(fig3, use_container_width=True)

        # --- Seasonal context line: where this month sits in the bloom calendar ---
        st.markdown("#### Seasonal bloom context")
        months = list(range(5, 11))
        month_names = ["May", "Jun", "Jul", "Aug", "Sep", "Oct"]
        seasonal_risk = [0.3, 0.5, 0.9, 1.0, 0.7, 0.4]  # illustrative bloom-likelihood curve
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=month_names, y=seasonal_risk, mode="lines+markers",
            line=dict(color="#14543B", width=3), marker=dict(size=8, color="#14543B"),
            name="Typical bloom likelihood"
        ))
        selected_idx = Month - 5
        fig4.add_trace(go.Scatter(
            x=[month_names[selected_idx]], y=[seasonal_risk[selected_idx]],
            mode="markers", marker=dict(size=16, color=color, line=dict(width=2, color="#14211A")),
            name="Your sample"
        ))
        fig4.update_layout(
            height=260, margin=dict(t=20, b=20, l=20, r=20),
            yaxis_title="Relative bloom likelihood",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#14211A"}, showlegend=True,
            legend={"font": {"color": "#14211A"}},
            xaxis={"tickfont": {"color": "#14211A"}, "title": {"font": {"color": "#14211A"}}},
            yaxis={"tickfont": {"color": "#14211A"}, "title": {"font": {"color": "#14211A"}}},
        )
        st.plotly_chart(fig4, use_container_width=True)

    else:
        with left:
            st.markdown("""
            <div class="metric-card" style="border-left-color:#4D5C53;">
                <div class="label">No prediction yet</div>
                <p style="margin-top:0.5rem; color:#14211A;">
                Set lake sample values in the sidebar and click <b>Run Prediction</b>.
                </p>
            </div>
            """, unsafe_allow_html=True)
        with right:
            st.empty()


# ========================================================================
# TAB 3 — MODEL PERFORMANCE
# ========================================================================
with tab_model:
    st.markdown("### Pipeline Performance (held-out test set)")
    c1, c2, c3 = st.columns(3)
    for col, label, val, suffix in [
        (c1, "Test R²", model.r2, ""),
        (c2, "Test RMSE", model.rmse, " (log scale)"),
        (c3, "Test MAE", model.mae, " (log scale)"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">{label}</div>
                <div class="value">{val:.4f}{suffix}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # R² as a simple gauge-style bar for visual context
    fig_r2 = go.Figure(go.Bar(
        x=[model.r2], y=["Test R²"], orientation="h",
        marker_color="#14543B", width=0.5,
    ))
    fig_r2.update_layout(
        xaxis=dict(range=[0, 1], title={"text": "R² (1.0 = perfect fit)", "font": {"color": "#14211A"}},
                    tickfont={"color": "#14211A"}),
        yaxis=dict(tickfont={"color": "#14211A"}),
        height=140, margin=dict(t=10, b=30, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#14211A"},
    )
    st.plotly_chart(fig_r2, use_container_width=True)

    st.markdown(f"""
    <div class="card">
    <p><b>Model type:</b> CatBoost Regressor<br>
    <b>Total features used:</b> {len(model.feature_names)}<br>
    <b>Risk categories:</b> Safe / Caution / Warning / Danger (WHO chlorophyll-a thresholds)</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("View all feature names used by the model"):
        st.dataframe(pd.DataFrame({"feature": model.feature_names}), use_container_width=True, hide_index=True)

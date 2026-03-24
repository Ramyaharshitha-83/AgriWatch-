"""
app.py  —  AgriWatch Dashboard
Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys, os

sys.path.insert(0, os.path.dirname(__file__))

from data.fetcher       import fetch_all_regions, fetch_stock_data, REGIONS, STOCKS
from data.disease_engine import compute_disease_risk, get_latest_risks, compute_region_summary
from models.ml_models   import DiseaseClassifier, StockImpactModel
from utils.charts       import (climate_correlation_chart, disease_trend_chart,
                                stock_chart, stock_vs_risk_scatter,
                                humidity_rainfall_heatmap, feature_importance_chart)
from utils.map_builder  import build_risk_map


# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgriWatch — Climate & Crop Disease Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0f0a !important;
    color: #e8f4ec;
}
.stApp { background-color: #0a0f0a; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #16201a;
    border: 1px solid #1e2e22;
    border-radius: 12px;
    padding: 16px;
}
[data-testid="stMetricValue"] { color: #4ade80 !important; font-family: 'Syne', sans-serif; }
[data-testid="stMetricDelta"] { font-size: 12px; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111811 !important;
    border-right: 1px solid #1e2e22;
}
[data-testid="stSidebar"] * { color: #9ca3af; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #111811;
    border-bottom: 1px solid #1e2e22;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #4a6452;
    border-radius: 6px 6px 0 0;
    padding: 8px 16px;
    font-size: 13px;
}
.stTabs [aria-selected="true"] {
    background: #16201a;
    color: #4ade80 !important;
    border-bottom: 2px solid #4ade80;
}

/* Buttons */
.stButton > button {
    background: #16201a;
    border: 1px solid #1e4a2a;
    color: #4ade80;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 500;
}
.stButton > button:hover {
    background: #1e4a2a;
    border-color: #4ade80;
}

/* Headers */
h1 { font-family: 'Syne', sans-serif !important; color: #e8f4ec !important; font-weight: 800 !important; }
h2, h3 { font-family: 'Syne', sans-serif !important; color: #9fbbaa !important; font-weight: 700 !important; }

/* Selectbox / inputs */
.stSelectbox [data-baseweb="select"] div {
    background: #16201a;
    border-color: #1e2e22;
    color: #e8f4ec;
}
.stSlider [data-baseweb="slider"] { background: #1e2e22; }

/* DataFrames */
[data-testid="stDataFrameContainer"] {
    background: #111811;
    border: 1px solid #1e2e22;
    border-radius: 8px;
}

/* Divider */
hr { border-color: #1e2e22; }

/* Alert boxes */
.alert-critical { background:#3d151522; border-left:3px solid #f87171; border-radius:6px; padding:10px 14px; margin:6px 0; }
.alert-warning  { background:#3d2d0022; border-left:3px solid #fbbf24; border-radius:6px; padding:10px 14px; margin:6px 0; }
.alert-info     { background:#0f2d1a22; border-left:3px solid #4ade80; border-radius:6px; padding:10px 14px; margin:6px 0; }
.badge-red   { background:#3d1515; color:#f87171; border-radius:12px; padding:2px 10px; font-size:11px; font-weight:600; }
.badge-amber { background:#3d2d00; color:#fbbf24; border-radius:12px; padding:2px 10px; font-size:11px; font-weight:600; }
.badge-green { background:#0f2d1a; color:#4ade80; border-radius:12px; padding:2px 10px; font-size:11px; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ─── Cached data loaders ──────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Fetching climate data...")
def load_climate(days):
    return fetch_all_regions(days)

@st.cache_data(ttl=3600, show_spinner="Fetching stock data...")
def load_stocks(days):
    return fetch_stock_data(days)

@st.cache_data(ttl=1800, show_spinner="Computing disease risk...")
def load_risk(climate_hash):
    climate_df = st.session_state.get("climate_df")
    if climate_df is None:
        return pd.DataFrame()
    return compute_disease_risk(climate_df)

@st.cache_resource(show_spinner="Loading ML models...")
def load_models():
    clf   = DiseaseClassifier();  clf.load()
    stock = StockImpactModel();   stock.load()
    return clf, stock


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌾 AgriWatch")
    st.markdown("<small style='color:#4a6452'>Climate × Crop × Capital</small>", unsafe_allow_html=True)
    st.divider()

    st.markdown("### ⚙️ Settings")
    days_back = st.slider("Historical days", 30, 180, 90, step=15)
    selected_region = st.selectbox("Focus Region", list(REGIONS.keys()))
    selected_stock  = st.selectbox("Focus Stock",  list(STOCKS.keys()))

    st.divider()
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.markdown("### 📊 Data Sources")
    st.markdown("""
    <small style='color:#4a6452'>
    • Climate: Open-Meteo API (free)<br>
    • Stocks: Yahoo Finance (free)<br>
    • ML: scikit-learn Random Forest<br>
    • Maps: Folium + OpenStreetMap<br>
    • Charts: Plotly
    </small>
    """, unsafe_allow_html=True)

    st.divider()
    now = datetime.now().strftime("%d %b %Y  %H:%M")
    st.markdown(f"<small style='color:#2d4a35'>Last updated: {now}</small>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════════════════════
with st.spinner("Loading data..."):
    climate_df = load_climate(days_back)
    stock_df   = load_stocks(days_back)
    st.session_state["climate_df"] = climate_df
    risk_df    = compute_disease_risk(climate_df)
    latest     = get_latest_risks(risk_df)
    region_sum = compute_region_summary(risk_df)
    clf, stock_model = load_models()


# ════════════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════════════
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("# 🌾 AgriWatch Intelligence Dashboard")
    st.markdown("<p style='color:#4a6452;margin-top:-12px'>Climate-driven crop disease analytics & agri-chemical market signals</p>",
                unsafe_allow_html=True)
with col_h2:
    st.markdown(f"""
    <div style='text-align:right;padding-top:12px'>
      <span style='background:#0f2d1a;border:1px solid #16a34a44;color:#4ade80;
                   border-radius:20px;padding:6px 14px;font-size:12px;font-weight:500'>
        ● Live · {selected_region}
      </span>
    </div>""", unsafe_allow_html=True)

st.divider()


# ════════════════════════════════════════════════════════════════════════════
# TOP METRICS
# ════════════════════════════════════════════════════════════════════════════
region_climate = climate_df[climate_df["region"] == selected_region]
latest_climate = region_climate.iloc[-1] if not region_climate.empty else {}

region_risks = latest[latest["region"] == selected_region]
max_risk  = region_risks["risk_score"].max() if not region_risks.empty else 0
top_disease = region_risks.iloc[0]["disease"] if not region_risks.empty else "N/A"

region_stock = stock_df[stock_df["company"] == selected_stock].sort_values("date")
latest_price  = region_stock["close"].iloc[-1]   if not region_stock.empty else 0
latest_change = region_stock["pct_change"].iloc[-1] if not region_stock.empty else 0

active_critical = (latest["risk_level"] == "Critical").sum()

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    t = float(latest_climate.get("temp_avg", 0)) if hasattr(latest_climate, "get") else 0
    st.metric("🌡️ Avg Temp", f"{t:.1f}°C", f"+{t-28:.1f}° vs normal")
with m2:
    h = float(latest_climate.get("humidity", 0)) if hasattr(latest_climate, "get") else 0
    st.metric("💧 Humidity", f"{h:.0f}%", "High" if h > 80 else "Normal")
with m3:
    st.metric("🦠 Disease Risk", f"{max_risk:.0f}/100",
              top_disease, delta_color="inverse")
with m4:
    st.metric("⚠️ Critical Zones", int(active_critical),
              f"across {len(REGIONS)} regions", delta_color="inverse")
with m5:
    chg_sign = "+" if latest_change >= 0 else ""
    st.metric(f"📈 {selected_stock[:12]}", f"₹{latest_price:,.0f}",
              f"{chg_sign}{latest_change:.2f}%",
              delta_color="normal" if latest_change >= 0 else "inverse")

st.divider()


# ════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🗺️ Disease Map",
    "🤖 ML Predictions",
    "📈 Stock Impact",
    "🚨 Alert Feed",
])


# ─── Tab 1: Overview ──────────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### Climate vs Disease Correlation")
        region_data = climate_df[climate_df["region"] == selected_region]
        if not region_data.empty:
            st.plotly_chart(climate_correlation_chart(region_data),
                            use_container_width=True, config={"displayModeBar": False})

        st.markdown("#### Disease Risk Trends")
        if not risk_df.empty:
            st.plotly_chart(disease_trend_chart(risk_df, selected_region),
                            use_container_width=True, config={"displayModeBar": False})

    with col2:
        st.markdown("#### Active Disease Threats")
        top_threats = latest.nlargest(8, "risk_score")[
            ["disease", "crop", "region", "risk_score", "risk_level"]
        ]
        for _, row in top_threats.iterrows():
            badge_class = "badge-red" if row["risk_level"] == "Critical" \
                     else "badge-amber" if row["risk_level"] == "High" else "badge-green"
            st.markdown(f"""
            <div style='background:#16201a;border:1px solid #1e2e22;border-radius:8px;
                        padding:10px 14px;margin-bottom:6px'>
              <div style='display:flex;justify-content:space-between;align-items:center'>
                <div>
                  <span style='font-size:13px;font-weight:500;color:#e8f4ec'>{row['disease']}</span><br>
                  <span style='font-size:11px;color:#4a6452'>{row['crop']} · {row['region']}</span>
                </div>
                <div style='text-align:right'>
                  <span class='{badge_class}'>{row['risk_level']}</span><br>
                  <span style='font-size:13px;font-weight:600;color:#9ca3af'>{row['risk_score']}%</span>
                </div>
              </div>
              <div style='height:4px;background:#1e2e22;border-radius:2px;margin-top:8px'>
                <div style='height:100%;width:{row["risk_score"]}%;background:{"#f87171" if row["risk_level"]=="Critical" else "#fbbf24" if row["risk_level"]=="High" else "#4ade80"};border-radius:2px'></div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.divider()
        st.markdown("#### Humidity Heatmap")
        st.plotly_chart(humidity_rainfall_heatmap(climate_df),
                        use_container_width=True, config={"displayModeBar": False})


# ─── Tab 2: Disease Map ───────────────────────────────────────────────────────
with tab2:
    st.markdown("#### 🗺️ India Crop Disease Risk Map")
    st.markdown("<small style='color:#4a6452'>Click any marker for details. Circle size = risk severity.</small>",
                unsafe_allow_html=True)

    region_sum_with_level = region_sum.copy()
    region_sum_with_level["risk_level"] = region_sum_with_level["max_risk"].apply(
        lambda x: "Critical" if x>=70 else "High" if x>=45 else "Moderate" if x>=25 else "Low"
    )

    try:
        from streamlit_folium import st_folium
        fmap = build_risk_map(region_sum_with_level)
        st_folium(fmap, width=None, height=480, returned_objects=[])
    except ImportError:
        st.info("Install streamlit-folium to view the interactive map: pip install streamlit-folium")

    st.divider()
    st.markdown("#### Region Risk Summary Table")
    display_cols = ["region", "state", "crop", "max_risk", "avg_risk", "risk_level", "diseases"]
    show_df = region_sum_with_level[[c for c in display_cols if c in region_sum_with_level.columns]]
    st.dataframe(show_df, use_container_width=True, hide_index=True)


# ─── Tab 3: ML Predictions ────────────────────────────────────────────────────
with tab3:
    st.markdown("#### 🤖 ML-Powered Disease Prediction")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("##### Manual Climate Input → Predict Risk")
        with st.form("predict_form"):
            p_temp   = st.slider("Temperature (°C)",    15.0, 45.0, 30.0, 0.5)
            p_humid  = st.slider("Humidity (%)",        20.0, 100.0, 72.0, 1.0)
            p_rain   = st.slider("Rainfall (mm/day)",    0.0, 80.0, 8.0, 0.5)
            p_wind   = st.slider("Wind Speed (km/h)",    0.0, 40.0, 12.0, 0.5)
            submitted = st.form_submit_button("🔍 Predict Disease Risk")

        if submitted:
            result = clf.predict({
                "temp_avg": p_temp, "temp_max": p_temp+3, "temp_min": p_temp-4,
                "humidity": p_humid, "rainfall": p_rain, "wind_speed": p_wind
            })
            level  = result["risk_level"]
            conf   = result["confidence"]
            color  = "#f87171" if level=="Critical" else "#fbbf24" if level in ("High","Moderate") else "#4ade80"
            st.markdown(f"""
            <div style='background:#16201a;border:1px solid #1e4a2a;border-radius:10px;padding:16px;margin-top:8px'>
              <p style='color:#9ca3af;font-size:12px;margin:0'>Predicted Risk Level</p>
              <p style='color:{color};font-family:Syne,sans-serif;font-size:28px;font-weight:800;margin:4px 0'>{level}</p>
              <p style='color:#4a6452;font-size:13px;margin:0'>Confidence: {conf}%</p>
              <hr style='border-color:#1e2e22;margin:10px 0'>
              <p style='color:#9ca3af;font-size:12px;margin:0 0 4px'>Probability breakdown:</p>
            """, unsafe_allow_html=True)
            for lvl, prob in result["probabilities"].items():
                c = "#f87171" if lvl=="Critical" else "#fbbf24" if lvl in ("High","Moderate") else "#4ade80"
                st.markdown(f"<div style='display:flex;justify-content:space-between;font-size:12px;margin:3px 0'>"
                            f"<span style='color:#9ca3af'>{lvl}</span>"
                            f"<span style='color:{c};font-weight:600'>{prob}%</span></div>",
                            unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown("##### Batch Prediction on Current Climate")
        if not climate_df.empty:
            latest_climate_all = (climate_df.sort_values("date")
                                            .groupby("region").last().reset_index())
            needed = ["temp_avg","temp_max","temp_min","humidity","rainfall","wind_speed"]
            for col in needed:
                if col not in latest_climate_all:
                    latest_climate_all[col] = 0
            predictions = clf.batch_predict(latest_climate_all[["region","crop"] + needed])
            st.dataframe(predictions[["region","crop","predicted_risk","confidence"]],
                         use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("##### 14-Day Disease Forecast (Prophet-style trend)")

    region_risk_ts = (risk_df[risk_df["region"] == selected_region]
                      .groupby("date")["risk_score"].mean().reset_index())
    if len(region_risk_ts) >= 10:
        import plotly.graph_objects as go
        x = np.arange(len(region_risk_ts))
        y = region_risk_ts["risk_score"].values
        coeffs = np.polyfit(x, y, deg=2)
        future_x = np.arange(len(x), len(x)+14)
        future_y = np.clip(np.polyval(coeffs, future_x), 0, 100)
        future_dates = pd.date_range(region_risk_ts["date"].max(), periods=15, freq="D")[1:]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=region_risk_ts["date"], y=y.round(1),
                                 name="Historical", line=dict(color="#4ade80", width=2)))
        fig.add_trace(go.Scatter(x=future_dates, y=future_y.round(1),
                                 name="Forecast", line=dict(color="#fbbf24", width=2, dash="dot"),
                                 fill="tozeroy", fillcolor="rgba(251,191,36,0.06)"))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#9ca3af"), height=260,
                          xaxis=dict(gridcolor="#1e2e22"),
                          yaxis=dict(gridcolor="#1e2e22", title="Avg Risk Score"),
                          legend=dict(orientation="h", y=1.1, x=1, xanchor="right"),
                          margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ─── Tab 4: Stock Impact ──────────────────────────────────────────────────────
with tab4:
    st.markdown("#### 📈 Agri-Chemical Stock Performance")

    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown(f"##### {selected_stock} — Price Chart")
        if not stock_df.empty:
            st.plotly_chart(stock_chart(stock_df, selected_stock),
                            use_container_width=True, config={"displayModeBar": False})

    with sc2:
        st.markdown("##### All Stocks — Today's Snapshot")
        latest_stocks = (stock_df.sort_values("date")
                                 .groupby("company").last()
                                 .reset_index()[["company","close","pct_change"]])
        for _, row in latest_stocks.iterrows():
            chg   = row["pct_change"]
            color = "#4ade80" if chg >= 0 else "#f87171"
            bg    = "#0f2d1a" if chg >= 0 else "#3d1515"
            st.markdown(f"""
            <div style='background:#16201a;border:1px solid #1e2e22;border-radius:8px;
                        padding:10px 14px;margin-bottom:6px;display:flex;
                        justify-content:space-between;align-items:center'>
              <div>
                <span style='font-size:13px;font-weight:500;color:#e8f4ec'>{row['company']}</span>
              </div>
              <div style='display:flex;align-items:center;gap:12px'>
                <span style='font-size:14px;font-weight:600;color:#e8f4ec'>₹{row['close']:,.0f}</span>
                <span style='background:{bg};color:{color};border-radius:6px;
                             padding:3px 10px;font-size:12px;font-weight:600'>
                  {"+" if chg>=0 else ""}{chg:.2f}%
                </span>
              </div>
            </div>""", unsafe_allow_html=True)

    st.divider()
    col_x, col_y = st.columns(2)
    with col_x:
        st.markdown("##### Disease Risk vs Stock Returns")
        if not risk_df.empty and not stock_df.empty:
            st.plotly_chart(stock_vs_risk_scatter(risk_df, stock_df),
                            use_container_width=True, config={"displayModeBar": False})

    with col_y:
        st.markdown("##### ML Stock Impact Prediction")
        region_r  = latest[latest["region"] == selected_region]
        avg_risk  = region_r["risk_score"].mean() if not region_r.empty else 50
        region_cl = climate_df[climate_df["region"] == selected_region]
        if not region_cl.empty:
            features = {
                "risk_score":          avg_risk,
                "avg_humidity":        region_cl["humidity"].mean(),
                "avg_temp":            region_cl["temp_avg"].mean(),
                "avg_rainfall":        region_cl["rainfall"].mean(),
                "days_above_30":       (region_cl["temp_avg"] > 30).sum(),
                "days_high_humidity":  (region_cl["humidity"] > 80).sum(),
                "rainfall_deficit":    region_cl["rainfall"].mean() - 10,
            }
            predicted_chg = stock_model.predict(features)
            direction = "positive" if predicted_chg >= 0 else "negative"
            color     = "#4ade80" if predicted_chg >= 0 else "#f87171"
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#0f2218,#111811);
                        border:1px solid #1e4a2a;border-radius:10px;padding:20px;margin-top:4px'>
              <p style='color:#4a6452;font-size:12px;margin:0'>Predicted Stock Impact</p>
              <p style='color:{color};font-family:Syne,sans-serif;font-size:32px;font-weight:800;margin:6px 0'>
                {"+" if predicted_chg>=0 else ""}{predicted_chg:.2f}%
              </p>
              <p style='color:#4a6452;font-size:12px;margin:0'>
                Based on {selected_region} disease risk index ({avg_risk:.0f}/100)
              </p>
              <p style='color:#9ca3af;font-size:12px;margin-top:8px'>
                Outlook: <b style='color:{color}'>{direction.upper()}</b> — 
                {"Fungicide/pesticide demand likely rising." if predicted_chg>0 else "Demand pressure easing."}
              </p>
            </div>""", unsafe_allow_html=True)

            st.markdown("##### Feature Importance")
            st.plotly_chart(feature_importance_chart(stock_model.feature_importance()),
                            use_container_width=True, config={"displayModeBar": False})


# ─── Tab 5: Alert Feed ────────────────────────────────────────────────────────
with tab5:
    st.markdown("#### 🚨 Early Warning Alert Feed")

    critical_threats = latest[latest["risk_level"] == "Critical"].sort_values("risk_score", ascending=False)
    high_threats     = latest[latest["risk_level"] == "High"].sort_values("risk_score", ascending=False)
    mod_threats      = latest[latest["risk_level"] == "Moderate"].sort_values("risk_score", ascending=False)

    for _, row in critical_threats.head(4).iterrows():
        st.markdown(f"""
        <div class='alert-critical'>
          <div style='display:flex;justify-content:space-between'>
            <b style='color:#f87171'>🔴 CRITICAL — {row['disease']}</b>
            <small style='color:#9ca3af'>Risk: {row['risk_score']}%</small>
          </div>
          <p style='color:#9ca3af;margin:4px 0 0;font-size:12px'>
            {row['crop']} crops in <b>{row['region']}</b> ({row.get('state','')}) — 
            Temp: {row.get('temp_avg',0):.1f}°C · Humidity: {row.get('humidity',0):.0f}% · 
            Rainfall: {row.get('rainfall',0):.1f}mm/day
          </p>
        </div>""", unsafe_allow_html=True)

    st.divider()
    for _, row in high_threats.head(4).iterrows():
        st.markdown(f"""
        <div class='alert-warning'>
          <div style='display:flex;justify-content:space-between'>
            <b style='color:#fbbf24'>🟡 HIGH RISK — {row['disease']}</b>
            <small style='color:#9ca3af'>Risk: {row['risk_score']}%</small>
          </div>
          <p style='color:#9ca3af;margin:4px 0 0;font-size:12px'>
            {row['crop']} in {row['region']} — Monitor closely. Consider preventive spraying.
          </p>
        </div>""", unsafe_allow_html=True)

    st.divider()
    for _, row in mod_threats.head(3).iterrows():
        st.markdown(f"""
        <div class='alert-info'>
          <b style='color:#4ade80'>🟢 MODERATE — {row['disease']}</b>
          <p style='color:#9ca3af;margin:4px 0 0;font-size:12px'>
            {row['crop']} in {row['region']} · Risk: {row['risk_score']}% — Within safe range. Continue monitoring.
          </p>
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("#### 📋 Full Alert Table")
    alert_df = latest[["disease","crop","region","risk_score","risk_level",
                        "temp_avg","humidity","rainfall"]].copy()
    alert_df.columns = ["Disease","Crop","Region","Risk Score","Level","Temp°C","Humidity%","Rainfall mm"]
    st.dataframe(alert_df.sort_values("Risk Score", ascending=False),
                 use_container_width=True, hide_index=True)


# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center;color:#2d4a35;font-size:12px;padding:8px 0'>
  AgriWatch · Built with Streamlit · Data: Open-Meteo API + Yahoo Finance · 
  ML: scikit-learn Random Forest · Maps: Folium
</div>
""", unsafe_allow_html=True)
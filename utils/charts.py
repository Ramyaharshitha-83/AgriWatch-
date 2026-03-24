"""
utils/charts.py
All Plotly chart builders for the Streamlit dashboard.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# ─── Theme ────────────────────────────────────────────────────────────────────
BG        = "rgba(0,0,0,0)"
PAPER_BG  = "rgba(0,0,0,0)"
GRID_CLR  = "rgba(255,255,255,0.06)"
TEXT_CLR  = "#9ca3af"
GREEN     = "#4ade80"
AMBER     = "#fbbf24"
RED       = "#f87171"
BLUE      = "#60a5fa"
TEAL      = "#2dd4bf"

LAYOUT_BASE = dict(
    paper_bgcolor=PAPER_BG,
    plot_bgcolor=BG,
    font=dict(color=TEXT_CLR, family="DM Sans, sans-serif", size=12),
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(size=11)),
)


def climate_correlation_chart(df: pd.DataFrame) -> go.Figure:
    """Dual-axis: temperature anomaly vs disease outbreak index."""
    monthly = (df.groupby(df["date"].dt.to_period("M"))
                 .agg(temp_avg=("temp_avg","mean"), humidity=("humidity","mean"),
                      rainfall=("rainfall","sum"))
                 .reset_index())
    monthly["date_str"] = monthly["date"].astype(str)

    # Synthetic outbreak index correlated with climate
    idx = (monthly["temp_avg"] - monthly["temp_avg"].mean()) * 3 \
        + (monthly["humidity"] - 70) * 0.8 \
        + np.random.default_rng(7).normal(0, 2, len(monthly))
    idx = np.clip(idx + 40, 5, 95)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly["date_str"], y=monthly["temp_avg"].round(1),
        name="Avg Temp (°C)", mode="lines+markers",
        line=dict(color=RED, width=2),
        marker=dict(size=5), yaxis="y",
        fill="tozeroy", fillcolor="rgba(248,113,113,0.08)",
    ))
    fig.add_trace(go.Scatter(
        x=monthly["date_str"], y=idx.round(1),
        name="Outbreak Index", mode="lines+markers",
        line=dict(color=GREEN, width=2, dash="dot"),
        marker=dict(size=5), yaxis="y2",
        fill="tozeroy", fillcolor="rgba(74,222,128,0.06)",
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        yaxis=dict(title="Temp °C", gridcolor=GRID_CLR, color=RED),
        yaxis2=dict(title="Outbreak Index", overlaying="y", side="right",
                    gridcolor=GRID_CLR, color=GREEN),
        xaxis=dict(gridcolor=GRID_CLR),
        height=260,
    )
    return fig


def disease_trend_chart(risk_df: pd.DataFrame, region: str) -> go.Figure:
    """Line chart of disease risk over time for a region."""
    df = risk_df[risk_df["region"] == region].copy()
    fig = go.Figure()
    colors = [RED, AMBER, GREEN, BLUE, TEAL, "#a78bfa", "#f472b6"]
    for i, (disease, grp) in enumerate(df.groupby("disease")):
        fig.add_trace(go.Scatter(
            x=grp["date"], y=grp["risk_score"],
            name=disease, mode="lines",
            line=dict(color=colors[i % len(colors)], width=2),
        ))
    fig.add_hline(y=70, line_dash="dot", line_color=RED,   annotation_text="Critical", annotation_font_size=10)
    fig.add_hline(y=45, line_dash="dot", line_color=AMBER, annotation_text="High",     annotation_font_size=10)
    fig.update_layout(
        **LAYOUT_BASE,
        xaxis=dict(gridcolor=GRID_CLR),
        yaxis=dict(title="Risk Score", gridcolor=GRID_CLR, range=[0, 105]),
        height=280,
        title=dict(text=f"Disease Risk Trend — {region}", font=dict(size=13, color="#e5e7eb")),
    )
    return fig


def stock_chart(stock_df: pd.DataFrame, company: str) -> go.Figure:
    """Candlestick-style close price chart for one stock."""
    df = stock_df[stock_df["company"] == company].copy().sort_values("date")
    color = GREEN if df["pct_change"].iloc[-1] >= 0 else RED
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["close"].round(2),
        mode="lines", name=company,
        line=dict(color=color, width=2),
        fill="tozeroy", fillcolor=f"rgba({_hex_to_rgb(color)},0.08)",
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        xaxis=dict(gridcolor=GRID_CLR),
        yaxis=dict(title="Price (₹)", gridcolor=GRID_CLR),
        height=220,
    )
    return fig


def stock_vs_risk_scatter(risk_df: pd.DataFrame, stock_df: pd.DataFrame) -> go.Figure:
    """Scatter: avg regional risk vs stock % change."""
    risk_summary = (risk_df.groupby(risk_df["date"].dt.to_period("M"))
                           .agg(avg_risk=("risk_score","mean")).reset_index())
    risk_summary["date"] = risk_summary["date"].dt.to_timestamp()

    fig = go.Figure()
    colors_map = {c: col for c, col in zip(stock_df["company"].unique(),
                                            [GREEN, AMBER, RED, BLUE, TEAL])}
    for company, grp in stock_df.groupby("company"):
        monthly = (grp.set_index("date")["pct_change"]
                      .resample("ME").mean().reset_index())
        merged  = pd.merge_asof(monthly.sort_values("date"),
                                risk_summary.sort_values("date"),
                                on="date", direction="nearest")
        if merged.empty:
            continue
        fig.add_trace(go.Scatter(
            x=merged["avg_risk"], y=merged["pct_change"].round(2),
            mode="markers", name=company,
            marker=dict(size=8, color=colors_map.get(company, BLUE), opacity=0.8),
        ))

    fig.update_layout(
        **LAYOUT_BASE,
        xaxis=dict(title="Avg Disease Risk Score", gridcolor=GRID_CLR),
        yaxis=dict(title="Monthly Stock % Change", gridcolor=GRID_CLR),
        height=300,
        title=dict(text="Disease Risk vs Stock Returns Correlation",
                   font=dict(size=13, color="#e5e7eb")),
    )
    return fig


def humidity_rainfall_heatmap(climate_df: pd.DataFrame) -> go.Figure:
    """Heatmap of avg humidity per region per month."""
    df = climate_df.copy()
    df["month"] = df["date"].dt.strftime("%b")
    pivot = df.pivot_table(index="region", columns="month", values="humidity", aggfunc="mean")
    month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot = pivot.reindex(columns=[m for m in month_order if m in pivot.columns])
    fig = px.imshow(
        pivot,
        color_continuous_scale=[[0,"#0f2218"],[0.5,"#16a34a"],[1,"#4ade80"]],
        text_auto=".0f",
        aspect="auto",
    )
    fig.update_layout(
        **LAYOUT_BASE,
        height=220,
        coloraxis_colorbar=dict(title="Humidity %", tickfont=dict(color=TEXT_CLR)),
        xaxis=dict(side="top"),
        title=dict(text="Monthly Humidity by Region (%)", font=dict(size=13, color="#e5e7eb")),
    )
    fig.update_traces(textfont_size=10)
    return fig


def feature_importance_chart(importances: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=importances["importance"].round(3),
        y=importances["feature"],
        orientation="h",
        marker=dict(color=GREEN, opacity=0.8),
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        xaxis=dict(title="Importance", gridcolor=GRID_CLR),
        yaxis=dict(gridcolor=GRID_CLR),
        height=240,
        title=dict(text="Stock Model — Feature Importance", font=dict(size=13, color="#e5e7eb")),
    )
    return fig


def _hex_to_rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"{r},{g},{b}"
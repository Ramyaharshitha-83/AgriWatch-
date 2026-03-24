"""
utils/map_builder.py
Builds an interactive Folium choropleth/marker map of disease risk across India.
"""

import folium
import pandas as pd
from data.fetcher import REGIONS


RISK_COLORS = {
    "Critical": "#ef4444",
    "High":     "#f97316",
    "Moderate": "#eab308",
    "Low":      "#22c55e",
}

RISK_RADIUS = {
    "Critical": 28,
    "High":     22,
    "Moderate": 16,
    "Low":      11,
}


def build_risk_map(region_summary: pd.DataFrame) -> folium.Map:
    """
    Build an interactive Folium map with circle markers sized/colored
    by disease risk. Returns a folium.Map object.
    """
    m = folium.Map(
        location=[20.5937, 78.9629],   # Centre of India
        zoom_start=5,
        tiles="CartoDB dark_matter",
    )

    for _, row in region_summary.iterrows():
        region_name = row["region"]
        if region_name not in REGIONS:
            continue

        coords  = REGIONS[region_name]
        risk    = row.get("risk_level", _classify(row.get("max_risk", 0)))
        color   = RISK_COLORS.get(risk, "#6b7280")
        radius  = RISK_RADIUS.get(risk, 12)
        max_r   = round(row.get("max_risk", 0), 1)
        avg_r   = round(row.get("avg_risk", 0), 1)
        crop    = row.get("crop", "")
        state   = row.get("state", "")
        diseases = row.get("diseases", "")

        popup_html = f"""
        <div style="font-family:sans-serif;min-width:180px;padding:6px">
          <b style="font-size:14px">{region_name}</b><br>
          <span style="color:#6b7280;font-size:12px">{state} · {crop}</span>
          <hr style="margin:6px 0;border-color:#e5e7eb">
          <table style="font-size:12px;width:100%">
            <tr><td>Max Risk</td><td style="color:{color};font-weight:bold;text-align:right">{max_r}%</td></tr>
            <tr><td>Avg Risk</td><td style="text-align:right">{avg_r}%</td></tr>
            <tr><td>Level</td><td style="color:{color};font-weight:bold;text-align:right">{risk}</td></tr>
          </table>
          <hr style="margin:6px 0;border-color:#e5e7eb">
          <div style="font-size:11px;color:#374151"><b>Diseases:</b><br>{diseases}</div>
        </div>
        """

        folium.CircleMarker(
            location=[coords["lat"], coords["lon"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"{region_name} — {risk} ({max_r}%)",
        ).add_to(m)

    # Legend
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:rgba(17,24,39,0.92);border-radius:8px;
                padding:12px 16px;border:1px solid #374151;">
      <p style="color:#f9fafb;font-weight:bold;font-size:13px;margin:0 0 8px">
        Disease Risk Level
      </p>
      <div style="display:flex;flex-direction:column;gap:5px">
        <span style="color:#ef4444;font-size:12px">⬤ Critical (&gt;70%)</span>
        <span style="color:#f97316;font-size:12px">⬤ High (45–70%)</span>
        <span style="color:#eab308;font-size:12px">⬤ Moderate (25–45%)</span>
        <span style="color:#22c55e;font-size:12px">⬤ Low (&lt;25%)</span>
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


def _classify(score: float) -> str:
    if score >= 70: return "Critical"
    if score >= 45: return "High"
    if score >= 25: return "Moderate"
    return "Low"
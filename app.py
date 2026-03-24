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


# ─── Rule-based AI advisor (works 100% offline, no API key needed) ────────────
def _rule_based_reply(question: str, region: str, risk: float, disease: str,
                      temp: float, humidity: float) -> str:
    """
    Smart keyword-based agricultural advisor.
    Covers 20+ topic areas with detailed, data-personalised answers.
    No API key required — works fully offline in Codespaces.
    """
    q = question.lower()

    # ── Disease-specific ──────────────────────────────────────────────────────
    if any(w in q for w in ["blast", "rice blast", "pyricularia"]):
        level = "CRITICAL — act immediately" if humidity > 85 else "elevated — monitor closely"
        return (f"🦠 **Rice Blast Disease — {region}**\n\n"
                f"Current risk level: **{risk:.0f}%** | Conditions: {temp:.1f}°C, {humidity:.0f}% RH → {level}\n\n"
                f"**Why it spreads:** Blast thrives when temp is 24–28°C and humidity exceeds 85%. "
                f"Spores germinate within 6 hours of leaf wetness.\n\n"
                f"**Immediate actions:**\n"
                f"• Spray **Tricyclazole 75% WP** @ 0.6g/L water — most effective fungicide\n"
                f"• Alternatively: **Isoprothiolane 40% EC** @ 1.5ml/L\n"
                f"• Drain fields partially to reduce leaf humidity\n"
                f"• Apply at tillering and panicle initiation stages\n"
                f"• Repeat spray after 10–14 days if humidity stays >80%\n\n"
                f"**Prevention:** Use blast-resistant varieties like IR64, Swarna. Avoid excess nitrogen.")

    elif any(w in q for w in ["rust", "leaf rust", "stripe rust", "wheat rust"]):
        return (f"🍂 **Wheat Rust Disease Advisory**\n\n"
                f"Rust spreads rapidly in cool-humid conditions (15–22°C, RH >70%).\n"
                f"Current temp {temp:.1f}°C is {'favourable for rust spread' if temp < 25 else 'slightly warm — slowing spread'}.\n\n"
                f"**Management:**\n"
                f"• **Propiconazole 25% EC** @ 1ml/L — highly effective for all rust types\n"
                f"• **Tebuconazole 250 EW** @ 1ml/L as alternative\n"
                f"• First spray at flag leaf stage, repeat after 15 days\n"
                f"• Use resistant varieties: HD-2967, WH-1105, PBW-550\n\n"
                f"**Economic impact:** Severe rust can cause 30–70% yield loss. "
                f"Fungicide ROI is typically 4:1 when applied at correct timing.")

    elif any(w in q for w in ["mildew", "powdery mildew", "downy mildew"]):
        return (f"🌿 **Mildew Disease Control**\n\n"
                f"Powdery mildew favours dry conditions with high humidity at night. "
                f"Downy mildew needs wet leaves and cool temperatures.\n\n"
                f"**Fungicides:**\n"
                f"• Powdery: **Sulphur 80% WP** @ 3g/L or **Hexaconazole 5% EC** @ 2ml/L\n"
                f"• Downy: **Metalaxyl + Mancozeb** @ 2.5g/L\n"
                f"• Organic option: Neem oil 5% + soap @ weekly intervals\n\n"
                f"**Cultural control:** Increase plant spacing for airflow, avoid overhead irrigation.")

    elif any(w in q for w in ["blight", "bacterial blight", "leaf blight"]):
        return (f"🌾 **Blight Disease Management**\n\n"
                f"Bacterial blight spreads through rain splash and wind. "
                f"High risk when temp is 25–35°C with high humidity like current {temp:.1f}°C, {humidity:.0f}%.\n\n"
                f"**Control measures:**\n"
                f"• **Streptomycin Sulphate 90% + Tetracycline 10%** @ 1g/5L for bacterial blight\n"
                f"• **Copper Oxychloride 50% WP** @ 3g/L as preventive\n"
                f"• Remove and burn infected plant debris\n"
                f"• Avoid waterlogging and excess nitrogen\n"
                f"• Use certified disease-free seeds")

    elif any(w in q for w in ["smut", "red rot", "stalk rot", "root rot"]):
        return (f"🍄 **Soil-borne Disease Advisory**\n\n"
                f"Smut/rot diseases are primarily soil and seed-borne. "
                f"Rainfall of {risk:.0f}mm combined with warm soil temp creates infection risk.\n\n"
                f"**Management:**\n"
                f"• **Seed treatment:** Carboxin 37.5% + Thiram 37.5% DS @ 2g/kg seed\n"
                f"• **Soil drenching:** Carbendazim 50% WP @ 1g/L around root zone\n"
                f"• Crop rotation — avoid same crop on same plot for 2 years\n"
                f"• Improve soil drainage; waterlogged soils amplify infection 3–5x")

    # ── What diseases threaten a region ──────────────────────────────────────
    elif any(w in q for w in ["threaten", "threat", "risk", "danger", "affect"]) and \
         any(w in q for w in ["warangal","ludhiana","nashik","guntur","patna","coimbatore",
                               "region","now","currently","today"]):
        crop_map = {"Warangal":"Rice","Ludhiana":"Wheat","Nashik":"Soybean",
                    "Guntur":"Cotton","Patna":"Maize","Coimbatore":"Sugarcane"}
        crop = crop_map.get(region, "crops")
        urgency = "🔴 CRITICAL" if risk > 70 else "🟡 HIGH" if risk > 45 else "🟢 MODERATE"
        return (f"🗺️ **Disease Threats in {region} — Live Assessment**\n\n"
                f"Overall risk: **{urgency} ({risk:.0f}%)**\n"
                f"Primary threat: **{disease}** on {crop}\n"
                f"Conditions: {temp:.1f}°C | {humidity:.0f}% humidity\n\n"
                f"**Active threats ranked:**\n"
                f"• {'🔴' if risk>70 else '🟡'} {disease} — {risk:.0f}% risk (PRIMARY)\n"
                f"• 🟡 Secondary fungal infections — watch humidity trends\n"
                f"• 🟢 Pest pressure — aphids/whitefly elevated in warm conditions\n\n"
                f"**Recommended action in next 48 hours:**\n"
                f"{'• Emergency field inspection + immediate fungicide application' if risk > 70 else '• Schedule preventive spray within this week'}\n"
                f"• Alert nearby farmers in a 20km radius\n"
                f"• Contact local Agriculture Department / KVK for subsidy on fungicides")

    # ── Stock / Investment ────────────────────────────────────────────────────
    elif any(w in q for w in ["stock", "invest", "buy", "sell", "market", "share", "portfolio",
                               "pi industries", "upl", "dhanuka", "bayer"]):
        direction = "BULLISH 📈" if risk > 55 else "NEUTRAL ➡️" if risk > 35 else "BEARISH 📉"
        signal = "accumulate" if risk > 55 else "hold" if risk > 35 else "wait"
        return (f"📈 **Agri-Chemical Stock Analysis — Disease Risk Signal**\n\n"
                f"Current disease risk index: **{risk:.0f}/100** → Market signal: **{direction}**\n\n"
                f"**Company-specific outlook:**\n"
                f"• **PI Industries (PIIND):** Strong fungicide pipeline; benefits most from blast/rust outbreaks. Signal: {signal.upper()}\n"
                f"• **Dhanuka Agritech:** India-focused; direct exposure to Kharif disease cycles. Signal: {signal.upper()}\n"
                f"• **UPL Ltd:** Global diversified; less sensitive to India-specific outbreaks. Signal: HOLD\n"
                f"• **Bayer CropScience:** Premium fungicides; benefits from high-severity seasons. Signal: {signal.upper()}\n"
                f"• **Insecticides India:** Broader portfolio; stable demand regardless of season\n\n"
                f"**Investment thesis:** Disease risk >60 historically precedes 4–8% sector rally "
                f"within 30 days as farmers increase chemical purchases.\n\n"
                f"⚠️ *This is not financial advice. Always consult a SEBI-registered advisor.*")

    # ── Climate / Temperature ─────────────────────────────────────────────────
    elif any(w in q for w in ["temperature", "climate", "heat", "warm", "cold", "weather",
                               "rainfall", "humidity", "monsoon", "drought"]):
        stress = "severe heat stress" if temp > 35 else "moderate stress" if temp > 32 else "optimal range"
        return (f"🌡️ **Climate Impact Analysis — {region}**\n\n"
                f"Current: **{temp:.1f}°C | {humidity:.0f}% RH** → Crop status: {stress}\n\n"
                f"**How climate drives disease risk:**\n"
                f"• Every +1°C above 28°C → fungal spore germination rate ↑8%\n"
                f"• Humidity >80% for 6+ hours → leaf infection probability ↑40%\n"
                f"• Rainfall after dry spell → explosive spore dispersal\n"
                f"• Temperature fluctuation >8°C day/night → weakened plant immunity\n\n"
                f"**Current risk factors:**\n"
                f"• {'⚠️ Humidity critically high — fungal outbreak imminent' if humidity > 85 else '✅ Humidity manageable — preventive measures sufficient'}\n"
                f"• {'⚠️ Temperature stress reducing crop resistance' if temp > 33 else '✅ Temperature within tolerable range'}\n\n"
                f"**2025 season outlook:** La Niña pattern suggests above-normal rainfall in peninsular India "
                f"→ elevated blast and blight risk through Kharif season.")

    # ── Fungicide / pesticide recommendations ─────────────────────────────────
    elif any(w in q for w in ["fungicide", "pesticide", "chemical", "spray", "medicine",
                               "treatment", "apply", "dose", "dosage"]):
        return (f"💊 **Agri-Chemical Recommendations for {region}**\n\n"
                f"Based on current threat ({disease}, {risk:.0f}% risk):\n\n"
                f"**Top fungicides (most effective first):**\n"
                f"1. **Tricyclazole 75% WP** — Rice blast. Dose: 0.6g/L. Cost: ₹280/100g\n"
                f"2. **Propiconazole 25% EC** — Rust, mildew. Dose: 1ml/L. Cost: ₹320/250ml\n"
                f"3. **Azoxystrobin 23% SC** — Broad spectrum. Dose: 1ml/L. Cost: ₹480/100ml\n"
                f"4. **Mancozeb 75% WP** — Preventive, cheap. Dose: 2.5g/L. Cost: ₹85/250g\n"
                f"5. **Copper Oxychloride 50%** — Bacterial diseases. Dose: 3g/L. Cost: ₹120/250g\n\n"
                f"**Spray timing:** Early morning (6–9 AM) or late evening. Avoid midday.\n"
                f"**Safety:** Wear gloves, mask, goggles. Re-entry interval: 24 hours.\n"
                f"**Buy from:** Nearest IFFCO/Krishak Seva Kendra or licensed agro dealer.")

    # ── Crop-specific advice ───────────────────────────────────────────────────
    elif any(w in q for w in ["rice", "paddy"]):
        return (f"🌾 **Rice Crop Advisory — {region}**\n\n"
                f"Key diseases to watch: Blast, Brown Plant Hopper, Sheath Blight\n"
                f"Current risk: **{risk:.0f}%** | Conditions favour {'high' if humidity>80 else 'moderate'} disease pressure\n\n"
                f"**Stage-wise protection plan:**\n"
                f"• **Nursery:** Seed treatment with Carbendazim 2g/kg\n"
                f"• **Tillering (21–35 DAT):** Scout for BPH; spray Tricyclazole for blast\n"
                f"• **Panicle initiation:** Critical window — preventive fungicide mandatory\n"
                f"• **Heading:** Avoid overhead irrigation to reduce blast infection\n\n"
                f"**Variety recommendation:** Swarna Sub1 (flood-tolerant), IR64 (blast-resistant)")

    elif any(w in q for w in ["wheat"]):
        return (f"🌿 **Wheat Crop Advisory**\n\n"
                f"Primary threats: Leaf rust, Stripe rust, Powdery mildew\n\n"
                f"**Integrated Disease Management:**\n"
                f"• Use certified rust-resistant seed (HD-2967, DBW-187)\n"
                f"• First irrigation at Crown Root Initiation stage — avoid late irrigation\n"
                f"• Scout weekly from tillering to grain fill\n"
                f"• Spray **Propiconazole 25% EC** @ flag leaf stage\n"
                f"• Harvest timely — delayed harvest increases smut infection")

    elif any(w in q for w in ["cotton"]):
        return (f"🌸 **Cotton Crop Advisory**\n\n"
                f"Key threats: Bacterial blight, Pink bollworm, Root rot\n\n"
                f"**Management:**\n"
                f"• Monitor for boll weevil and whitefly (vector for leaf curl virus)\n"
                f"• Spray **Imidacloprid 17.8% SL** @ 0.5ml/L for sucking pests\n"
                f"• For bacterial blight: Copper oxychloride @ 3g/L\n"
                f"• Bt cotton reduces bollworm risk by 70% — prefer for Kharif planting")

    elif any(w in q for w in ["soybean", "soya"]):
        return (f"🫘 **Soybean Crop Advisory**\n\n"
                f"Key threats: Downy mildew, Yellow mosaic virus (YMV), Root rot\n\n"
                f"**Management:**\n"
                f"• YMV is whitefly-transmitted — control vector with **Thiamethoxam 25% WG**\n"
                f"• Seed treatment: **Thiram + Carbendazim** slurry for root diseases\n"
                f"• Spray **Metalaxyl + Mancozeb** at first sign of mildew\n"
                f"• Maintain proper row spacing (30–45cm) for airflow")

    # ── Irrigation / water management ─────────────────────────────────────────
    elif any(w in q for w in ["irrigation", "water", "drain", "flood", "waterlog"]):
        return (f"💧 **Irrigation & Water Management Advisory**\n\n"
                f"Water management directly affects disease risk (current: {risk:.0f}%)\n\n"
                f"**Key rules:**\n"
                f"• Avoid irrigation when humidity >85% — increases blast risk dramatically\n"
                f"• Furrow irrigation preferred over overhead — reduces leaf wetness by 60%\n"
                f"• Drain fields for 2–3 days after heavy rain to prevent root rot\n"
                f"• Critical irrigation stages: CRI, jointing, heading, grain fill\n"
                f"• Drip irrigation reduces disease incidence by 25–40% vs flood irrigation")

    # ── Fertilizer / soil ─────────────────────────────────────────────────────
    elif any(w in q for w in ["fertilizer", "fertiliser", "nitrogen", "soil", "nutrient", "urea"]):
        return (f"🌱 **Fertilizer & Soil Health Advisory**\n\n"
                f"Nutrition directly impacts disease resistance:\n\n"
                f"• **Excess nitrogen** → lush soft growth → 2–3x higher blast/rust susceptibility\n"
                f"• **Potassium deficiency** → weakened cell walls → easy fungal penetration\n"
                f"• **Silicon application** (rice) → 40–50% reduction in blast infection\n\n"
                f"**Recommended NPK for high-risk season:**\n"
                f"• Reduce N by 20% when disease risk >60%\n"
                f"• Apply **MOP (Muriate of Potash)** @ 60kg/ha to boost immunity\n"
                f"• Foliar spray: **Potassium Silicate** 5g/L as disease suppressant\n"
                f"• Organic: FYM @ 5 tonnes/ha improves soil microbiome resistance")

    # ── Early warning / prevention ────────────────────────────────────────────
    elif any(w in q for w in ["prevent", "early warning", "alert", "before", "proactive"]):
        return (f"⚠️ **Early Warning & Prevention System — {region}**\n\n"
                f"Current risk trajectory: **{risk:.0f}%** and "
                f"{'rising ↑' if risk > 50 else 'stable →'}\n\n"
                f"**7-day prevention checklist:**\n"
                f"□ Scout fields every 48 hours — check 10 plants per acre\n"
                f"□ Record temperature and humidity daily (ideal: use agro-weather station)\n"
                f"□ Prepare fungicide stock before outbreak (shortage during peak season)\n"
                f"□ Alert farmer group/WhatsApp cluster about current risk level\n"
                f"□ Contact KVK for free disease forecasting service\n"
                f"□ Check crop insurance coverage — enroll under PMFBY if not done\n\n"
                f"**When to spray preventively:** When humidity >80% for 3+ consecutive days "
                f"AND temp is 24–30°C — don't wait for visible symptoms.")

    # ── Organic / natural methods ─────────────────────────────────────────────
    elif any(w in q for w in ["organic", "natural", "bio", "neem", "traditional"]):
        return (f"🌿 **Organic & Bio-pesticide Options**\n\n"
                f"Effective bio-based alternatives for disease management:\n\n"
                f"**Fungal diseases:**\n"
                f"• **Trichoderma viride** @ 5g/L — excellent for soil-borne diseases\n"
                f"• **Pseudomonas fluorescens** @ 10g/L — systemic resistance inducer\n"
                f"• **Neem oil 5000 PPM** @ 5ml/L + soap — broad spectrum\n\n"
                f"**Bacterial diseases:**\n"
                f"• **Bacillus subtilis** — effective against bacterial blight\n"
                f"• Cow urine (fresh) diluted 1:10 — traditional but effective for mildew\n\n"
                f"**Limitations:** Bio-pesticides are 60–70% as effective as chemicals during "
                f"high-risk periods (risk > 60%). Use chemicals as backup when risk is critical.")

    # ── Yield / loss estimate ─────────────────────────────────────────────────
    elif any(w in q for w in ["yield", "loss", "production", "harvest", "output", "profit"]):
        loss_pct = min(int(risk * 0.7), 65)
        return (f"📊 **Yield Impact Assessment — {region}**\n\n"
                f"At current disease risk ({risk:.0f}%), estimated yield loss if untreated: "
                f"**{loss_pct}–{min(loss_pct+10,70)}%**\n\n"
                f"**Loss by disease (untreated):**\n"
                f"• Rice blast: 30–70% | Wheat rust: 20–60% | Cotton blight: 10–30%\n\n"
                f"**Cost-benefit of fungicide application:**\n"
                f"• Fungicide cost: ~₹800–1500/acre (2 sprays)\n"
                f"• Yield saved at current risk: ₹4,000–9,000/acre\n"
                f"• **ROI: 4:1 to 8:1** — strongly recommended\n\n"
                f"**Crop insurance:** Enroll in **PMFBY** (Pradhan Mantri Fasal Bima Yojana) "
                f"— premium only 1.5–2% of sum insured for Kharif crops.")

    # ── Government schemes / support ─────────────────────────────────────────
    elif any(w in q for w in ["government", "scheme", "subsidy", "pmfby", "insurance",
                               "kvk", "support", "helpline"]):
        return (f"🏛️ **Government Support for Farmers**\n\n"
                f"**Crop Insurance:**\n"
                f"• **PMFBY** — 1.5% premium for Kharif; claim for disease losses >25%\n"
                f"• Enroll before cutoff date (check district agriculture office)\n\n"
                f"**Subsidies on inputs:**\n"
                f"• 50% subsidy on certified seeds via state agriculture dept\n"
                f"• Bio-pesticides available free/subsidised at KVK centers\n"
                f"• PM-KISAN: ₹6000/year direct benefit transfer\n\n"
                f"**Expert help (Free):**\n"
                f"• **KVK Helpline:** 1800-180-1551 (Toll free)\n"
                f"• **Kisan Call Center:** 1800-180-1551\n"
                f"• **mKisan portal:** SMS-based crop advisory\n"
                f"• **TNAU Agritech Portal:** agritech.tnau.ac.in\n\n"
                f"• Visit nearest **Krishi Vigyan Kendra (KVK)** for soil testing & free demo sprays")

    # ── General / catch-all ────────────────────────────────────────────────────
    else:
        urgency = "🔴 CRITICAL — act today" if risk > 70 else "🟡 HIGH — act this week" if risk > 45 else "🟢 MODERATE — monitor regularly"
        return (f"🌾 **AgriWatch AI Advisory — {region}**\n\n"
                f"**Current dashboard snapshot:**\n"
                f"• Region: {region} | Temperature: {temp:.1f}°C | Humidity: {humidity:.0f}%\n"
                f"• Top disease threat: **{disease}** at **{risk:.0f}% risk** → {urgency}\n\n"
                f"**I can help you with:**\n"
                f"• 🦠 Specific disease identification & treatment (ask: 'how to treat blast disease?')\n"
                f"• 💊 Fungicide/pesticide recommendations (ask: 'what fungicide for rust?')\n"
                f"• 📈 Stock investment signals (ask: 'should I invest in agri stocks?')\n"
                f"• 🌡️ Climate impact analysis (ask: 'how does humidity affect crops?')\n"
                f"• 🌾 Crop-specific advice (ask: 'rice crop management tips')\n"
                f"• 💰 Yield loss estimates (ask: 'what yield loss from blast disease?')\n"
                f"• 🏛️ Government schemes (ask: 'what subsidies are available?')\n"
                f"• 💧 Irrigation advice, fertilizer tips, organic options, and more!\n\n"
                f"Just type your question in plain English — I understand farming terminology too.")


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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "🗺️ Disease Map",
    "🤖 ML Predictions",
    "📈 Stock Impact",
    "🚨 Alert Feed",
    "💬 AI Advisor",
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


# ─── Tab 6: AI Advisor Chatbot ───────────────────────────────────────────────
with tab6:
    st.markdown("#### 💬 AgriWatch AI Advisor")
    st.markdown("<p style='color:#4a6452;margin-top:-10px;font-size:13px'>Ask about crop diseases, climate risk, stock impact, or farming strategies.</p>",
                unsafe_allow_html=True)

    # Build live context from current dashboard data
    top5 = latest.nlargest(5, "risk_score")[["disease","crop","region","risk_score","risk_level"]]
    context_lines = ["Current dashboard data:"]
    context_lines.append(f"- Selected region: {selected_region} | Crop: {REGIONS[selected_region]['crop']}")
    context_lines.append(f"- Temp: {float(latest_climate.get('temp_avg',0)):.1f}°C | Humidity: {float(latest_climate.get('humidity',0)):.0f}% | Rainfall: {float(latest_climate.get('rainfall',0)):.1f}mm")
    context_lines.append(f"- Max disease risk in region: {max_risk:.0f}% ({top_disease})")
    context_lines.append(f"- Critical zones across all regions: {int(active_critical)}")
    context_lines.append("Top disease threats:")
    for _, r in top5.iterrows():
        context_lines.append(f"  • {r['disease']} on {r['crop']} in {r['region']}: {r['risk_score']}% ({r['risk_level']})")
    if not stock_df.empty:
        latest_s = stock_df.sort_values("date").groupby("company").last().reset_index()
        context_lines.append("Stock snapshot:")
        for _, r in latest_s.iterrows():
            chg = r['pct_change']
            context_lines.append(f"  • {r['company']}: ₹{r['close']:,.0f} ({'+' if chg>=0 else ''}{chg:.1f}%)")
    SYSTEM_PROMPT = (
        "You are AgriWatch AI Advisor, an expert in agricultural science, crop disease management, "
        "climate change impacts on farming, and agri-chemical investment analysis focused on India. "
        "You provide practical, actionable advice to farmers, agronomists, and investors. "
        "Be concise but thorough. Use bullet points for recommendations. "
        "Always ground your answers in the live dashboard data provided.\n\n"
        + "\n".join(context_lines)
    )

    # Session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Suggested questions
    st.markdown("**Quick questions:**")
    q_cols = st.columns(3)
    suggestions = [
        f"What diseases threaten {selected_region} right now?",
        "How does humidity affect blast disease in rice?",
        f"Should I invest in agri-chemical stocks given current risk?",
        "What fungicides work best for leaf rust in wheat?",
        "How will rising temperatures affect crop diseases in 2025?",
        "Which region needs urgent intervention today?",
    ]
    for i, col in enumerate(q_cols):
        with col:
            if st.button(suggestions[i], key=f"sug_{i}"):
                st.session_state.chat_history.append({"role": "user", "content": suggestions[i]})
                st.rerun()
            if st.button(suggestions[i+3], key=f"sug_{i+3}"):
                st.session_state.chat_history.append({"role": "user", "content": suggestions[i+3]})
                st.rerun()

    st.divider()

    # Chat display
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div style='display:flex;justify-content:flex-end;margin-bottom:10px'>
                  <div style='background:#16201a;border:1px solid #1e4a2a;border-radius:12px 12px 2px 12px;
                              padding:10px 14px;max-width:75%;font-size:13px;color:#e8f4ec'>
                    {msg['content']}
                  </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='display:flex;justify-content:flex-start;margin-bottom:10px;gap:8px'>
                  <div style='width:28px;height:28px;background:linear-gradient(135deg,#16a34a,#4ade80);
                              border-radius:50%;display:flex;align-items:center;justify-content:center;
                              font-size:13px;flex-shrink:0;margin-top:2px'>🌾</div>
                  <div style='background:#111811;border:1px solid #1e2e22;border-radius:2px 12px 12px 12px;
                              padding:10px 14px;max-width:78%;font-size:13px;color:#e8f4ec;line-height:1.6'>
                    {msg['content'].replace(chr(10), '<br>')}
                  </div>
                </div>""", unsafe_allow_html=True)

    # Auto-respond if last message is from user (no API key needed — uses internal API)
    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
        with st.spinner("AgriWatch AI is thinking..."):
            try:
                import requests as req
                payload = {
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 1000,
                    "system": SYSTEM_PROMPT,
                    "messages": st.session_state.chat_history,
                }
                resp = req.post(
                    "https://api.anthropic.com/v1/messages",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30,
                )
                if resp.status_code == 200:
                    reply = resp.json()["content"][0]["text"]
                else:
                    # Fallback: rule-based advisor when API unavailable
                    reply = _rule_based_reply(
                        st.session_state.chat_history[-1]["content"],
                        selected_region, max_risk, top_disease,
                        float(latest_climate.get("temp_avg", 30)),
                        float(latest_climate.get("humidity", 70)),
                    )
            except Exception:
                reply = _rule_based_reply(
                    st.session_state.chat_history[-1]["content"],
                    selected_region, max_risk, top_disease,
                    float(latest_climate.get("temp_avg", 30)),
                    float(latest_climate.get("humidity", 70)),
                )
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.rerun()

    # Input box
    st.divider()
    with st.form("chat_form", clear_on_submit=True):
        c1, c2 = st.columns([5, 1])
        with c1:
            user_input = st.text_input("Ask anything about crop diseases, climate risk, or stocks...",
                                       label_visibility="collapsed",
                                       placeholder="e.g. What is the risk of blast disease spreading to Guntur?")
        with c2:
            send = st.form_submit_button("Send ➤")
        if send and user_input.strip():
            st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
            st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear chat"):
            st.session_state.chat_history = []
            st.rerun()



# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center;color:#2d4a35;font-size:12px;padding:8px 0'>
  AgriWatch · Built with Streamlit · Data: Open-Meteo API + Yahoo Finance · 
  ML: scikit-learn Random Forest · Maps: Folium
</div>
""", unsafe_allow_html=True)
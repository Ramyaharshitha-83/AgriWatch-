"""
data/disease_engine.py
Computes crop-disease risk scores from climate features.
Rules are derived from agronomic literature; no external API needed.
"""

import pandas as pd
import numpy as np


# ─── Disease rules per crop ───────────────────────────────────────────────────
# Each rule: (disease_name, favorable_conditions_fn) → 0-100 risk
DISEASE_RULES = {
    "Rice": [
        ("Blast Disease",       lambda r: _blast_risk(r)),
        ("Brown Plant Hopper",  lambda r: _bph_risk(r)),
        ("Sheath Blight",       lambda r: _sheath_blight_risk(r)),
    ],
    "Wheat": [
        ("Leaf Rust",           lambda r: _rust_risk(r)),
        ("Stripe Rust",         lambda r: _stripe_rust_risk(r)),
        ("Powdery Mildew",      lambda r: _powdery_mildew_risk(r)),
    ],
    "Soybean": [
        ("Downy Mildew",        lambda r: _downy_mildew_risk(r)),
        ("Soybean Mosaic Virus",lambda r: _virus_risk(r)),
    ],
    "Cotton": [
        ("Bacterial Blight",    lambda r: _bacterial_blight_risk(r)),
        ("Root Rot",            lambda r: _root_rot_risk(r)),
    ],
    "Maize": [
        ("Northern Leaf Blight",lambda r: _leaf_blight_risk(r)),
        ("Stalk Rot",           lambda r: _stalk_rot_risk(r)),
    ],
    "Sugarcane": [
        ("Red Rot",             lambda r: _red_rot_risk(r)),
        ("Smut Disease",        lambda r: _smut_risk(r)),
    ],
}


def compute_disease_risk(climate_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a climate DataFrame (one row per day per region),
    returns a DataFrame with daily disease risk scores.
    """
    records = []
    for _, row in climate_df.iterrows():
        crop     = row.get("crop", "Rice")
        diseases = DISEASE_RULES.get(crop, DISEASE_RULES["Rice"])
        for disease_name, risk_fn in diseases:
            risk = float(np.clip(risk_fn(row), 0, 100))
            records.append({
                "date":         row["date"],
                "region":       row["region"],
                "state":        row["state"],
                "crop":         crop,
                "disease":      disease_name,
                "risk_score":   round(risk, 1),
                "risk_level":   _classify(risk),
                "temp_avg":     row.get("temp_avg", 30),
                "humidity":     row.get("humidity", 70),
                "rainfall":     row.get("rainfall", 5),
            })
    return pd.DataFrame(records)


def get_latest_risks(risk_df: pd.DataFrame) -> pd.DataFrame:
    """Returns the most recent risk score per region+disease."""
    return (
        risk_df.sort_values("date")
               .groupby(["region", "disease", "crop"])
               .last()
               .reset_index()
               .sort_values("risk_score", ascending=False)
    )


def compute_region_summary(risk_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate max risk per region for map display."""
    latest = get_latest_risks(risk_df)
    return (
        latest.groupby(["region", "state", "crop"])
              .agg(max_risk=("risk_score", "max"),
                   avg_risk=("risk_score", "mean"),
                   diseases=("disease", lambda x: ", ".join(x.tolist())))
              .reset_index()
              .sort_values("max_risk", ascending=False)
    )


# ─── Individual disease risk functions ───────────────────────────────────────
def _blast_risk(r):
    # Rice blast thrives: temp 24-28°C, humidity >90%, rainfall 5-30mm
    t, h, rain = r.get("temp_avg",30), r.get("humidity",70), r.get("rainfall",5)
    score = 0
    score += 40 * _bell(t, 26, 4)
    score += 35 * _sigmoid(h, 85, 5)
    score += 25 * _bell(rain, 15, 10)
    return score

def _bph_risk(r):
    t, h = r.get("temp_avg",30), r.get("humidity",70)
    return 30 * _bell(t, 29, 3) + 40 * _sigmoid(h, 80, 8) + 30 * _noise(r)

def _sheath_blight_risk(r):
    t, h = r.get("temp_avg",30), r.get("humidity",70)
    return 50 * _bell(t, 30, 5) + 50 * _sigmoid(h, 88, 6)

def _rust_risk(r):
    t, h = r.get("temp_avg",30), r.get("humidity",70)
    return 60 * _bell(t, 20, 6) + 40 * _sigmoid(h, 75, 8)

def _stripe_rust_risk(r):
    t, h = r.get("temp_avg",30), r.get("humidity",70)
    return 70 * _bell(t, 15, 5) + 30 * _sigmoid(h, 80, 6)

def _powdery_mildew_risk(r):
    t, h = r.get("temp_avg",30), r.get("humidity",70)
    return 50 * _bell(t, 22, 5) + 30 * _bell(h, 65, 15) + 20 * _noise(r)

def _downy_mildew_risk(r):
    h, rain = r.get("humidity",70), r.get("rainfall",5)
    return 60 * _sigmoid(h, 85, 5) + 40 * _sigmoid(rain, 10, 5)

def _virus_risk(r):
    t = r.get("temp_avg",30)
    return 70 * _bell(t, 28, 6) + 30 * _noise(r)

def _bacterial_blight_risk(r):
    t, h = r.get("temp_avg",30), r.get("humidity",70)
    return 55 * _bell(t, 30, 5) + 45 * _sigmoid(h, 80, 6)

def _root_rot_risk(r):
    rain = r.get("rainfall", 5)
    return 80 * _sigmoid(rain, 20, 5) + 20 * _noise(r)

def _leaf_blight_risk(r):
    t, h = r.get("temp_avg",30), r.get("humidity",70)
    return 50 * _bell(t, 25, 5) + 50 * _sigmoid(h, 80, 6)

def _stalk_rot_risk(r):
    rain = r.get("rainfall", 5)
    t    = r.get("temp_avg", 30)
    return 40 * _sigmoid(rain, 15, 5) + 60 * _bell(t, 28, 6)

def _red_rot_risk(r):
    h, rain = r.get("humidity",70), r.get("rainfall",5)
    return 50 * _sigmoid(h, 82, 5) + 50 * _sigmoid(rain, 12, 4)

def _smut_risk(r):
    t = r.get("temp_avg", 30)
    return 80 * _bell(t, 27, 5) + 20 * _noise(r)


# ─── Helper math ─────────────────────────────────────────────────────────────
def _bell(x, center, width):
    """Gaussian bell — peaks at center, decays with width."""
    return float(np.exp(-0.5 * ((x - center) / width) ** 2))

def _sigmoid(x, threshold, steepness):
    """Logistic — rises steeply around threshold."""
    return float(1 / (1 + np.exp(-(x - threshold) / steepness)))

def _noise(r):
    """Tiny pseudo-random noise to avoid flat zero scores."""
    return float(np.random.default_rng(int(r.get("temp_avg",30)*100) % (2**32)).uniform(0.1, 0.3))

def _classify(score: float) -> str:
    if score >= 70: return "Critical"
    if score >= 45: return "High"
    if score >= 25: return "Moderate"
    return "Low"
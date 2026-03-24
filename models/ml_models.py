"""
models/ml_models.py
Two models:
  1. DiseaseClassifier  – Random Forest predicting disease risk level
  2. StockImpactModel   – Gradient Boosting predicting % stock change
     given outbreak severity and climate features.
Both train on synthetic data if no saved model exists.
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error

MODEL_DIR = os.path.join(os.path.dirname(__file__))
DISEASE_MODEL_PATH = os.path.join(MODEL_DIR, "disease_model.pkl")
STOCK_MODEL_PATH   = os.path.join(MODEL_DIR, "stock_model.pkl")
ENCODER_PATH       = os.path.join(MODEL_DIR, "label_encoder.pkl")


# ─── Feature columns ─────────────────────────────────────────────────────────
CLIMATE_FEATURES = ["temp_avg", "temp_max", "temp_min", "humidity", "rainfall", "wind_speed"]
STOCK_FEATURES   = ["risk_score", "avg_humidity", "avg_temp", "avg_rainfall",
                    "days_above_30", "days_high_humidity", "rainfall_deficit"]


# ════════════════════════════════════════════════════════════════════════════
# 1. Disease Risk Classifier
# ════════════════════════════════════════════════════════════════════════════

class DiseaseClassifier:
    def __init__(self):
        self.model   = None
        self.encoder = LabelEncoder()

    def _generate_training_data(self, n: int = 5000) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        rows = []
        for _ in range(n):
            temp_avg   = rng.uniform(15, 40)
            humidity   = rng.uniform(30, 98)
            rainfall   = rng.exponential(8)
            temp_max   = temp_avg + rng.uniform(2, 6)
            temp_min   = temp_avg - rng.uniform(3, 8)
            wind_speed = rng.uniform(2, 30)

            # Synthetic label logic (mirrors disease_engine thresholds)
            score = 0
            if 23 <= temp_avg <= 29 and humidity > 85: score += 45
            if humidity > 90:                           score += 25
            if 5 < rainfall < 30:                      score += 15
            if temp_avg > 32 and humidity > 80:         score += 20
            score += rng.uniform(-10, 10)
            score = float(np.clip(score, 0, 100))

            if   score >= 70: label = "Critical"
            elif score >= 45: label = "High"
            elif score >= 25: label = "Moderate"
            else:             label = "Low"

            rows.append([temp_avg, temp_max, temp_min, humidity, rainfall, wind_speed, label])

        return pd.DataFrame(rows, columns=CLIMATE_FEATURES + ["risk_level"])

    def train(self, df: pd.DataFrame = None, verbose: bool = True):
        if df is None:
            df = self._generate_training_data()

        X = df[CLIMATE_FEATURES].values
        y = self.encoder.fit_transform(df["risk_level"].values)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)

        if verbose:
            preds = self.model.predict(X_test)
            print("\n── Disease Classifier Report ──────────────────")
            print(classification_report(y_test, preds,
                  target_names=self.encoder.classes_))

        self.save()
        return self

    def predict(self, climate_row: dict) -> dict:
        """Predict risk level and probabilities for a single observation."""
        if self.model is None:
            self.load()
        X = np.array([[climate_row.get(f, 0) for f in CLIMATE_FEATURES]])
        proba  = self.model.predict_proba(X)[0]
        idx    = np.argmax(proba)
        label  = self.encoder.inverse_transform([idx])[0]
        return {
            "risk_level":   label,
            "confidence":   round(float(proba[idx]) * 100, 1),
            "probabilities": dict(zip(self.encoder.classes_, np.round(proba * 100, 1).tolist())),
        }

    def batch_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            self.load()
        X      = df[CLIMATE_FEATURES].fillna(df[CLIMATE_FEATURES].mean()).values
        preds  = self.encoder.inverse_transform(self.model.predict(X))
        probas = self.model.predict_proba(X).max(axis=1) * 100
        return df.assign(predicted_risk=preds, confidence=np.round(probas, 1))

    def save(self):
        joblib.dump(self.model,   DISEASE_MODEL_PATH)
        joblib.dump(self.encoder, ENCODER_PATH)

    def load(self):
        if os.path.exists(DISEASE_MODEL_PATH):
            self.model   = joblib.load(DISEASE_MODEL_PATH)
            self.encoder = joblib.load(ENCODER_PATH)
        else:
            self.train(verbose=False)


# ════════════════════════════════════════════════════════════════════════════
# 2. Stock Impact Regressor
# ════════════════════════════════════════════════════════════════════════════

class StockImpactModel:
    def __init__(self):
        self.model = None

    def _generate_training_data(self, n: int = 3000) -> pd.DataFrame:
        rng = np.random.default_rng(99)
        risk_score         = rng.uniform(0, 100, n)
        avg_humidity       = rng.uniform(40, 95, n)
        avg_temp           = rng.uniform(18, 40, n)
        avg_rainfall       = rng.exponential(8, n)
        days_above_30      = rng.integers(0, 30, n).astype(float)
        days_high_humidity = rng.integers(0, 30, n).astype(float)
        rainfall_deficit   = rng.uniform(-50, 50, n)

        # Target: % stock change — higher risk → higher demand → positive returns
        target = (
            0.04  * risk_score
            + 0.02  * days_high_humidity
            - 0.015 * rainfall_deficit
            + 0.01  * (avg_temp - 28)
            + rng.normal(0, 1.5, n)
        )
        return pd.DataFrame({
            "risk_score":          risk_score,
            "avg_humidity":        avg_humidity,
            "avg_temp":            avg_temp,
            "avg_rainfall":        avg_rainfall,
            "days_above_30":       days_above_30,
            "days_high_humidity":  days_high_humidity,
            "rainfall_deficit":    rainfall_deficit,
            "stock_pct_change":    target,
        })

    def train(self, df: pd.DataFrame = None, verbose: bool = True):
        if df is None:
            df = self._generate_training_data()

        X = df[STOCK_FEATURES].values
        y = df["stock_pct_change"].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=5,
            random_state=42,
        )
        self.model.fit(X_train, y_train)

        if verbose:
            preds = self.model.predict(X_test)
            mae   = mean_absolute_error(y_test, preds)
            print(f"\n── Stock Impact Regressor ─────────────────────")
            print(f"   MAE: {mae:.3f}%   (mean absolute error on held-out set)")

        self.save()
        return self

    def predict(self, features: dict) -> float:
        """Returns predicted % change for a stock given disease/climate context."""
        if self.model is None:
            self.load()
        X = np.array([[features.get(f, 0) for f in STOCK_FEATURES]])
        return round(float(self.model.predict(X)[0]), 2)

    def feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            self.load()
        return (pd.DataFrame({
                "feature":    STOCK_FEATURES,
                "importance": self.model.feature_importances_,
               })
               .sort_values("importance", ascending=False)
               .reset_index(drop=True))

    def save(self):
        joblib.dump(self.model, STOCK_MODEL_PATH)

    def load(self):
        if os.path.exists(STOCK_MODEL_PATH):
            self.model = joblib.load(STOCK_MODEL_PATH)
        else:
            self.train(verbose=False)


# ─── Convenience: train both ─────────────────────────────────────────────────
def train_all_models():
    print("Training Disease Classifier...")
    DiseaseClassifier().train()
    print("\nTraining Stock Impact Model...")
    StockImpactModel().train()
    print("\n✓ All models trained and saved.")


if __name__ == "__main__":
    train_all_models()
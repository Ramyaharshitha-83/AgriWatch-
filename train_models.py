"""
train_models.py
Run once to pre-train and save ML models before starting the dashboard.
Usage:  python train_models.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from models.ml_models import train_all_models

if __name__ == "__main__":
    print("=" * 50)
    print("  AgriWatch — Training ML Models")
    print("=" * 50)
    train_all_models()
    print("\n✅ Models saved to models/ directory.")
    print("   Now run:  streamlit run app.py")
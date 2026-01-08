import sys

print("Checking dependencies...")
try:
    import xgboost
    print("XGBoost: INSTALLED")
    print(f"  Version: {xgboost.__version__}")
except ImportError:
    print("XGBoost: MISSING")

try:
    import shap
    print("SHAP: INSTALLED")
    print(f"  Version: {shap.__version__}")
except ImportError:
    print("SHAP: MISSING")

try:
    import matplotlib
    print("Matplotlib: INSTALLED")
    print(f"  Version: {matplotlib.__version__}")
except ImportError:
    print("Matplotlib: MISSING")

import xgboost as xgb

# Load your old model
booster = xgb.Booster()
booster.load_model("fraud_model_best.json")

# Re-save it using the current XGBoost version
booster.save_model("fraud_model_cleaned.json")
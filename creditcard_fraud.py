import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, classification_report
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("creditcard.csv")

# Add more realistic noise to make separation harder
for v in [f"V{i}" for i in range(1,29)]:
    df[v] += np.random.normal(0, 0.05, size=df.shape[0])
df['Amount'] += np.random.normal(0, 1.0, size=df.shape[0])
df['Time'] += np.random.normal(0, 3.0, size=df.shape[0])

# Scale Time and Amount
scaler_time = StandardScaler()
scaler_amount = StandardScaler()
df['scaled_time'] = scaler_time.fit_transform(df[['Time']])
df['scaled_amount'] = scaler_amount.fit_transform(df[['Amount']])

# Engineered features
df['amount_per_time'] = df['Amount'] / (df['Time'] + 1e-6)
df['v_mean'] = df[[f"V{i}" for i in range(1,29)]].mean(axis=1)
df['v_std'] = df[[f"V{i}" for i in range(1,29)]].std(axis=1)

# Drop raw columns
df.drop(['Time','Amount'], axis=1, inplace=True)

# Features & target
feature_cols = [f"V{i}" for i in range(1,29)] + ['scaled_time','scaled_amount','amount_per_time','v_mean','v_std']
X = df[feature_cols]
y = df['Class']

# Train/validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# XGBoost with lower complexity to prevent perfect separation
model = XGBClassifier(
    n_estimators=35,
    max_depth=1,
    learning_rate=0.1,
    subsample=0.5,
    colsample_bytree=0.5,
    reg_alpha=0.5,
    reg_lambda=0.5,
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)
model.fit(X_train, y_train)

# Predict
y_probs = model.predict_proba(X_valid)[:,1]

# Threshold tuning in narrower range
thresholds = np.arange(0.45, 0.55, 0.01)
f1_scores_fraud = [f1_score(y_valid, (y_probs > t).astype(int), pos_label=1) for t in thresholds]
best_threshold = thresholds[np.argmax(f1_scores_fraud)]
y_pred = (y_probs > best_threshold).astype(int)

# Metrics
f1_fraud = f1_score(y_valid, y_pred, pos_label=1)
f1_legit = f1_score(y_valid, y_pred, pos_label=0)
f1_macro = f1_score(y_valid, y_pred, average='macro')
roc_auc = roc_auc_score(y_valid, y_probs)
pr_auc = average_precision_score(y_valid, y_probs)

print(f"Best Threshold: {best_threshold:.2f}")
print(f"F1 Legit: {f1_legit:.4f}")
print(f"F1 Fraud: {f1_fraud:.4f}")
print(f"F1 Macro: {f1_macro:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")
print(classification_report(y_valid, y_pred, digits=4))

# Save model
model.save_model("fraud_model_best.json")

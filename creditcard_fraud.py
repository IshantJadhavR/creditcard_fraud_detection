import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    f1_score, classification_report, precision_recall_curve,
    auc, roc_auc_score, roc_curve, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
import joblib

pd.set_option("display.max_columns", None)
sns.set(style="whitegrid")

# ===== Load & preprocess =====
data_path = r"C:\Users\purab\OneDrive\Desktop\creditcard\creditcard.csv"
df = pd.read_csv(data_path)

df['scaled_time'] = StandardScaler().fit_transform(df[['Time']])
df['scaled_amount'] = StandardScaler().fit_transform(df[['Amount']])
df.drop(['Amount', 'Time'], axis=1, inplace=True)

expected_order = [f"V{i}" for i in range(1, 29)] + ['scaled_time', 'scaled_amount', 'Class']
df = df.reindex(columns=expected_order)

X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Balance training set  (1 fraud : 5 nonfraud)
train_df = pd.concat([X_train, y_train], axis=1)
fraud_train = train_df[train_df['Class'] == 1]
non_fraud_train = train_df[train_df['Class'] == 0].sample(
    n=len(fraud_train) * 5, random_state=42
)
balanced_train_df = pd.concat([fraud_train, non_fraud_train]).sample(
    frac=1, random_state=42
).reset_index(drop=True)

X_train_bal = balanced_train_df.drop('Class', axis=1)
y_train_bal = balanced_train_df['Class']

# ===== Hyper-param search =====
scale_pos_weight = len(non_fraud_train) / len(fraud_train)

xgb = XGBClassifier(
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)

param_dist = {
    'max_depth': [4, 5, 6, 7, 8],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'n_estimators': [300, 500, 700, 900],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

search = RandomizedSearchCV(
    xgb,
    param_distributions=param_dist,
    n_iter=15,
    scoring='f1',   # F1 for fraud class during CV
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)
search.fit(X_train_bal, y_train_bal)

print("\nBest params from search:\n", search.best_params_)
model = search.best_estimator_

# ===== Predictions & threshold selection =====
y_pred_prob = model.predict_proba(X_test)[:, 1]

# pick threshold that maximises F1 for LEGIT class (pos_label=0)
thresholds = np.arange(0.01, 0.5, 0.01)
f1_scores_legit = [f1_score(y_test, (y_pred_prob > t).astype(int), pos_label=0)
                   for t in thresholds]
best_idx = np.argmax(f1_scores_legit)
best_threshold = thresholds[best_idx]
y_pred = (y_pred_prob > best_threshold).astype(int)

precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(recall, precision)
roc_auc = roc_auc_score(y_test, y_pred_prob)

f1_fraud = f1_score(y_test, y_pred, pos_label=1)
f1_legit = f1_score(y_test, y_pred, pos_label=0)
f1_macro = f1_score(y_test, y_pred, average='macro')

print(f"\nThreshold chosen (maximising F1 for legit): {best_threshold:.2f}")
print(f"F1 Legit (Class=0): {f1_legit:.4f}")
print(f"F1 Fraud (Class=1): {f1_fraud:.4f}")
print(f"F1 Macro Average:   {f1_macro:.4f}")
print(f"PR-AUC:             {pr_auc:.4f}")
print(f"ROC-AUC:            {roc_auc:.4f}\n")
print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))

# ===== Save model =====
model.save_model("fraud_model_best.json")
print("✅ Saved best model to fraud_model_best.json")

# ===== Optional plots =====
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve (PR-AUC = {pr_auc:.4f})')
plt.tight_layout()
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'ROC-AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.show()

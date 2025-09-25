import streamlit as st
import pandas as pd
import numpy as np
import random
from xgboost import Booster, DMatrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_recall_curve, auc, roc_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="💳 Credit Card Fraud Detection", page_icon="💳", layout="wide")
st.title("💳 Credit Card Fraud Detection App")
st.markdown("""
**Features:**  
- Batch CSV prediction  
- Single transaction prediction  
- Random row prediction  
- Metrics with confusion matrix, PR & ROC curves
""")

# Load trained model
booster = Booster()
booster.load_model("fraud_model_best.json")

# Columns used during training
cols_order = [f"V{i}" for i in range(1, 29)] + [
    'scaled_time', 'scaled_amount', 'amount_per_time', 'v_mean', 'v_std'
]

# Feature preparation
def prepare_features(df):
    scaler_time = StandardScaler().fit(df[['Time']])
    scaler_amount = StandardScaler().fit(df[['Amount']])
    df['scaled_time'] = scaler_time.transform(df[['Time']])
    df['scaled_amount'] = scaler_amount.transform(df[['Amount']])
    df['amount_per_time'] = df['Amount'] / (df['Time'] + 1e-6)
    df['v_mean'] = df[[f"V{i}" for i in range(1, 29)]].mean(axis=1)
    df['v_std'] = df[[f"V{i}" for i in range(1, 29)]].std(axis=1)
    df = df.drop(columns=['Time', 'Amount'])
    return df[cols_order]

# Batch prediction
st.header("Batch Prediction from CSV")
uploaded_file = st.file_uploader("Upload CSV with transactions", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df_for_metrics = df.copy()
        df_model = prepare_features(df)

        st.subheader("Prepared Data Preview")
        st.dataframe(df_model.head())

        dmatrix = DMatrix(df_model)
        probs = booster.predict(dmatrix)
        preds = (probs > 0.5).astype(int)

        out = pd.DataFrame({
            "Prediction": preds,
            "Fraud Probability": probs
        })
        st.subheader("Predictions (0 = Legit, 1 = Fraud)")
        st.dataframe(out)

        # Metrics
        if 'Class' in df_for_metrics.columns:
            true_labels = df_for_metrics['Class']
            f1_fraud = f1_score(true_labels, preds, pos_label=1)
            f1_legit = f1_score(true_labels, preds, pos_label=0)
            precision, recall, _ = precision_recall_curve(true_labels, probs)
            pr_auc = auc(recall, precision)
            fpr, tpr, _ = roc_curve(true_labels, probs)
            roc_auc = roc_auc_score(true_labels, probs)

            st.header("Batch Prediction Metrics")
            st.metric("F1 Score (Fraud)", f"{f1_fraud:.4f}")
            st.metric("F1 Score (Legit)", f"{f1_legit:.4f}")
            st.metric("PR-AUC", f"{pr_auc:.4f}")
            st.metric("ROC-AUC", f"{roc_auc:.4f}")

            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(confusion_matrix(true_labels, preds), annot=True, fmt='d', cmap='Blues', ax=ax_cm)
            ax_cm.set_xlabel('Predicted')
            ax_cm.set_ylabel('Actual')
            ax_cm.set_title('Confusion Matrix')
            st.pyplot(fig_cm)

            fig_pr, ax_pr = plt.subplots()
            ax_pr.plot(recall, precision, marker='.')
            ax_pr.set_xlabel('Recall')
            ax_pr.set_ylabel('Precision')
            ax_pr.set_title(f'Precision-Recall Curve (PR-AUC = {pr_auc:.4f})')
            st.pyplot(fig_pr)

            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, label=f'ROC-AUC = {roc_auc:.4f}')
            ax_roc.plot([0,1],[0,1], linestyle='--', color='gray')
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title('ROC Curve')
            ax_roc.legend()
            st.pyplot(fig_roc)

        # Row prediction
        st.header("Select a Row for Prediction")
        row_idx = st.number_input("Row index (0-based)", min_value=0, max_value=len(df_model)-1, value=0)
        if st.button("Predict Selected Row"):
            selected_row = df_model.iloc[[row_idx]]
            dmatrix_row = DMatrix(selected_row)
            prob_row = booster.predict(dmatrix_row)[0]
            pred_row = int(prob_row > 0.5)
            st.success(f"Row {row_idx} Prediction: {'Fraud' if pred_row==1 else 'Legit'} (Fraud Probability: {prob_row:.2%})")

        # Random row prediction
        st.header("Random Row Prediction")
        if st.button("Predict Random Row"):
            rand_idx = random.randint(0, len(df_model)-1)
            random_row = df_model.iloc[[rand_idx]]
            dmatrix_rand = DMatrix(random_row)
            prob_rand = booster.predict(dmatrix_rand)[0]
            pred_rand = int(prob_rand > 0.5)
            st.info(f"Random Row Index: {rand_idx}")
            st.success(f"Random Row Prediction: {'Fraud' if pred_rand==1 else 'Legit'} (Fraud Probability: {prob_rand:.2%})")

    except Exception as e:
        st.error(f"Error reading CSV file: {e}")

# Manual prediction
st.header("Predict a Single Transaction (Manual Input)")
input_dict = {f"V{i}": st.slider(f"V{i}", -30.0, 30.0, 0.0, 0.01) for i in range(1,29)}
time_val = st.number_input("Time", value=0.0, format="%.6f")
amount_val = st.number_input("Amount", value=0.0, format="%.6f")

if st.button("Predict Manual Transaction"):
    single_df = pd.DataFrame([input_dict])
    single_df['scaled_time'] = StandardScaler().fit_transform([[time_val]])
    single_df['scaled_amount'] = StandardScaler().fit_transform([[amount_val]])
    single_df['amount_per_time'] = amount_val / (time_val + 1e-6)
    single_df['v_mean'] = single_df[[f"V{i}" for i in range(1,29)]].mean(axis=1)
    single_df['v_std'] = single_df[[f"V{i}" for i in range(1,29)]].std(axis=1)
    single_df = single_df[cols_order]

    dmatrix_single = DMatrix(single_df)
    prob = booster.predict(dmatrix_single)[0]
    pred = int(prob > 0.5)
    st.success(f"Manual Transaction Prediction: {'Fraud' if pred==1 else 'Legit'} (Fraud Probability: {prob:.2%})")

st.caption("Model: XGBoost Booster trained with engineered features and scaled Time & Amount.")

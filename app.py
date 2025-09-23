import streamlit as st
import pandas as pd
from xgboost import Booster, DMatrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_recall_curve,
    auc, roc_curve, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import zipfile

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳")
st.title("💳 Credit Card Fraud Detection App")

# --- Load trained model ---
booster = Booster()
booster.load_model("fraud_model_cleaned.json")

# Column order used during training
cols_order = [f"V{i}" for i in range(1, 29)] + ['scaled_time', 'scaled_amount']

# --- Upload ZIP for batch prediction ---
st.header("Batch Prediction from ZIP file")
uploaded_file = st.file_uploader("Upload a ZIP file containing your CSV", type=["zip"])

if uploaded_file is not None:
    try:
        with zipfile.ZipFile(uploaded_file, "r") as z:
            csv_filename = z.namelist()[0]
            with z.open(csv_filename) as f:
                df = pd.read_csv(f)

        df_for_metrics = df.copy()

        # Fit scalers on uploaded data
        scaler_time = StandardScaler().fit(df[['Time']])
        scaler_amount = StandardScaler().fit(df[['Amount']])

        if 'Class' in df.columns:
            df = df.drop(columns=['Class'])

        df['scaled_time'] = scaler_time.transform(df_for_metrics[['Time']])
        df['scaled_amount'] = scaler_amount.transform(df_for_metrics[['Amount']])
        df = df.drop(columns=['Time', 'Amount'])
        df = df.reindex(columns=cols_order)

        st.subheader("Preview of prepared data")
        st.dataframe(df.head())

        dmatrix = DMatrix(df)
        probs = booster.predict(dmatrix)
        preds = (probs > 0.5).astype(int)

        out = pd.DataFrame({
            "Prediction": preds,
            "Fraud Probability": probs
        })

        st.subheader("Predictions (0 = Legit, 1 = Fraud)")
        st.dataframe(out)

        # --- Metrics and Diagrams ---
        st.header("Batch Prediction Metrics & Diagrams")
        if 'Class' in df_for_metrics.columns:
            true_labels = df_for_metrics['Class']

            f1_legit = f1_score(true_labels, preds, pos_label=0)
            precision, recall, _ = precision_recall_curve(true_labels, probs)
            pr_auc = auc(recall, precision)
            fpr, tpr, _ = roc_curve(true_labels, probs)
            roc_auc = roc_auc_score(true_labels, probs)

            st.metric("F1 Score", f"{f1_legit:.4f}")
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
            ax_roc.plot([0,1], [0,1], linestyle='--', color='gray')
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate (Recall)')
            ax_roc.set_title('ROC Curve')
            ax_roc.legend()
            st.pyplot(fig_roc)

        # --- Row selection for single prediction ---
        st.header("Select a Row from Uploaded CSV for Prediction")
        row_idx = st.number_input("Select row index (0-based)", min_value=0, max_value=len(df)-1, value=0)
        if st.button("Predict Selected Row"):
            selected_row = df.iloc[[row_idx]]
            dmatrix_row = DMatrix(selected_row)
            prob_row = booster.predict(dmatrix_row)[0]
            pred_row = int(prob_row > 0.5)
            st.success(f"Selected Row Prediction: {'Fraud' if pred_row==1 else 'Legit'} "
                       f"(Fraud probability: {prob_row:.2%})")

    except Exception as e:
        st.error(f"Error reading ZIP file: {e}")

# --- Single transaction input ---
st.header("Single Transaction Prediction (Manual Input)")

input_dict = {}
for col in [f"V{i}" for i in range(1, 29)]:
    input_dict[col] = st.slider(col, min_value=-30.0, max_value=30.0, value=0.0, step=0.01)

time_val = st.number_input("Time", value=0.0, format="%.6f")
amount_val = st.number_input("Amount", value=0.0, format="%.6f")

if st.button("Predict Single Transaction (Manual)"):
    single_df = pd.DataFrame([input_dict])
    single_df['scaled_time'] = scaler_time.transform([[time_val]])
    single_df['scaled_amount'] = scaler_amount.transform([[amount_val]])
    single_df = single_df[cols_order]

    dmatrix_single = DMatrix(single_df)
    prob = booster.predict(dmatrix_single)[0]
    pred = int(prob > 0.5)

    st.success(f"Prediction: {'Fraud' if pred==1 else 'Legit'} (Fraud probability: {prob:.2%})")

st.caption("Model: XGBoost Booster, trained with scaled Time & Amount. Metrics above show F1 for both classes.")

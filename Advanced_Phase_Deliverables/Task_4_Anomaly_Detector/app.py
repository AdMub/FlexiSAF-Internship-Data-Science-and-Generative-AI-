import streamlit as st
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_google_genai import ChatGoogleGenerativeAI

# --- LOAD ENVIRONMENT VARIABLES ---
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    st.error("API Key not found. Please ensure you have a .env file with GOOGLE_API_KEY set.")
    st.stop()

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Financial Anomaly Detector", page_icon="🚨", layout="wide")
st.title("🚨 School Financial Anomaly Detector")
st.markdown("Upload a financial CSV to use **Unsupervised Learning (Isolation Forest)** to flag suspicious fee payments and **Generative AI** to explain potential fraud risks.")

# --- FILE UPLOAD ---
st.sidebar.header("1. Upload Financial Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is None:
    st.info("👈 Please upload your financial dataset (CSV) in the sidebar to begin the audit.")
    st.stop()

# Load the uploaded file
df = pd.read_csv(uploaded_file)

# Check if required columns exist to prevent crashes
required_columns = ['Date', 'Transaction_ID', 'Student_ID', 'Amount', 'Method', 'Type']
if not all(col in df.columns for col in required_columns):
    st.error(f"⚠️ Uploaded CSV is missing required columns. It must contain: {', '.join(required_columns)}")
    st.stop()

# --- ANOMALY DETECTION (Machine Learning) ---
@st.cache_data
def detect_anomalies(data):
    # Train the Isolation Forest primarily on the 'Amount' feature
    X = data[['Amount']]
    
    # Initialize Isolation Forest (contamination = expected percentage of anomalies)
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    model.fit(X)
    
    # Predict (-1 is anomaly, 1 is normal)
    data['Anomaly_Flag'] = model.predict(X)
    data['Status'] = data['Anomaly_Flag'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')
    return data

with st.spinner("Running Unsupervised Machine Learning models..."):
    df_processed = detect_anomalies(df.copy())
    anomalies_only = df_processed[df_processed['Status'] == 'Anomaly']

# --- AI EXPLANATION GENERATOR ---
def generate_fraud_explanation(txn_id, amount, method, txn_type):
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
        prompt = f"""
        You are a Forensic Financial Auditor for a University. 
        The Machine Learning system flagged the following transaction as an anomaly:
        - Transaction ID: {txn_id}
        - Amount: ${amount}
        - Payment Method: {method}
        - Transaction Type: {txn_type}
        
        The average normal tuition fee is usually around $1,500 via Bank Transfer.
        
        Write a concise, 3-sentence professional audit note explaining exactly WHY this specific transaction was flagged as highly suspicious, and suggest one immediate investigative action the finance team should take.
        """
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        if "429" in str(e) or "Quota" in str(e):
            return "⚠️ **AI Auditor on Cooldown:** You have hit the free-tier API speed limit. The Machine Learning model successfully flagged this anomaly, but please wait about 60 seconds before asking the AI to generate the forensic report."
        else:
            return f"⚠️ **An error occurred:** {str(e)}"

# --- UI LAYOUT ---
# 1. KPIs
st.subheader("📊 Financial Integrity Dashboard")
col1, col2, col3 = st.columns(3)
col1.metric("Total Transactions Analyzed", len(df_processed))
col2.metric("Normal Transactions", len(df_processed[df_processed['Status'] == 'Normal']))
col3.metric("🚨 Suspicious Anomalies Flagged", len(anomalies_only))

st.markdown("---")

# 2. Data Visualization
st.subheader("📈 Transaction Distribution (Isolation Forest Results)")
fig, ax = plt.subplots(figsize=(10, 4))
sns.scatterplot(
    data=df_processed, 
    x=df_processed.index, 
    y='Amount', 
    hue='Status', 
    palette={'Normal': '#2ecc71', 'Anomaly': '#e74c3c'},
    alpha=0.7, 
    s=100,
    ax=ax
)
ax.set_title("Financial Transactions by Amount")
ax.set_xlabel("Transaction Index")
ax.set_ylabel("Transaction Amount ($)")
st.pyplot(fig)

# 3. Anomaly Investigation Table
st.markdown("---")
st.subheader("🕵️‍♂️ Flagged Transactions (Requires Review)")

if anomalies_only.empty:
    st.success("No anomalies detected! Financial records look clean.")
else:
    st.dataframe(anomalies_only[['Date', 'Transaction_ID', 'Student_ID', 'Type', 'Method', 'Amount']], use_container_width=True)
    
    st.subheader("🤖 AI Auditor Explanation")
    st.markdown("Select a flagged transaction to generate a forensic AI analysis.")
    
    # Dropdown to select an anomaly to explain
    selected_txn = st.selectbox("Select Transaction ID:", anomalies_only['Transaction_ID'].tolist())
    
    if st.button("Generate Forensic Analysis"):
        with st.spinner("AI Auditor is reviewing the transaction..."):
            txn_data = anomalies_only[anomalies_only['Transaction_ID'] == selected_txn].iloc[0]
            explanation = generate_fraud_explanation(
                txn_data['Transaction_ID'], 
                txn_data['Amount'], 
                txn_data['Method'], 
                txn_data['Type']
            )
            st.error(f"**Audit Note for {selected_txn}:**")
            st.write(explanation)
import streamlit as st
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_google_genai import ChatGoogleGenerativeAI

# --- LOAD ENVIRONMENT VARIABLES ---
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    st.error("API Key not found. Please ensure you have a .env file with GOOGLE_API_KEY set.")
    st.stop()

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Program Matcher", page_icon="🎯", layout="wide")
st.title("🎯 AI-Powered Program Recommendation Engine")
st.markdown("Matches applicants to university programs using **TF-IDF Cosine Similarity** and explains the fit using **Generative AI**.")

# --- LOAD DATA ---
try:
    df_applicants = pd.read_csv("applicants.csv")
    df_programs = pd.read_csv("programs.csv")
except FileNotFoundError:
    st.error("⚠️ Error: Could not find 'applicants.csv' or 'programs.csv'. Please make sure you saved both files in the same folder as this script!")
    st.stop()

# --- RECOMMENDATION ENGINE (ML) ---
@st.cache_data
def get_recommendations(applicant_idx, df_app, df_prog):
    # 1. Create content profiles
    applicant_profile = df_app.iloc[applicant_idx]['Academic_History'] + " " + df_app.iloc[applicant_idx]['Interests']
    program_profiles = df_prog['Program_Name'] + " " + df_prog['Description']
    
    # 2. TF-IDF Vectorization
    vectorizer = TfidfVectorizer(stop_words='english')
    all_profiles = [applicant_profile] + program_profiles.tolist()
    tfidf_matrix = vectorizer.fit_transform(all_profiles)
    
    # 3. Cosine Similarity Calculation
    applicant_vector = tfidf_matrix[0:1]
    program_vectors = tfidf_matrix[1:]
    similarity_scores = cosine_similarity(applicant_vector, program_vectors).flatten()
    
    # 4. Get Top 3
    top_3_indices = similarity_scores.argsort()[-3:][::-1]
    
    results = []
    for idx in top_3_indices:
        results.append({
            "Program_Name": df_prog.iloc[idx]['Program_Name'],
            "Description": df_prog.iloc[idx]['Description'],
            "Confidence_Score": similarity_scores[idx] * 100 # Convert to percentage
        })
    return results

# --- GENERATIVE AI EXPLANATION ---
def generate_explanation(applicant_name, app_profile, program_name, prog_desc, score):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    prompt = f"""
    You are an expert University Admissions Counselor. 
    Applicant Name: {applicant_name}
    Applicant Profile (History & Interests): {app_profile}
    
    Recommended Program: {program_name}
    Program Description: {prog_desc}
    Algorithm Match Score: {score:.1f}%
    
    Write a short, highly encouraging, and professional 1-paragraph explanation (max 4 sentences) detailing exactly WHY this applicant is a strong fit for this specific program based on their background and the program's requirements. Address the applicant directly using their name.
    """
    response = llm.invoke(prompt)
    return response.content

# --- UI LAYOUT ---
st.sidebar.header("Applicant Selection")
selected_applicant_name = st.sidebar.selectbox("Select an Applicant:", df_applicants['Name'].tolist())

# Find applicant index
applicant_idx = df_applicants[df_applicants['Name'] == selected_applicant_name].index[0]
applicant_data = df_applicants.iloc[applicant_idx]

st.subheader(f"👤 Applicant Profile: {applicant_data['Name']}")
st.write(f"**Academic History:** {applicant_data['Academic_History']}")
st.write(f"**Core Interests:** {applicant_data['Interests']}")
st.markdown("---")

st.subheader("📊 Top 3 Recommended Programs")
recommendations = get_recommendations(applicant_idx, df_applicants, df_programs)

# Visualize Confidence Levels using a Bar Chart
viz_data = pd.DataFrame({
    "Program": [rec['Program_Name'] for rec in recommendations],
    "Match Confidence (%)": [rec['Confidence_Score'] for rec in recommendations]
}).set_index("Program")
st.bar_chart(viz_data)

# Display Recommendation Details & AI Explanations
for i, rec in enumerate(recommendations):
    with st.expander(f"🏆 #{i+1} Match: {rec['Program_Name']} (Score: {rec['Confidence_Score']:.1f}%)", expanded=(i==0)):
        st.write(f"**Program Description:** {rec['Description']}")
        
        if st.button(f"Generate AI Explanation for {rec['Program_Name']}", key=f"btn_{i}"):
            with st.spinner("Analyzing fit using Gemini AI..."):
                app_profile = applicant_data['Academic_History'] + " " + applicant_data['Interests']
                explanation = generate_explanation(
                    applicant_data['Name'], app_profile, 
                    rec['Program_Name'], rec['Description'], 
                    rec['Confidence_Score']
                )
                st.success("AI Narrative Generated!")
                st.info(explanation)
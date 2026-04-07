import streamlit as st
import pandas as pd
import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from pypdf import PdfReader

# --- LOAD ENVIRONMENT VARIABLES ---
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    st.error("API Key not found. Please ensure you have a .env file with GOOGLE_API_KEY set.")
    st.stop()

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Quiz Generator", page_icon="📝", layout="wide")
st.title("📝 Automated AI Quiz Generator")
st.markdown("Upload your course notes (PDF or TXT), and the AI will automatically generate a structured multiple-choice quiz with answer keys and explanations.")

# --- HELPER FUNCTIONS ---
def extract_text_from_file(uploaded_file):
    """Extracts text from TXT or PDF files."""
    text = ""
    if uploaded_file.name.endswith(".txt"):
        text = uploaded_file.read().decode("utf-8")
    elif uploaded_file.name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def generate_quiz(text, num_questions, difficulty):
    """Uses Gemini to generate strict JSON quiz data with graceful error handling."""
    try:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
        
        prompt = f"""
        You are an expert AI educator creating a quiz for the Distinction.app platform. 
        Based ONLY on the following text, generate {num_questions} multiple-choice questions at a {difficulty} difficulty level.
        
        TEXT:
        {text[:15000]} # Limit to 15k characters to prevent context overload
        
        CRITICAL INSTRUCTION: You MUST respond ONLY with a valid JSON array. Do not include markdown formatting (like ```json), introduction text, or anything else. Just the raw JSON array.
        
        Format EXACTLY like this:
        [
          {{
            "question": "What is the capital of France?",
            "options": ["London", "Berlin", "Paris", "Madrid"],
            "correct_answer": "Paris",
            "explanation": "Paris is the capital and most populous city of France."
          }}
        ]
        """
        
        response = llm.invoke(prompt)
        raw_output = response.content.strip()
        
        # Clean up output just in case the LLM ignored instructions and added markdown
        if raw_output.startswith("```json"):
            raw_output = raw_output[7:]
        if raw_output.endswith("```"):
            raw_output = raw_output[:-3]
            
        return json.loads(raw_output)

    except Exception as e:
        error_msg = str(e)
        # Catch the quota limit and display a friendly UI message instead of crashing
        if "429" in error_msg or "Quota" in error_msg or "exhausted" in error_msg.lower():
            st.error("⚠️ **AI on Cooldown:** You have hit the free-tier API speed limit. Please wait about 60 seconds and click Generate again.")
        else:
            st.error(f"⚠️ **Generation Error:** {error_msg}")
        return None

# --- UI LAYOUT ---
# Sidebar Settings
st.sidebar.header("⚙️ Quiz Settings")
num_questions = st.sidebar.slider("Number of Questions", min_value=3, max_value=15, value=5)
difficulty = st.sidebar.selectbox("Difficulty Level", ["Beginner", "Intermediate", "Advanced"])

# File Uploader
st.subheader("1. Upload Course Material")
uploaded_file = st.file_uploader("Upload course notes (.txt or .pdf)", type=["txt", "pdf"])

if uploaded_file is not None:
    with st.expander("📄 View Extracted Text Preview"):
        document_text = extract_text_from_file(uploaded_file)
        st.text(document_text[:1000] + "...\n\n[Text truncated for preview]")
        
    if st.button("🚀 Generate AI Quiz"):
        with st.spinner(f"Generating {num_questions} {difficulty} questions... This may take a few seconds."):
            
            quiz_data = generate_quiz(document_text, num_questions, difficulty)
            
            if quiz_data:
                st.success("✅ Quiz successfully generated!")
                
                # Convert to Pandas DataFrame for easy exporting
                df_quiz = pd.DataFrame(quiz_data)
                
                # --- DISPLAY QUIZ IN UI ---
                st.markdown("---")
                st.subheader("🎯 Interactive Quiz Preview")
                
                for i, q in enumerate(quiz_data):
                    st.markdown(f"**Q{i+1}: {q['question']}**")
                    # Display options as a static radio button
                    st.radio("Options", q['options'], key=f"q_{i}")
                    
                    with st.expander("Show Answer & Explanation"):
                        st.success(f"**Correct Answer:** {q['correct_answer']}")
                        st.info(f"**Explanation:** {q['explanation']}")
                    st.write("") # Spacer
                
                # --- EXPORT DATA ---
                st.markdown("---")
                st.subheader("💾 Export Quiz Data")
                st.markdown("Download the generated questions, options, and answers for use in LMS platforms or databases.")
                
                col1, col2 = st.columns(2)
                
                # CSV Export
                csv_data = df_quiz.to_csv(index=False).encode('utf-8')
                col1.download_button(
                    label="📥 Download as CSV",
                    data=csv_data,
                    file_name="generated_quiz.csv",
                    mime="text/csv"
                )
                
                # JSON Export
                json_data = json.dumps(quiz_data, indent=4).encode('utf-8')
                col2.download_button(
                    label="📥 Download as JSON",
                    data=json_data,
                    file_name="generated_quiz.json",
                    mime="application/json"
                )
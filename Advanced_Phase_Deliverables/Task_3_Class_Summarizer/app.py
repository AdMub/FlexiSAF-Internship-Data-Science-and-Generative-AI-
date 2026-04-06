import streamlit as st
import pandas as pd
from keybert import KeyBERT
from transformers import pipeline
from fpdf import FPDF
import tempfile
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Intelligent Class Summarizer", page_icon="📝", layout="wide")
st.title("📝 Intelligent Class Summarizer")
st.markdown("Upload a virtual lesson transcript or chat log. The AI will extract topics, summarize the lesson, and generate actionable next steps using Hugging Face Transformers.")

# --- CACHE LOCAL AI MODELS (Hugging Face) ---
@st.cache_resource
def load_ai_models():
    # KeyBERT for Topic Extraction
    kw_model = KeyBERT()
    # DistilBART for fast, high-quality text summarization
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    # Flan-T5 for generating insights (Next Steps/Assignments)
    insight_generator = pipeline("text2text-generation", model="google/flan-t5-base")
    
    return kw_model, summarizer, insight_generator

with st.spinner("Loading Hugging Face Models... (This takes a moment on the first run)"):
    kw_model, summarizer, insight_generator = load_ai_models()

# --- PDF GENERATOR CLASS ---
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Intelligent Class Summary Report', 0, 1, 'C')
        self.ln(5)

def create_pdf(topics, summary, insights):
    pdf = PDFReport()
    pdf.add_page()
    
    # Topics
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "1. Key Topics Covered:", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 8, txt=", ".join(topics))
    pdf.ln(5)
    
    # Summary
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "2. Lesson Summary:", ln=True)
    pdf.set_font("Arial", size=11)
    # Clean encoding for FPDF
    clean_summary = summary.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 8, txt=clean_summary)
    pdf.ln(5)
    
    # Insights
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "3. Actionable Insights & Next Steps:", ln=True)
    pdf.set_font("Arial", size=11)
    clean_insights = insights.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 8, txt=clean_insights)
    
    pdf_file_name = "Class_Summary_Report.pdf"
    pdf.output(pdf_file_name)
    return pdf_file_name

# --- MAIN UI ---
st.sidebar.header("1. Upload Transcript")
uploaded_file = st.sidebar.file_uploader("Upload a .txt or .csv file", type=["txt", "csv"])

if uploaded_file is not None:
    # Read File
    if uploaded_file.name.endswith(".txt"):
        transcript_text = uploaded_file.read().decode("utf-8")
    elif uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        # Assuming the CSV has a column with text, we join it. 
        # (Grabs the last column by default assuming it's the message/text column)
        text_col = df.columns[-1]
        transcript_text = " ".join(df[text_col].astype(str).tolist())
    
    st.subheader("Raw Transcript Preview")
    with st.expander("Click to view raw text"):
        st.text(transcript_text[:1000] + "...\n[Text Truncated for Preview]")

    if st.button("🚀 Generate Intelligent Summary"):
        # HF models have token limits (~1024). We truncate the text for the local prototype to prevent crashing.
        safe_text = transcript_text[:3000] 

        col1, col2 = st.columns(2)
        
        with col1:
            with st.spinner("Extracting Key Topics using KeyBERT..."):
                keywords = kw_model.extract_keywords(safe_text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
                topic_list = [kw[0].title() for kw in keywords]
                st.success("✅ Topics Extracted")
                st.write("**Key Topics:**", ", ".join(topic_list))
                
            with st.spinner("Analyzing Next Steps & Assignments..."):
                prompt = f"Identify the specific assignments, homework, and next steps mentioned in this lesson: {safe_text}"
                insights_out = insight_generator(prompt, max_length=150)[0]['generated_text']
                st.success("✅ Insights Generated")
                st.write("**Actionable Insights:**")
                st.info(insights_out)
                
        with col2:
            with st.spinner("Generating Transformer Summary..."):
                summary_out = summarizer(safe_text, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
                st.success("✅ Summary Generated")
                st.write("**Lesson Summary:**")
                st.write(summary_out)
                
        # Generate PDF
        st.markdown("---")
        st.subheader("📥 Export Report")
        pdf_path = create_pdf(topic_list, summary_out, insights_out)
        
        with open(pdf_path, "rb") as pdf_file:
            st.download_button(
                label="📄 Download Structured PDF Report",
                data=pdf_file.read(),
                file_name="Class_Summary_Report.pdf",
                mime="application/pdf"
            )
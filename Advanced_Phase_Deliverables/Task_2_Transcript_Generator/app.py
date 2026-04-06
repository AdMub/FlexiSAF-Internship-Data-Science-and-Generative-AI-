import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image
import pandas as pd
from fpdf import FPDF
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import base64

# --- LOAD ENVIRONMENT VARIABLES ---
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    st.error("API Key not found. Please ensure you have a .env file with GOOGLE_API_KEY set.")
    st.stop()

# --- CONFIGURE TESSERACT PATH (WINDOWS SPECIFIC) ---
# Update this path if you installed Tesseract somewhere else
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Automated Transcript Engine", page_icon="🎓", layout="wide")
st.title("🎓 Automated Transcript Processing Engine")
st.markdown("Upload a scanned transcript image to extract data, generate an AI summary, and export a verified PDF.")

# --- FUNCTIONS ---
def preprocess_image_for_ocr(image):
    """Use OpenCV to convert the image to grayscale and apply thresholding for better OCR."""
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    # Apply adaptive thresholding to make text pop
    processed_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return processed_img

def generate_academic_summary(extracted_text):
    """Use Gemini LLM to analyze the OCR text and generate a professional summary."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    prompt = f"""
    You are an Academic Records Officer. Review the following extracted text from a student's transcript:
    
    {extracted_text}
    
    Please provide a concise, professional 3-paragraph summary of their academic standing, identifying their strongest subjects, and concluding with a general assessment of their performance. If the text is messy due to OCR errors, do your best to infer the correct academic context. Do not use asterisks or markdown formatting in your response.
    """
    response = llm.invoke(prompt)
    return response.content

class PDF(FPDF):
    """Custom FPDF class to add a watermark."""
    def header(self):
        self.set_font('Arial', 'B', 50)
        self.set_text_color(240, 240, 240) # Very light gray for watermark
        # Calculate width to center the watermark
        w = self.get_string_width('OFFICIAL TRANSCRIPT') + 6
        self.set_x((210 - w) / 2)
        self.set_y(100)
        # Rotate and place watermark (simulated by placing it large in the center)
        self.cell(w, 10, 'OFFICIAL TRANSCRIPT', 0, 0, 'C')
        self.set_y(20) # Reset Y for actual text

def create_pdf(ocr_text, ai_summary):
    """Generate the final watermarked PDF."""
    pdf = PDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(200, 10, txt="Automated Academic Transcript Report", ln=True, align='C')
    pdf.ln(10)
    
    # AI Summary Section
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="1. AI Academic Assessment Summary:", ln=True)
    pdf.set_font("Arial", size=11)
    # Clean unicode characters that might break FPDF
    clean_summary = ai_summary.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 8, txt=clean_summary)
    pdf.ln(10)
    
    # Raw OCR Section
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="2. Extracted Raw Data (OCR):", ln=True)
    pdf.set_font("Courier", size=9)
    clean_ocr = ocr_text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 5, txt=clean_ocr)
    
    # Output to file
    pdf_file_name = "Verified_Transcript.pdf"
    pdf.output(pdf_file_name)
    return pdf_file_name

# --- MAIN UI ---
uploaded_file = st.file_uploader("Upload Scanned Transcript (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("Processing Pipeline")
        
        with st.spinner("Step 1: Applying OpenCV Preprocessing & Tesseract OCR..."):
            processed_cv_img = preprocess_image_for_ocr(image)
            extracted_text = pytesseract.image_to_string(processed_cv_img)
            st.success("✅ OCR Extraction Complete")
            with st.expander("View Raw Extracted Text"):
                st.text(extracted_text)
                
        with st.spinner("Step 2: Generating NLP Academic Summary using Gemini..."):
            try:
                ai_summary = generate_academic_summary(extracted_text)
                st.success("✅ NLP Summarization Complete")
                st.info(ai_summary)
            except Exception as e:
                st.error(f"Error communicating with Gemini: {e}")
                st.stop()
                
        with st.spinner("Step 3: Generating Watermarked PDF..."):
            pdf_path = create_pdf(extracted_text, ai_summary)
            st.success("✅ PDF Generation Complete")
            
            # Provide Download Button
            with open(pdf_path, "rb") as pdf_file:
                pdf_bytes = pdf_file.read()
                st.download_button(
                    label="📄 Download Official Transcript PDF",
                    data=pdf_bytes,
                    file_name="Verified_Transcript_Report.pdf",
                    mime="application/pdf"
                )
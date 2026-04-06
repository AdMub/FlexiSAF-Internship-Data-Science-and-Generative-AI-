import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import tempfile

# --- LOAD ENVIRONMENT VARIABLES ---
load_dotenv() # This reads your .env file automatically!
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    st.error("API Key not found. Please ensure you have a .env file with GOOGLE_API_KEY set.")
    st.stop()

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Distinction AI Tutor", page_icon="📚", layout="wide")
st.title("📚 Distinction AI Tutor (Powered by Gemini)")
st.markdown("Upload your study notes (PDF) and ask questions, get summaries, or generate quizzes!")

# --- FILE UPLOAD & PROCESSING ---
st.sidebar.header("1. Upload Study Materials")
uploaded_file = st.sidebar.file_uploader("Upload a PDF document", type=["pdf"])

@st.cache_resource
def process_document(file):
    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file.read())
        tmp_file_path = tmp_file.name

    # 1. Load the PDF
    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()

    # 2. Break text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # 3. Store embeddings using Google's embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    return vectorstore, " ".join([doc.page_content for doc in documents])

if uploaded_file:
    with st.spinner("Processing document... Please wait."):
        vectorstore, full_text = process_document(uploaded_file)
        st.sidebar.success("Document processed and stored in FAISS!")

    # --- INITIALIZE GEMINI LLM ---
    # Using gemini-2.5-flash for fast, high-quality responses
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

    # --- TABS FOR FEATURES ---
    tab1, tab2, tab3 = st.tabs(["💬 Chat & Q&A", "📝 Summarize Notes", "🎯 Generate Quiz"])

    # FEATURE 1: Q&A Chatbot
    with tab1:
        st.subheader("Ask questions about your notes")
        user_question = st.text_input("What would you like to know?")
        if user_question:
            with st.spinner("Thinking..."):
                answer = qa_chain.invoke(user_question)
                st.write("**AI Tutor:**")
                st.info(answer['result'])

    # FEATURE 2: Summarization
    with tab2:
        st.subheader("Get a quick summary of the document")
        if st.button("Generate Summary"):
            with st.spinner("Summarizing..."):
                summary_prompt = f"Provide a comprehensive, easy-to-understand summary of the following text:\n\n{full_text[:10000]}..."
                summary = llm.invoke(summary_prompt).content
                st.success(summary)

    # FEATURE 3: Quiz Generation
    with tab3:
        st.subheader("Test your knowledge")
        if st.button("Generate a 3-Question Quiz"):
            with st.spinner("Generating quiz..."):
                quiz_prompt = f"Based on the following text, generate a multiple-choice quiz with 3 questions. Include the correct answers at the bottom:\n\n{full_text[:10000]}..."
                quiz = llm.invoke(quiz_prompt).content
                st.markdown(quiz)
else:
    st.info("👈 Please upload a PDF document in the sidebar to get started.")
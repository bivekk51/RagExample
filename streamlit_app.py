import os
import streamlit as st
from dotenv import load_dotenv
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai

# === Load API key ===
load_dotenv()
api_key = os.getenv("API_KEY")


# === Step 1: Load PDF and extract text ===
def load_pdf(path):
    doc = fitz.open(path)
    return "\n".join([page.get_text() for page in doc])

# === Step 2: Split text into overlapping chunks ===
def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

# === Step 3: Create embeddings and store in FAISS ===
def create_embeddings(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))
    return model, index, embeddings

# === Step 4: Retrieve top-k relevant chunks ===
def retrieve_top_chunks(query, model, index, chunks, k=3):
    query_embedding = model.encode([query])
    _, indices = index.search(np.array(query_embedding), k)
    return [chunks[i] for i in indices[0]]


# === Step 5: Ask Gemini with the retrieved context ===
def ask_gemini(api_key, context, question):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')

    prompt = f"""You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question:
{question}
"""

    response = model.generate_content(prompt)
    return response.text

# PDF processing functions (same as before)
# ... [your load_pdf, split_text, create_embeddings, etc.]

st.title("📚 PDF Chat with Gemini")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
user_question = st.text_input("Ask a question based on the PDF")

if uploaded_file and user_question and st.button("Get Answer"):
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    
    text = load_pdf("temp.pdf")
    chunks = split_text(text)
    model, index, _ = create_embeddings(chunks)
    top_chunks = retrieve_top_chunks(user_question, model, index, chunks)
    context = "\n\n".join(top_chunks)
    answer = ask_gemini(api_key, context, user_question)

    st.markdown("### 📌 Answer from Gemini")
    st.write(answer)

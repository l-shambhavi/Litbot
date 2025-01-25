import pdfplumber
import nltk
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import pipeline
import streamlit as st
from PIL import Image
import io
import requests
import re
import os

# Download NLTK resources
nltk.download("punkt")

# Function to clean text
def clean_text(text):
    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces/newlines with a single space
    text = re.sub(r"[^\x20-\x7E]", "", text)  # Remove non-printable ASCII characters
    return text.strip()


# URL of the book (PDF file)
BOOK_URL = "https://myweb.sabanciuniv.edu/rdehkharghani/files/2016/02/The-Morgan-Kaufmann-Series-in-Data-Management-Systems-Jiawei-Han-Micheline-Kamber-Jian-Pei-Data-Mining.-Concepts-and-Techniques-3rd-Edition-Morgan-Kaufmann-2011.pdf"
BOOK_FILENAME = "downloaded_book.pdf"

# Download the book if not already present
if not os.path.exists(BOOK_FILENAME):
    st.info("Downloading the book...")
    response = requests.get(BOOK_URL)
    with open(BOOK_FILENAME, "wb") as file:
        file.write(response.content)
    st.success("Book downloaded successfully!")

# Extract text and images from PDF
@st.cache
def load_model():
    return SentenceTransformer("paraphrase-MiniLM-L6-v2")

@st.cache_data
def extract_text_and_images(pdf_file):
    text = ""
    images = {}
    with pdfplumber.open(pdf_file) as pdf:
        for page_number, page in enumerate(pdf.pages):
            # Extract text
            if page.extract_text():
                text += clean_text(page.extract_text()) + "\n"
            # Extract images
            if page.images:
                for img_info in page.images:
                    try:
                        img_bbox = img_info.get("bbox")
                        if img_bbox:
                            image = page.within_bbox(img_bbox).to_image()
                            images[page_number] = image.original
                    except Exception:
                        continue
    return text, images


# Load data
book_text, book_images = extract_text_and_images(BOOK_FILENAME)

# Prepare sentences for embedding
book_sentences = [
    clean_text(sentence)
    for sentence in nltk.sent_tokenize(book_text)
    if clean_text(sentence)
]

# Load sentence transformer model
st.info("Generating sentence embeddings...")
embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
embeddings = embedding_model.encode(book_sentences)

# Create FAISS index
faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
faiss_index.add(np.array(embeddings))

# Search book for relevant sentences
def search_book(query, num_sentences=5):
    query_embedding = embedding_model.encode([query])
    distances, indices = faiss_index.search(np.array(query_embedding), k=num_sentences)
    return [(book_sentences[i], i) for i in indices[0]]


# Load QA model
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def get_answer(question, context):
    try:
        result = qa_pipeline(question=question, context=context)
        return result["answer"]
    except Exception as e:
        return f"Error in QA pipeline: {e}"


# Streamlit App
st.title("📚 AI-Powered Book Chatbot")
st.sidebar.header("About")
st.sidebar.write("This chatbot uses AI to answer questions from the book: *Data Mining: Concepts and Techniques*.")

# User Input
question = st.text_input("🔍 Ask your question:", "")

if question:
    st.subheader("Your Question:")
    st.write(question)

    # Retrieve relevant sentences
    results = search_book(question, num_sentences=5)

    if results:
        st.subheader("📖 Relevant Sentences:")
        for i, (sentence, idx) in enumerate(results, start=1):
            st.markdown(f"**{i}.** {sentence}")

        # Combine sentences for QA
        context = " ".join([sentence for sentence, _ in results])
        answer = get_answer(question, context)

        st.subheader("✅ Answer:")
        st.success(answer)

        # Display relevant pages and images
        pages = {idx + 1 for _, idx in results}
        st.subheader("📄 Relevant Pages:")
        st.write(sorted(pages))

        for page in pages:
            if page - 1 in book_images:
                img_data = book_images[page - 1]
                img = Image.open(io.BytesIO(img_data))
                st.image(img, caption=f"Image from page {page}")
    else:
        st.warning("No relevant sentences found. Try rephrasing your query.")

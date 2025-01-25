import pdfplumber
import nltk
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import pipeline
import requests
import re

# Download NLTK resources
nltk.download('punkt')

# Function to clean text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
    text = re.sub(r'[^\x20-\x7E]', '', text)  # Remove non-printable ASCII characters
    return text.strip()

# URL of the book (PDF file)
book_url = "https://myweb.sabanciuniv.edu/rdehkharghani/files/2016/02/The-Morgan-Kaufmann-Series-in-Data-Management-Systems-Jiawei-Han-Micheline-Kamber-Jian-Pei-Data-Mining.-Concepts-and-Techniques-3rd-Edition-Morgan-Kaufmann-2011.pdf"

# Function to download the book
def download_book(url, filename):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        with open(filename, 'wb') as file:
            file.write(response.content)
        print(f"Book downloaded successfully: {filename}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the book: {e}")

# Specify the filename to save the book
book_filename = "downloaded_book.pdf"

# Call the function to download the book
download_book(book_url, book_filename)

# Function to extract text and images from PDF
def extract_text_and_images_from_pdf(pdf_file):
    text = ''
    images = {}
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page_number, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    clean_page_text = clean_text(page_text)
                    text += clean_page_text + '\n'
    except Exception as e:
        print(f"Error while extracting text/images: {e}")
    return text, images

# Extract text from the downloaded book
book_text, book_images = extract_text_and_images_from_pdf(book_filename)

# Prepare sentences for embedding
book_sentences = [clean_text(sentence) for sentence in nltk.sent_tokenize(book_text) if clean_text(sentence)]

# Load the pre-trained sentence transformer model
print("Generating sentence embeddings...")
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Generate embeddings for book sentences
embeddings = model.encode(book_sentences)

# Create FAISS index for fast similarity search
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# Function to search the book for relevant sentences
def search_book(query, num_sentences=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k=num_sentences)
    return [(book_sentences[i], i) for i in indices[0]]  # Return sentences and their indices

# Load a pre-trained model for question-answering
qa_pipeline = pipeline('question-answering', model='distilbert/distilbert-base-cased-distilled-squad')

# Function to get the final answer from the chatbot
def get_answer(question):
    # Step 1: Search for relevant sentences
    relevant_sentences_and_pages = search_book(question, num_sentences=3)
    
    if not relevant_sentences_and_pages:
        return "No relevant sentences found."
    
    # Combine the context
    combined_context = " ".join([item[0] for item in relevant_sentences_and_pages])
    
    # Step 2: Use QA model to answer based on the context
    try:
        result = qa_pipeline(question=question, context=combined_context)
        answer = result['answer']
    except Exception as e:
        print(f"Error in get_answer: {e}")
        answer = "Sorry, I couldn't find an answer."
    
    # Step 3: Prepare the response
    response = f"Question: {question}\n\n"
    response += "Relevant Sentences:\n"
    for i, (sentence, index) in enumerate(relevant_sentences_and_pages, start=1):
        response += f"{i}. {sentence}\n"
    response += f"\nAnswer: {answer}"
    return response

# Gradio interface
import gradio as gr

# Create Gradio interface
if __name__ == "__main__":
    iface = gr.Interface(
        fn=get_answer,
        inputs=gr.inputs.Textbox(lines=2, placeholder="Ask a question about the book..."),
        outputs="text",
        title="AI-Powered Book Chatbot",
        description="Ask questions about the book, and the chatbot will find relevant answers for you!"
    )

    iface.launch(share=True)  # This makes the app publicly accessible


# Run the app
iface.launch()

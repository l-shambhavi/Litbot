import pdfplumber
import nltk
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import pipeline
import tkinter as tk
from tkinter import scrolledtext, messagebox
from PIL import Image, ImageTk
import io
import requests
import re
import os

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
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
                
                # Only proceed if images exist on the page
                if page.images:
                    for image_info in page.images:
                        img_bbox = image_info.get("bbox")
                        if img_bbox:
                            try:
                                image = page.within_bbox(img_bbox).to_image()
                                images[page_number] = image.original
                            except Exception as e:
                                print(f"Error extracting image on page {page_number}: {e}")
                else:
                    print(f"No images on page {page_number}")
    except Exception as e:
        print(f"Error while extracting text/images: {e}")
    return text, images

# Extract text and images from the downloaded book
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

# Function to search the book for relevant sentences and page numbers
def search_book(query, num_sentences=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k=num_sentences)
    return [(book_sentences[i], i) for i in indices[0]]  # Return sentences and their indices

# Load a pre-trained model for question-answering
qa_pipeline = pipeline('question-answering', model='distilbert/distilbert-base-cased-distilled-squad')

# Function to get the final answer from the chatbot
def get_answer(question, context):
    try:
        result = qa_pipeline(question=question, context=context)
        return result['answer']
    except Exception as e:
        print(f"Error in get_answer: {e}")
        return "Sorry, I couldn't find an answer."

def ask_question():
    question = question_entry.get()
    if not question:
        messagebox.showwarning("Input Error", "Please enter a question.")
        return
    
    # Step 1: Search for relevant sentences (limit to 3 highly relevant ones)
    relevant_sentences_and_pages = search_book(question, num_sentences=3)
    
    if not relevant_sentences_and_pages:
        output_text.insert(tk.END, "No relevant sentences found.\n")
        return

    # Clear previous output
    output_text.delete(1.0, tk.END)

    # Display the question
    output_text.insert(tk.END, f"Question: {question}\n", "bold")
    
    # Step 2: Display relevant sentences with improved formatting and spacing
    output_text.insert(tk.END, "Relevant Sentences:\n\n", "underline")
    
    for i, (sentence, index) in enumerate(relevant_sentences_and_pages, start=1):
        clean_sentence = re.sub(r'\s+', ' ', sentence)
        output_text.insert(tk.END, f"{i}. {clean_sentence}\n\n", "regular")
    
    # Step 3: Use QA model to answer based on the most relevant text
    combined_context = " ".join([item[0] for item in relevant_sentences_and_pages])
    answer = get_answer(question, combined_context)
    
    output_text.insert(tk.END, f"\nGenerated Answer:\n{answer}\n\n", "italic")
    
    # Step 4: Display relevant page numbers (indices)
    page_numbers = {item[1] + 1 for item in relevant_sentences_and_pages}
    output_text.insert(tk.END, f"Relevant content found on pages: {sorted(page_numbers)}\n\n", "bold")

    # Step 5: Display relevant images (if available)
    for _, page in relevant_sentences_and_pages:
        if page in book_images:
            img_data = book_images[page]
            img = Image.open(io.BytesIO(img_data))
            img.thumbnail((250, 250))
            img_tk = ImageTk.PhotoImage(img)
            image_label.configure(image=img_tk)
            image_label.image = img_tk
        else:
            image_label.configure(image='')  # Clear the image if none is available

# Create the UI using tkinter
root = tk.Tk()
root.title("AI-Powered Chatbot")

# Set the window size and background color
root.geometry("800x600")
root.configure(bg="#f5f5f5")

# Create frames for better structure
input_frame = tk.Frame(root, bg="#f0f0f0", padx=10, pady=10)
input_frame.pack(pady=10, fill=tk.X)

output_frame = tk.Frame(root, bg="#ffffff", padx=10, pady=10)
output_frame.pack(pady=10, fill=tk.BOTH, expand=True)

image_frame = tk.Frame(root, bg="#f0f0f0", padx=10, pady=10)
image_frame.pack(pady=10)

# Input for the question
question_label = tk.Label(input_frame, text="Ask your question:", bg="#f0f0f0", font=("Helvetica", 12))
question_label.grid(row=0, column=0, sticky="w")

question_entry = tk.Entry(input_frame, width=50, font=("Helvetica", 12))
question_entry.grid(row=0, column=1, padx=10)

# Button to submit the question
ask_button = tk.Button(input_frame, text="Ask", command=ask_question, font=("Helvetica", 12), bg="#4CAF50", fg="#fff")
ask_button.grid(row=0, column=2)

# Output text area to display the answer
output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, font=("Helvetica", 12), bg="#f9f9f9", height=15)
output_text.tag_configure("bold", font=("Helvetica", 12, "bold"))
output_text.tag_configure("underline", font=("Helvetica", 12, "underline"))
output_text.tag_configure("regular", font=("Helvetica", 12))
output_text.tag_configure("italic", font=("Helvetica", 12, "italic"))
output_text.pack(fill=tk.BOTH, expand=True)

# Image label to display images
image_label = tk.Label(image_frame, bg="#f0f0f0")
image_label.pack()

# Status bar
status_bar = tk.Label(root, text="Chatbot ready", bd=1, relief=tk.SUNKEN, anchor=tk.W, font=("Helvetica", 10))
status_bar.pack(side=tk.BOTTOM, fill=tk.X)

# Start the Tkinter main loop
root.mainloop()
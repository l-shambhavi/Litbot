# Use an official Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . .

# Install system dependencies for Tkinter, NLTK, and other required libraries
RUN apt-get update && apt-get install -y \
    python3-tk \
    libgl1-mesa-glx \
    tesseract-ocr \
    libsm6 libxext6 libxrender-dev \
    gcc g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    pdfplumber \
    nltk \
    sentence-transformers \
    numpy \
    faiss-cpu \
    transformers \
    Pillow \
    requests \
    gradio

# Download necessary NLTK resources
RUN python -c "import nltk; nltk.download('punkt')"

# Expose port if needed (optional, for hosting)
EXPOSE 8080

# Set the command to run your script
CMD ["python", "-u", "app.py"]

import os
import pymongo
import fitz  # PyMuPDF for PDF processing
from config import MONGO_URI, DB_NAME, COLLECTION_NAME

# Connect to MongoDB
client = pymongo.MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Folder containing research papers
FOLDER_PATH = "./Research Papers"

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

def store_in_mongodb():
    """Extract text from PDFs and store in MongoDB."""
    for filename in os.listdir(FOLDER_PATH):
        if filename.endswith(".pdf"):
            file_path = os.path.join(FOLDER_PATH, filename)
            print(f"📄 Processing: {filename}")

            text = extract_text_from_pdf(file_path)
            data = {"filename": filename, "content": text}

            collection.insert_one(data)
            print(f"✅ Stored in MongoDB: {filename}")

if __name__ == "__main__":
    store_in_mongodb()

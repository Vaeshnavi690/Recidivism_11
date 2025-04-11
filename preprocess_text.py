import pymongo
import re
import spacy
from config import MONGO_URI, DB_NAME, COLLECTION_NAME

# Load SpaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Connect to MongoDB
client = pymongo.MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

def clean_text(text):
    """Cleans and preprocesses text."""
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'\[.*?\]', '', text)  # Remove references like [1], [2]
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    return text.strip()

def preprocess_text(text):
    """Performs tokenization, stopword removal, and lemmatization."""
    doc = nlp(text.lower())  # Convert text to lowercase
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

def process_documents():
    """Fetches documents from MongoDB, cleans, processes, and updates them."""
    for doc in collection.find():
        raw_text = doc.get("content", "")
        cleaned_text = clean_text(raw_text)
        processed_text = preprocess_text(cleaned_text)

        # Update MongoDB with processed text
        collection.update_one({"_id": doc["_id"]}, {"$set": {"processed_content": processed_text}})
        print(f"✅ Processed: {doc['filename']}")

if __name__ == "__main__":
    process_documents()

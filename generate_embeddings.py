import pymongo
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config import MONGO_URI, DB_NAME, COLLECTION_NAME

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to MongoDB
client = pymongo.MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

def generate_embeddings():
    """Generate embeddings for processed text and store in MongoDB."""
    texts = []
    doc_ids = []

    for doc in collection.find():
        if "processed_content" in doc:
            texts.append(doc["processed_content"])
            doc_ids.append(doc["_id"])

    # Generate embeddings
    print("🚀 Generating embeddings for documents...")
    embeddings = model.encode(texts, show_progress_bar=True)

    # Store embeddings in MongoDB
    for i, doc_id in enumerate(doc_ids):
        collection.update_one({"_id": doc_id}, {"$set": {"embedding": embeddings[i].tolist()}})
    
    print("✅ Embeddings generated and stored successfully.")

if __name__ == "__main__":
    generate_embeddings()

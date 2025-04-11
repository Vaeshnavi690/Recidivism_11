import faiss
import pickle
import numpy as np
import pymongo
from sentence_transformers import SentenceTransformer
from config import MONGO_URI, DB_NAME, COLLECTION_NAME

# Load FAISS index and document IDs
index = faiss.read_index("vector_store.index")
with open("doc_ids.pkl", "rb") as f:
    doc_ids = pickle.load(f)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to MongoDB
client = pymongo.MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

def search(query, top_k=3):
    """Search FAISS index for similar documents."""
    query_embedding = model.encode([query]).astype(np.float32)
    
    # Perform similarity search
    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    for i in range(top_k):
        doc_id = doc_ids[indices[0][i]]
        doc = collection.find_one({"_id": doc_id})
        results.append({"document": doc["processed_content"], "score": distances[0][i]})

    return results

if __name__ == "__main__":
    query = input("🔍 Enter your search query: ")
    results = search(query)
    
    print("\n🎯 **Top Matches:**")
    for i, res in enumerate(results):
        print(f"\n🔹 Match {i+1}:")
        print(f"📄 Document: {res['document'][:300]}...")  # Show first 300 characters
        print(f"💡 Similarity Score: {res['score']}\n")

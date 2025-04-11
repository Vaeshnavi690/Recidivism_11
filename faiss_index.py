import pymongo
import faiss
import numpy as np
import pickle
from config import MONGO_URI, DB_NAME, COLLECTION_NAME

# Connect to MongoDB
client = pymongo.MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

def load_embeddings():
    """Load embeddings and document IDs from MongoDB."""
    embeddings = []
    doc_ids = []

    for doc in collection.find({}, {"embedding": 1}):
        if "embedding" in doc:
            embeddings.append(doc["embedding"])
            doc_ids.append(doc["_id"])

    return np.array(embeddings, dtype=np.float32), doc_ids

def build_faiss_index():
    """Create and store a FAISS index."""
    embeddings, doc_ids = load_embeddings()

    if embeddings.shape[0] == 0:
        print("❌ No embeddings found in database!")
        return

    # Create FAISS index
    d = embeddings.shape[1]  # Dimension of embeddings
    index = faiss.IndexFlatL2(d)  # L2 distance metric
    index.add(embeddings)  # Add embeddings to the index

    # Save FAISS index and document IDs
    faiss.write_index(index, "vector_store.index")
    with open("doc_ids.pkl", "wb") as f:
        pickle.dump(doc_ids, f)

    print("✅ FAISS index built and saved successfully!")

if __name__ == "__main__":
    build_faiss_index()

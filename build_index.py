# build_index.py
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import pickle

df = pd.read_csv('tickets.csv')
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['description'].tolist(), convert_to_tensor=False)

index = faiss.IndexFlatL2(embeddings[0].shape[0])
index.add(embeddings)

# Save index and data
faiss.write_index(index, "ticket_index.faiss")
with open("ticket_data.pkl", "wb") as f:
    pickle.dump(df, f)

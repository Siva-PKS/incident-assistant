# app.py
import streamlit as st
import pandas as pd
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import openai

st.set_page_config(page_title="Incident Ticket Assistant", layout="wide")

# Load data and index
@st.cache_resource
def load_model_and_index():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index("ticket_index.faiss")
    with open("ticket_data.pkl", "rb") as f:
        df = pickle.load(f)
    return model, index, df

model, index, df = load_model_and_index()

# UI
st.title("🛠 Incident Ticket Auto-Resolution Assistant")

query = st.text_area("📝 Describe the new issue:", height=150)
openai_api_key = st.text_input("🔑 OpenAI API Key", type="password")

if st.button("🔍 Find Similar & Generate Response") and query and openai_api_key:
    query_embedding = model.encode([query])[0]
    D, I = index.search([query_embedding], k=3)

    st.subheader("🧾 Top Matching Tickets:")
    for i in I[0]:
        st.markdown(f"- **Ticket #{df.iloc[i]['ticket_id']}**: {df.iloc[i]['description']}")

    # LLM call
    context = "\n".join([df.iloc[i]['description'] + " -> " + df.iloc[i]['resolution'] for i in I[0]])
    prompt = f"""You are an IT support assistant. Based on these past tickets and their resolutions:\n\n{context}\n\nWhat would you suggest for this issue:\n{query}"""

    openai.api_key = openai_api_key
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    st.subheader("🧠 Suggested Response:")
    st.success(response.choices[0].message['content'])


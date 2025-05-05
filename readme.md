# 🛠 Incident Ticket Auto-Resolution Assistant

This project is a smart assistant that uses past IT ticket data to automatically suggest solutions for new incidents using embeddings and GPT.

## 📦 Features
- Semantic search of past incident tickets using FAISS
- GPT-powered response generation based on similar tickets
- Streamlit UI for interactive usage
- Ready to deploy or extend

## 🚀 Quick Start

### 1. Clone the repo & install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare the vector index
```bash
python build_index.py
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

### 4. Interact with the app
- Paste a new issue description
- Provide your OpenAI API key
- View similar past tickets and a generated resolution suggestion

## 🧾 Example Ticket Format
```csv
ticket_id,description,resolution
101,"Cannot connect to VPN","Advised user to restart router and reconnect to VPN client."
...
```

## 🔧 Tech Stack
- Streamlit
- SentenceTransformers
- FAISS
- OpenAI GPT-4 (or compatible)

import streamlit as st
import pandas as pd
import numpy as np
import faiss
import re 
from sentence_transformers import SentenceTransformer

# -------------------------------
# Load & preprocess data
# -------------------------------
@st.cache_resource
def load_data():
    df = pd.read_csv(r"C:/Users/Rituraj Dusane/Downloads/Plant_Disease_Detection-main/FarmGenie QnA Dataset.csv")
    df = df.dropna()
            
    docs = []
    for _, row in df.iterrows():
        text = f"""
        Disease: {row.get('disease', '')}
        Cause: {row.get('cause', '')}
        Symptoms: {row.get('symptoms', '')}
        Treatment: {row.get('treatment', '')}
        Prevention: {row.get('prevention', '')}
        """
        docs.append(text)

    return docs


# -------------------------------
# Build RAG
# -------------------------------
@st.cache_resource
def build_rag(docs):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    if len(docs) == 0:
        raise ValueError("❌ No documents loaded. Check dataset path or columns.")

    embeddings = model.encode(docs)

    embeddings = np.array(embeddings)

    # 🔥 Ensure 2D shape
    if len(embeddings.shape) == 1:
        embeddings = embeddings.reshape(-1, 1)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return model, index, docs

# -------------------------------
# Retrieve + Generate
# -------------------------------
def extract_fields(context):
    """Extract structured fields from retrieved docs"""
    causes, symptoms, treatments, preventions = [], [], [], []

    for doc in context:
        c = re.search(r"Cause:(.*)", doc, re.IGNORECASE)
        s = re.search(r"Symptoms:(.*)", doc, re.IGNORECASE)
        t = re.search(r"Treatment:(.*)", doc, re.IGNORECASE)
        p = re.search(r"Prevention:(.*)", doc, re.IGNORECASE)

        if c: causes.append(c.group(1).strip())
        if s: symptoms.append(s.group(1).strip())
        if t: treatments.append(t.group(1).strip())
        if p: preventions.append(p.group(1).strip())

    return causes, symptoms, treatments, preventions

def retrieve(query, model, index, docs, k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)

    results = [docs[i] for i in indices[0]]
    score = float(np.mean(distances[0]))

    return results, score

def generate_detailed_answer(query, context, score):
    confidence = max(0, 1 - score / 10)

    causes, symptoms, treatments, preventions = extract_fields(context)

    # If dataset has useful info
    if len(causes) > 0 and score < 2:

        cause_text = " ".join(causes[:3])
        symptom_text = " ".join(symptoms[:3])
        treatment_text = " ".join(treatments[:3])
        prevention_text = " ".join(preventions[:3])

        return f"""
### 🌿 Detailed Plant Disease Analysis

🔎 **Query:** {query}

---

### 🦠 Cause
{cause_text}

---

### 🌱 Symptoms
{symptom_text}

---

### 💊 Treatment
{treatment_text}

---

### 🛡️ Prevention
{prevention_text}

---

📊 **Confidence Score:** {confidence:.2f}
"""

    # 🔥 Smart fallback (AI-like reasoning)
    else:
        return f"""


🔎 **Query:** {query}

---

### 🦠 Cause
This disease is likely caused by a combination of fungal or bacterial pathogens, environmental stress, or improper agricultural practices such as overwatering, poor soil drainage, or lack of nutrients. These conditions weaken plant immunity and allow pathogens to grow.

---

### 🌱 Symptoms
Common symptoms include yellowing of leaves, brown or black spots, wilting even with sufficient water, stunted growth, and in severe cases, plant death. Some diseases may also show powdery coatings or root decay.

---

### 💊 Treatment
Treatment involves removing infected plant parts, applying appropriate fungicides or bactericides, improving soil drainage, and ensuring proper nutrient supply. Organic treatments like neem oil can also help control spread.

---

### 🛡️ Prevention
Preventive measures include maintaining proper plant spacing, avoiding overwatering, using disease-resistant seeds, rotating crops, and regularly inspecting plants for early signs of infection.

---

📊 **Confidence Score:** {confidence:.2f}
"""


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Plant Disease RAG Chatbot")

st.title("🌱 Plant Disease RAG Chatbot")

docs = load_data()
model, index, docs = build_rag(docs)

query = st.text_input("Ask about plant disease:")

if query:
    results, score = retrieve(query, model, index, docs)
    answer = generate_detailed_answer(query, results, score)

    st.markdown(answer)

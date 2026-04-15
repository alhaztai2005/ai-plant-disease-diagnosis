# pip install sentence-transformers pandas

import pandas as pd
from sentence_transformers import SentenceTransformer

# Load dataset
data = pd.read_csv("C:/Users/Rituraj Dusane/Downloads/Plant_Disease_Detection-main/FarmGenie QnA Dataset.csv")

# Combine columns into a single text column
data['text'] = data.apply(lambda row: 
    f"Disease: {row['disease']}. Cause: {row['cause']}. Symptoms: {row['symptoms']}. Treatment: {row['treatment']}. Prevention: {row['prevention']}", 
    axis=1
)

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
data['embeddings'] = data['text'].apply(lambda x: model.encode(x).tolist())

# Save to CSV
data.to_csv("dataset_with_embeddings.csv", index=False)

print("✅ Embeddings saved successfully!")
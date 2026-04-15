# pip install pandas gensim nltk scikit-learn

import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

nltk.download('punkt')

# Load dataset
data = pd.read_csv("C:/Users/Rituraj Dusane/Downloads/Plant_Disease_Detection-main/plant_disease_dataset.csv")

# Combine columns
data['text'] = data[['disease','cause','symptoms','treatment','prevention']].fillna('').agg(' '.join, axis=1)

# Tokenize
data['tokens'] = data['text'].apply(word_tokenize)

# Train Word2Vec
w2v_model = Word2Vec(
    sentences=data['tokens'],
    vector_size=100,
    window=5,
    min_count=1,
    workers=4
)

# Sentence → vector
def sentence_vector(tokens):
    vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(100)

data['embedding'] = data['tokens'].apply(sentence_vector)

# Features & labels
X = np.vstack(data['embedding'].values)
y = data['disease']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Overall metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print results
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
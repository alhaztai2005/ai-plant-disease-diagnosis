# pip install pandas gensim nltk

import pandas as pd
import nltk
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

nltk.download('punkt')

# Load dataset
data = pd.read_csv("C:/Users/Rituraj Dusane/Downloads/Plant_Disease_Detection-main/FarmGenie QnA Dataset.csv")

# Combine text
data['text'] = data[['disease','cause','symptoms','treatment','prevention']].fillna('').agg(' '.join, axis=1)

# Tokenization
data['tokens'] = data['text'].apply(word_tokenize)

# Train Word2Vec
model = Word2Vec(
    sentences=data['tokens'],
    vector_size=100,
    window=5,
    min_count=1,
    workers=4
)

# Example: get vector of a word
print(model.wv['disease'])
# pip install pandas scikit-learn

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
data = pd.read_csv("C:/Users/Rituraj Dusane/Downloads/Plant_Disease_Detection-main/FarmGenie QnA Dataset.csv")

# Combine columns into single text
data['text'] = data[['disease','cause','symptoms','treatment','prevention']].fillna('').agg(' '.join, axis=1)

# Initialize TF-IDF
tfidf = TfidfVectorizer(
    max_features=5000,      # limit features (can increase)
    stop_words='english'    # remove common words
)

# Fit and transform
tfidf_matrix = tfidf.fit_transform(data['text'])

# Convert to DataFrame (optional)
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())

# Save to CSV
tfidf_df.to_csv("tfidf_features.csv", index=False)

print("✅ TF-IDF features saved!")

import pandas as pd
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Paths
CSV_PATH = "C:/Users/Rituraj Dusane/Downloads/Plant_Disease_Detection-main/CHATBOT/qna_preprocessed_chatbot_data.csv"
OUTPUT_PATH = "qna_pos_tagged_spacy.csv"

# Load CSV
df = pd.read_csv(CSV_PATH)

def apply_pos_spacy(text):
    if pd.isna(text):
        return ""
    doc = nlp(text)
    # word/POS format
    return " ".join([f"{token.text}/{token.pos_}" for token in doc])

# Apply POS tagging
df["input_pos"] = df["input"].apply(apply_pos_spacy)
df["response_pos"] = df["response"].apply(apply_pos_spacy)

# Save output
df.to_csv(OUTPUT_PATH, index=False)

print("✅ spaCy POS tagging completed")
print(f"📄 Saved as: {OUTPUT_PATH}")


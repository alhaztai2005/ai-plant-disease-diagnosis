import re
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

# -------- CONFIG --------
PARQUET_PATH = r"C:/Users/Rituraj Dusane/Downloads/Plant_Disease_Detection-main/CHATBOT for NLP/CSV outputs/FarmGenie QnA Dataset.csv"
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
CSV_OUTPUT_PATH = "qna_preprocessed_chatbot_data.csv"
# -----------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9?.!,']", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def main():
    print("Loading local parquet file...")
    df = pd.read_parquet(PARQUET_PATH)

    print("Available columns:", list(df.columns))

    # -------- COLUMN DETECTION (UPDATED) --------
    if {"QUESTION.question", "ANSWER"}.issubset(df.columns):
        input_col = "QUESTION.question"
        response_col = "ANSWER"

    elif {"question", "answer"}.issubset(df.columns):
        input_col, response_col = "question", "answer"

    elif {"question", "answers"}.issubset(df.columns):
        input_col, response_col = "question", "answers"

    elif {"instruction", "output"}.issubset(df.columns):
        input_col, response_col = "instruction", "output"

    elif {"input", "response"}.issubset(df.columns):
        input_col, response_col = "input", "response"

    else:
        raise ValueError("❌ Unable to detect input/response columns automatically")

    print(f"Using columns → input: {input_col}, response: {response_col}")

    # -------- TEXT CLEANING --------
    df["input"] = df[input_col].apply(clean_text)
    df["response"] = df[response_col].apply(clean_text)

    df = df[["input", "response"]]

    # -------- SAVE CSV --------
    df.to_csv(CSV_OUTPUT_PATH, index=False)
    print(f"✅ Cleaned CSV saved: {CSV_OUTPUT_PATH}")

    # -------- TOKENIZATION --------
    print("Tokenizing...")
    hf_ds = Dataset.from_pandas(df)

    tokenized = hf_ds.map(
        lambda x: tokenizer(
            x["input"],
            text_target=x["response"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        ),
        batched=True,
        remove_columns=["input", "response"]
    )

    final_ds = tokenized.train_test_split(test_size=0.1)
    final_ds.save_to_disk("processed_chatbot_dataset")

    print("✅ Preprocessing completed successfully")

if __name__ == "__main__":
    main()


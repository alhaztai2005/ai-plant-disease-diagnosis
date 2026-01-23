"""
Week 1: Data Preprocessing & Feature Extraction
Plant Disease AI System
"""

import re
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

PARQUET_PATH = "dataset/raw/train-00000-of-00001.parquet"
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
CSV_OUTPUT_PATH = "dataset/processed/preprocessed_chatbot_data.csv"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9?.!,']", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def preprocess_data():
    df = pd.read_parquet(PARQUET_PATH)

    if {"question", "answer"}.issubset(df.columns):
        input_col, response_col = "question", "answer"
    elif {"instruction", "output"}.issubset(df.columns):
        input_col, response_col = "instruction", "output"
    else:
        raise ValueError("Unable to detect columns")

    df["input"] = df[input_col].apply(clean_text)
    df["response"] = df[response_col].apply(clean_text)
    df = df[["input", "response"]]

    df.to_csv(CSV_OUTPUT_PATH, index=False)

    dataset = Dataset.from_pandas(df)

    tokenized_ds = dataset.map(
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

    return tokenized_ds

if __name__ == "__main__":
    preprocess_data()


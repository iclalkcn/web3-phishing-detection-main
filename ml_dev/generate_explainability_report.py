import pandas as pd
import joblib
from transformers import AutoTokenizer, AutoModel
import sys
sys.path.append("../app")

from utils import (
    get_embedding,
)
from lime.lime_text import LimeTextExplainer
import torch
import numpy as np

MODEL_PATH = "model_outputs/lgbm_model.pkl"
BASE_MODEL = "distilbert-base-uncased"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Model yükleniyor...")

classifier = joblib.load(MODEL_PATH)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
bert_model = AutoModel.from_pretrained(BASE_MODEL)

bert_model.to(device)
bert_model.eval()

print("Dataset yükleniyor...")

df = pd.read_csv("data/web3_pure_dataset.csv")

phishing_samples = df[df["label"] == 1].sample(20, random_state=42)

explainer = LimeTextExplainer(
    class_names=["GENUINE", "PHISHING"]
)

report_lines = []

for idx, row in phishing_samples.iterrows():

    text = row["text"]

    def predictor(texts):
        embeddings = []

        for t in texts:
            emb = get_embedding(t, tokenizer, bert_model)
            embeddings.append(emb[0])

        embeddings = np.array(embeddings)

        return classifier.predict_proba(embeddings)

    explanation = explainer.explain_instance(
        text,
        predictor,
        num_features=8,
        num_samples=200
    )

    report_lines.append("=" * 60)
    report_lines.append(f"MESAJ:\n{text}\n")

    report_lines.append("LIME AÇIKLAMASI:")

    for word, score in explanation.as_list():
        report_lines.append(f"{word} --> {score:.4f}")

    report_lines.append("\n")

with open("lime_phishing_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print("Rapor oluşturuldu: lime_phishing_report.txt")
#python generate_explainability_report.py
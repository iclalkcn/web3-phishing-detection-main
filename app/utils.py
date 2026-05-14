from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import torch
from lime.lime_text import LimeTextExplainer
from transformers import AutoModel, AutoTokenizer

BASE_MODEL = "distilbert-base-uncased"
MODEL_PATH = "models/lgbm_model.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_tokenizer_for_app(
    model_path: str = MODEL_PATH,
    tokenizer_name: str = BASE_MODEL,
) -> Tuple[Any, AutoTokenizer, AutoModel]:
    try:
        classifier = joblib.load(model_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        bert_model = AutoModel.from_pretrained(tokenizer_name)
        bert_model.to(device)
        bert_model.eval()
        return classifier, tokenizer, bert_model
    except Exception as e:
        print(f"An error occurred while loading the model/tokenizer: {e}")
        return None, None, None


def get_embedding(text: str, tokenizer: AutoTokenizer, bert_model: AutoModel) -> np.ndarray:
    encoding = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )

    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = bert_model(**encoding)

    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return embedding


def predict_proba_texts(texts, classifier, tokenizer, bert_model):
    embeddings = []

    for text in texts:
        emb = get_embedding(text, tokenizer, bert_model)
        embeddings.append(emb[0])

    embeddings = np.array(embeddings)
    probs = classifier.predict_proba(embeddings)

    return probs


def get_prediction(
    classifier, tokenizer: AutoTokenizer, bert_model: AutoModel, text: str
) -> Optional[Dict[str, Any]]:
    try:
        risky_keywords = [
            "seed phrase",
            "recovery phrase",
            "private key",
            "restore your wallet",
            "verify your wallet",
            "connect your wallet",
            "claim airdrop",
            "free airdrop",
            "wallet validation",
            "wallet verification",
        ]

        lower_text = text.lower()

        for keyword in risky_keywords:
            if keyword in lower_text:
                return {
                    "LABEL": "PHISHING",
                    "probability": 99.9,
                }

        embedding = get_embedding(text, tokenizer, bert_model)
        probs = classifier.predict_proba(embedding)[0]

        predicted_class = int(np.argmax(probs))

        label = "PHISHING" if predicted_class == 1 else "GENUINE"
        probability = float(probs[predicted_class])

        return {
            "LABEL": label,
            "probability": round(probability * 100, 2),
        }

    except Exception as e:
        print(f"An error occurred while getting prediction: {e}")
        return None


def get_lime_explanation(
    classifier, tokenizer: AutoTokenizer, bert_model: AutoModel, text: str
):
    try:
        explainer = LimeTextExplainer(class_names=["GENUINE", "PHISHING"])

        explanation = explainer.explain_instance(
            text,
            lambda texts: predict_proba_texts(texts, classifier, tokenizer, bert_model),
            num_features=8,
            num_samples=200,
        )

        return explanation.as_list()

    except Exception as e:
        print(f"An error occurred while explaining prediction: {e}")
        return 
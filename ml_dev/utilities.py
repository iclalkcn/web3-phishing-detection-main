import numpy as np
import pandas as pd
import torch
import yaml
import joblib
import os

from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer

def load_config(config_path="config.yaml"):
    """Konfigürasyon dosyasını güvenli bir şekilde yükler."""
    try:
        if not os.path.exists(config_path):
            return {}
        with open(config_path, "r") as file:
            data = yaml.safe_load(file)
        return data if data is not None else {}
    except Exception as e:
        return {}

def load_data(data_file_path):
    """Farklı kaynaklardan gelen verileri standart formata (text, label) sokar."""
    data = pd.read_csv(data_file_path)
    
    # Kolon isimlerini standartlaştır
    if 'msg_content' in data.columns:
        data = data.rename(columns={"msg_content": "text"})
    elif 'description' in data.columns:
        data = data.rename(columns={"description": "text"})
        
    if 'lable' in data.columns:
        data = data.rename(columns={"lable": "label"})
    
    data["text"] = data["text"].astype(str)
    
    # Kopya mesajları temizle (Ezberlemeyi önlemek için)
    data = data.drop_duplicates(subset=["text"])
    
    # Etiket standardı: 1 = PHISHING, 0 = GENUINE
    if data["label"].dtype == object:
        data["label"] = data["label"].apply(
            lambda x: 1 if str(x).lower().strip() in ["phishing", "spam", "1"] else 0
        )
    
    return data[["text", "label"]]

def prepare_data(data_file_path, test_size=0.2, random_state=42):
    df = load_data(data_file_path)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df, test_df

def extract_embeddings(text_list, model_name=None, batch_size=16):
    """BERT (DistilBERT) kullanarak metinlerden sayısal özellik vektörleri çıkarır."""
    if model_name is None:
        config = load_config()
        # Sigorta: Eğer config'de yoksa varsayılan modeli kullan
        model_name = config.get("BASE_MODEL", "distilbert-base-uncased-finetuned-sst-2-english")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    embeddings = []
    
    for i in range(0, len(text_list), batch_size):
        batch_texts = text_list[i:i+batch_size]
        encoded = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        with torch.no_grad():
            outputs = model(**encoded)
            # [CLS] token'ının vektörünü alıyoruz
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(batch_embeddings)
            
    return np.vstack(embeddings)

def load_hybrid_model():
    """BERT bileşenlerini ve eğitilmiş LightGBM modelini yükler."""
    config = load_config()
    model_name = config.get("BASE_MODEL", "distilbert-base-uncased-finetuned-sst-2-english")
    
    # Çıktı klasörü kontrolü
    output_dir = config.get("output_dir", "model_outputs")
    exp_name = config.get("experiment_name", "web3_phishing")
    model_path = f"./{output_dir}/{exp_name}/"
    lgbm_filename = config.get("lgbm_model_filename", "lgbm_model.pkl")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = AutoModel.from_pretrained(model_name)
    lgbm_model = joblib.load(f"{model_path}/{lgbm_filename}")
    
    return bert_model, tokenizer, lgbm_model

def get_prediction(bert_model, tokenizer, lgbm_model, text):
    """Hibrit model ile bir mesajın Phishing olup olmadığını tahmin eder."""
    bert_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model.to(device)
    
    encoded = tokenizer([text], padding=True, truncation=True, return_tensors="pt", max_length=512)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    
    with torch.no_grad():
        outputs = bert_model(**encoded)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
    probs = lgbm_model.predict_proba(embedding)[0]
    label = np.argmax(probs)
    
    return {
        "LABEL": "PHISHING" if label == 1 else "GENUINE",
        "probability": float(probs[label]),
    }
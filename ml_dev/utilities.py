import evaluate
import numpy as np
import pandas as pd
import torch
import yaml
import joblib
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer

def read_yaml(file_path):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return data

config_file_path = "config.yaml"
config = read_yaml(config_file_path)

BASE_MODEL = config["BASE_MODEL"]
output_dir = config["output_dir"]
experiment_name = config["experiment_name"]
full_output_dir = f"{output_dir}/{experiment_name}"

def load_data(data_file_path):
    data = pd.read_csv(data_file_path)
    data = data.rename(columns={"Messages": "text", "gen_label": "label"})
    return data

def prepare_data(data_file_path, test_size=0.2, random_state=42):
    train_df, test_df = train_test_split(
        load_data(data_file_path), test_size=test_size, random_state=random_state
    )
    return train_df, test_df

def extract_embeddings(text_list, model_name=BASE_MODEL, batch_size=16):
    """
    BERT modelini kullanarak metinlerden embedding (özellik vektörü) çıkarır.
    Sınıflandırma yapmaz, LightGBM için veri hazırlar.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Sınıflandırma katmanı olmayan saf modeli yüklüyoruz
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    embeddings = []
    
    for i in range(0, len(text_list), batch_size):
        batch_texts = text_list[i:i+batch_size].tolist() if isinstance(text_list, pd.Series) else text_list[i:i+batch_size]
        encoded = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        with torch.no_grad():
            outputs = model(**encoded)
            # [CLS] token'ının vektörünü alıyoruz (cümle özeti)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(batch_embeddings)
            
    return np.vstack(embeddings)

def load_hybrid_model(model_path=f"./{full_output_dir}/", lgbm_filename="lgbm_model.pkl"):
    """BERT tokenizer ve eğitilmiş LightGBM modelini yükler."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    bert_model = AutoModel.from_pretrained(BASE_MODEL)
    lgbm_model = joblib.load(f"{model_path}/{lgbm_filename}")
    
    return bert_model, tokenizer, lgbm_model

def get_prediction(bert_model, tokenizer, lgbm_model, text):
    """Yeni metni BERT ile vektörleştirip LightGBM ile tahmin eder."""
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
        "LABEL": "GENUINE" if label == 1 else "PHISHING",
        "probability": float(probs[1] if label == 1 else probs[0]),
    }
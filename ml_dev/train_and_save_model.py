import os
import joblib
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utilities import (
    read_yaml,
    prepare_data,
    extract_embeddings,
    load_hybrid_model,
    get_prediction
)

config_file_path = "config.yaml"
config = read_yaml(config_file_path)

data_file_path = config["data_file_path"]
BASE_MODEL = config["BASE_MODEL"]
output_dir = config["output_dir"]
experiment_name = config["experiment_name"]
full_output_dir = f"{output_dir}/{experiment_name}"
LGBM_MODEL_FILENAME = "lgbm_model.pkl"

def main():
    print("1. Veri seti yükleniyor ve ayrılıyor...")
    train_df, test_df = prepare_data(data_file_path)
    
    print(f"2. {BASE_MODEL} kullanılarak Train verisinden özellik vektörleri (embeddings) çıkarılıyor... (Bu biraz sürebilir)")
    X_train = extract_embeddings(train_df["text"])
    y_train = train_df["label"].values
    
    print(f"3. {BASE_MODEL} kullanılarak Test verisinden özellik vektörleri çıkarılıyor...")
    X_test = extract_embeddings(test_df["text"])
    y_test = test_df["label"].values
    
    print("4. LightGBM Modeli eğitiliyor...")
    # LightGBM parametrelerini buradan ayarlayabilirsin
    lgbm = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    lgbm.fit(X_train, y_train)
    
    print("5. Test seti üzerinde tahminler yapılıyor ve metrikler hesaplanıyor...")
    y_pred = lgbm.predict(X_test)
    
    print("\n--- LightGBM Model Performansı ---")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score : {f1_score(y_test, y_pred):.4f}")
    print("----------------------------------\n")
    
    print("6. Eğitilen LightGBM modeli kaydediliyor...")
    os.makedirs(f"./{full_output_dir}/", exist_ok=True)
    joblib.save(lgbm, f"./{full_output_dir}/{LGBM_MODEL_FILENAME}")
    
    print("7. Kaydedilen hibrit model test ediliyor...")
    bert_model, tokenizer, saved_lgbm = load_hybrid_model()
    
    example = """Verify your wallet seed phrase on our new Web3 portal to unlock your exclusive Airdrop. 
    Complete the verification process now or risk losing your pending transactions."""
    
    print(f"Örnek Metin: {example}")
    print("Tahmin Sonucu:")
    print(get_prediction(bert_model, tokenizer, saved_lgbm, example))

if __name__ == "__main__":
    main()
import os
# Teknik uyarıları ve çakışmaları en başta engelliyoruz
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import joblib
import warnings
import logging
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score
from utilities import prepare_data, extract_embeddings, load_config

# Gereksiz kütüphane loglarını susturuyoruz
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

def main():
    # Ayarları yükle
    config = load_config()
    data_file_path = config["data_file_path"]
    full_output_dir = config["output_dir"]
    LGBM_MODEL_FILENAME = config["lgbm_model_filename"]

    print("\n" + "="*40)
    print("  WEB3 PHISHING DETECTION SYSTEM (HİBRİT)")
    print("="*40)
    
    # 1. Veri Hazırlama
    print("\n[1/4] Veri seti yükleniyor...")
    train_df, test_df = prepare_data(data_file_path)
    
    # 2. BERT Özellik Çıkarımı
    print("[2/4] BERT (DistilBERT) ile metin analizi yapılıyor...")
    # BERT mesajları okuyup sayısal vektörlere dönüştürür
    X_train = extract_embeddings(train_df["text"].tolist())
    X_test = extract_embeddings(test_df["text"].tolist())
    
    # 3. LightGBM Eğitimi
    print("[3/4] Hibrit Model (LightGBM) eğitiliyor...")
    lgbm = LGBMClassifier(verbose=-1, n_estimators=100)
    lgbm.fit(X_train, train_df["label"])
    
    # 4. Değerlendirme
    print("[4/4] Performans testleri tamamlandı.")
    
    y_pred = lgbm.predict(X_test)
    
    print("\n" + "-"*30)
    print("      MODEL BAŞARI RAPORU")
    print("-"*30)
    print(f"Doğruluk (Accuracy) : %{accuracy_score(test_df['label'], y_pred)*100:.2f}")
    print(f"F1-Skoru            : {f1_score(test_df['label'], y_pred):.4f}")
    print("-"*30)

    # Modeli Kaydet (Hata verdiğimiz dump kısmını düzelttim)
    if not os.path.exists(full_output_dir):
        os.makedirs(full_output_dir)
        
    joblib.dump(lgbm, f"./{full_output_dir}/{LGBM_MODEL_FILENAME}")
    print(f"\nModel başarıyla kaydedildi: {LGBM_MODEL_FILENAME}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
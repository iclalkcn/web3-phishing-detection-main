# Web3 Odaklı Hibrit Kimlik Avı Tespit Sistemi
Bu çalışma, Web3 ve kripto para ekosistemine özgü sofistike 
oltalama (phishing) saldırılarını tespit etmek amacıyla geliştirilmiş 
hibrit bir makine öğrenmesi mimarisidir. Sistem, metin verilerinin anlamsal 
analizi için Transformer tabanlı DistilBERT modelini, nihai sınıflandırma kararı 
için ise gradyan artırma temelli LightGBM algoritmasını kullanmaktadır.

## 1. Mimari Genel Bakış

Sistem, doğal dil işleme (NLP) ve geleneksel makine öğrenmesi yöntemlerini birleştiren iki ana aşamadan oluşmaktadır:

Aşama 1: Özellik Çıkarımı (Feature Extraction)
DistilBERT modeli kullanılarak ham metin verileri 768 boyutlu yoğun 
vektörlere (embeddings) dönüştürülür. Bu süreç, "seed phrase", "airdrop" veya "mint"
gibi terimlerin bağlamsal ilişkilerinin korunmasını sağlar.

Aşama 2: Sınıflandırma (Classification)
Elde edilen anlamsal vektörler, LightGBM sınıflandırıcısına girdi olarak verilir.
Bu hibrit yapı, BERT'in anlama yeteneği ile LightGBM'in hız ve düşük hata oranını birleştirir.

## 2. Veri Kümesi Yapısı (Data Fusion)

Modelin genelleme kabiliyetini artırmak amacıyla üç farklı platformdan toplanan veriler birleştirilmiştir:

Discord: Topluluk odaklı bot mesajları ve sosyal mühendislik içerikleri.

Telegram: Agresif oltalama tekniklerini içeren kısa ve dinamik metinler.

CryptoScamDB: Twitter ve Web3 tabanlı, doğrulanmış gerçek dünya saldırı vakaları.

### 3. Teknik Kurulum

Projenin çalıştırılabilmesi için gerekli kütüphaneler aşağıdaki komut ile yüklenmelidir:

Bash
pip install pandas numpy torch scikit-learn lightgbm transformers joblib pyyaml
Dizin Hiyerarşisi

```markdown
ml_dev/
├── data/
│   ├── dataset.csv
│   ├── discord-message-phishing.csv
│   └── uris.csv
├── merge_web3_data.py
├── train_and_save_model.py
└── utilities.py
```

### 4. Kullanım Klavuzu

Adım 1: Veri Ön İşleme
Farklı kaynaklardan gelen verileri harmonize etmek ve Web3 odaklı anahtar kelime filtrelemesi uygulamak için:

Bash
python merge_web3_data.py
Adım 2: Model Eğitimi
Hibrit modeli eğitmek ve eğitilmiş parametreleri .pkl formatında dışa aktarmak için:

Bash
python train_and_save_model.py

### 5. Performans Analizi

* F1 score: %94.13
* Accuracy: 0.8367
* Temel Model: distilbert-base-uncased
* Sınıflandırıcı: LightGBM (100 Estimators)

## 6. Lisans

Bu proje MIT Lisansı kapsamında sunulmaktadır. Daha fazla bilgi için LICENSE dosyasına bakınız.

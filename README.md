Web3 Odaklı Hibrit Kimlik Avı Tespit Sistemi (BERT + LightGBM)

Bu proje, Web3 ve kripto para ekosistemine özgü oltalama (phishing) saldırılarını tespit etmek amacıyla geliştirilmiş hibrit bir makine öğrenmesi modelidir. Sistem, metin analizi için Transformer tabanlı DistilBERT modelini, sınıflandırma aşaması için ise yüksek performanslı LightGBM algoritmasını kullanmaktadır.

Proje Mimarisi
Sistem iki ana aşamadan oluşmaktadır:
Özellik Çıkarımı (Feature Extraction): DistilBERT modeli, ham metin verilerini 768 boyutlu anlamsal vektörlere (embeddings) dönüştürür. Bu aşama, Web3 jargonunun (seed phrase, airdrop, mint vb.) bağlamsal olarak anlaşılmasını sağlar.
Sınıflandırma (Classification): BERT tarafından üretilen vektörler, LightGBM algoritmasına girdi olarak verilir. LightGBM, bu karmaşık veri setinde hızlı ve optimize edilmiş bir karar mekanizması yürüterek mesajın "Güvenli" (Genuine) veya "Kimlik Avı" (Phishing) olduğuna karar verir.

Veri Kümesi Özellikleri (Data Fusion)
Modelin genelleme yeteneğini artırmak amacıyla üç farklı platformdan toplanan veriler birleştirilmiştir
:Discord: Topluluk tabanlı saldırı dilleri ve bot mesajları.
:Telegram: Agresif oltalama teknikleri ve kısa spam mesajları.
CryptoScamDB (Twitter/Web3): Doğrudan cüzdan hırsızlığına yönelik gerçek dünya vakaları.

Kurulum ve Hazırlık
Gereksinimler
Projenin çalışması için aşağıdaki kütüphanelerin yüklü olması gerekmektedir:
Bash
pip install pandas numpy torch scikit-learn lightgbm transformers joblib pyyaml

Dizin Yapısı
Veri dosyalarının ml_dev/data/ dizini altında bulunduğundan emin olun:
dataset.csv (Genel veri havuzu)
discord-message-phishing - Sheet1.csv (Discord verileri)
uris.csv (Twitter/Web3 verileri)

Kullanım Adımları
Sistemi eğitmek ve çalıştırmak için aşağıdaki işlem sırası takip edilmelidir:
1. Veri Birleştirme ve Filtreleme
Farklı kaynaklardan gelen verileri standardize etmek ve sadece Web3 odaklı içerikleri süzmek için birleştirme betiğini çalıştırın:
Bash
python merge_web3_data.py
Bu işlem sonucunda data/web3_pure_dataset.csv dosyası oluşturulacaktır.

2. Model Eğitimi ve Kayıt
Hibrit modeli eğitmek ve eğitilmiş ağırlıkları .pkl formatında kaydetmek için ana eğitim betiğini çalıştırın:
Bash
python train_and_save_model.py

          Performans Analizi
Metrik                        Değer
Doğruluk Oranı (Accuracy)     %94.13
F1-Skoru                      0.8367
Kullanılan Temel Model        distilbert-base-uncased
Sınıflandırıcı                LightGBM


Lisans
Bu proje MIT Lisansı altında lisanslanmıştır. Daha fazla bilgi için LICENSE dosyasına bakınız.

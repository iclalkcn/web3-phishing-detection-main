import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Senin terminalden aldığın gerçek test sonuçları
# [TN, FP]
# [FN, TP]
cm = np.array([[552, 9],   # Güvenli Mesajlar (Genuine)
               [32, 105]]) # Saldırı Mesajları (Phishing)

plt.figure(figsize=(8, 6))

# Görselleştirme ayarları
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 16},
            xticklabels=['GENUINE', 'PHISHING'], 
            yticklabels=['GENUINE', 'PHISHING'])

# Akademik başlıklar
plt.title('Web3 Phishing Detection - Confusion Matrix', fontsize=14, pad=20)
plt.ylabel('Gerçek Sınıf (Actual)', fontsize=12)
plt.xlabel('Tahmin Edilen Sınıf (Predicted)', fontsize=12)

# Görseli yüksek çözünürlükte rapor için kaydet
plt.savefig('confusion_matrix_final.png', dpi=300, bbox_inches='tight')
print("Görsel 'confusion_matrix_final.png' adıyla başarıyla kaydedildi!")
plt.show()
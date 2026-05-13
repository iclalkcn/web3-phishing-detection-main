import pandas as pd
import os

# Veri yollarını belirliyoruz
data_dir = 'data'
keywords = ['crypto', 'wallet', 'eth', 'bnb', 'solana', 'metamask', 'airdrop', 'seed phrase', 'nft', 'binance', 'token']

def merge_data():
    print("--- Veri Birleştirme İşlemi Başladı ---")

    # 1. Genel dataset.csv'den Web3 olanları çek (Telegram/Genel)
    df_genel = pd.read_csv(os.path.join(data_dir, 'dataset.csv'))
    df_web3_tg = df_genel[df_genel['text'].str.contains('|'.join(keywords), case=False, na=False)]
    df_web3_tg['label'] = df_web3_tg['text_type'].apply(lambda x: 1 if x == 'spam' else 0)
    df_web3_tg = df_web3_tg[['text', 'label']]
    print(f"Telegram/Genel veriden {len(df_web3_tg)} Web3 mesajı filtrelendi.")

    # 2. Discord Verilerini Al
    df_discord = pd.read_csv(os.path.join(data_dir, 'discord-message-phishing%20-%20Sheet1.csv'))
    df_discord = df_discord[['msg_content', 'lable']].rename(columns={'msg_content': 'text', 'lable': 'label'})
    print(f"Discord verisinden {len(df_discord)} mesaj eklendi.")

    # 3. URIs (Twitter/CryptoScam) Verilerini Al
    df_uris = pd.read_csv(os.path.join(data_dir, 'uris.csv'))
    df_uris['label'] = df_uris['category'].apply(lambda x: 1 if x in ['Scamming', 'Phishing'] else 0)
    df_uris = df_uris[['description', 'label']].rename(columns={'description': 'text'})
    print(f"URIs verisinden {len(df_uris)} mesaj eklendi.")

    # 4. Hepsini Birleştir ve Temizle
    final_df = pd.concat([df_web3_tg, df_discord, df_uris], ignore_index=True)
    final_df = final_df.dropna().drop_duplicates(subset=['text'])
    
    # Yeni dosyayı kaydet
    output_path = os.path.join(data_dir, 'web3_pure_dataset.csv')
    final_df.to_csv(output_path, index=False)
    print(f"\nBAŞARILI! Toplam {len(final_df)} satırlık Web3 veri seti oluşturuldu: {output_path}")

if __name__ == "__main__":
    merge_data()
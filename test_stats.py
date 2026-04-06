import pandas as pd
import numpy as np

# Mevcut test verisini yükle
df = pd.read_csv("test_dataset.csv")

# olcum_degeri sütununun %5'ini rastgele seçip NaN (boş) yapalım
prop_missing = 0.05
mask = np.random.rand(len(df)) < prop_missing
df.loc[mask, 'olcum_degeri'] = np.nan

print(f"Toplam boş bırakılan satır sayısı: {df['olcum_degeri'].isna().sum()}")

# Test için yeni dosya
df.to_csv("test_dataset_missing.csv", index=False)
print("test_dataset_missing.csv oluşturuldu. Uygulamaya bunu yükle.")
# 🔬 Data-Autopsy

> Veri bilimindeki "kirli ve yorucu" işleri çözen profesyonel masaüstü uygulaması.

**Dil / Language:** TR | EN (uygulama içinden değiştirilebilir)

---

## 📦 Kurulum

```bash
# 1. Repoyu klonla
git clone https://github.com/yourname/data-autopsy
cd data-autopsy

# 2. Sanal ortam oluştur (önerilir)
python -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

# 3. Bağımlılıkları yükle
pip install -r requirements.txt

# 4. Çalıştır
python main.py
```

---

## 🗂 Proje Yapısı

```
data-autopsy/
│
├── main.py                    # Flet uygulama giriş noktası
├── requirements.txt
│
├── core/                      # Altyapı katmanı
│   ├── i18n_manager.py        # Dil yöneticisi (TR/EN)
│   ├── translations.py        # Çeviri sözlükleri
│   ├── database.py            # DuckDB motoru
│   └── audit_logger.py        # Metodolojik audit trail
│
├── modules/                   # Analiz modülleri
│   ├── normalizer.py          # Türkçe normalizasyon
│   ├── statistical_auditor.py # Benford, IQR/Z-Score, Varyans
│   ├── fuzzy_matcher.py       # Olasılıksal eşleştirme
│   └── smart_imputer.py       # Akıllı eksik veri doldurma
│
├── ui/
│   └── views.py               # Flet panel görünümleri
│
├── audits/                    # Otomatik oluşturulur — audit logları
├── reports/                   # Otomatik oluşturulur — raporlar
└── data/                      # Örnek veri dosyaları
```

---

## 🧩 Modüller

### 1. Normalizasyon Modülü
- **Encoding tespiti**: chardet + charset-normalizer + manuel fallback zinciri (UTF-8 → Windows-1254 → ISO-8859-9 → CP857 → Latin-1)
- **Türkçe büyük/küçük harf**: `str.upper()` / `str.lower()` yerine, `i/İ` ve `I/ı` ayrımını doğru yapan özel fonksiyonlar
- **Pipeline mimarisi**: `SeriesNormalizer` ile zincirleme işlem

### 2. İstatistiksel Denetçi
- **Benford Yasası**: Ki-kare iyilik uyum testi + Nigrini MAD kriteri
- **Aykırı Değer**: IQR (Tukey) + Robust Z-Score (Modified Z-Score, Iglewicz & Hoaglin 1993)
- **Varyans Etkisi**: One-Way ANOVA, Eta-kare (η²), Omega-kare (ω²), Cohen etki büyüklüğü yorumu

### 3. Olasılıksal Eşleştirici
- **Algoritmalar**: Levenshtein, Jaro-Winkler, Token Sort Ratio
- **Kütüphane**: rapidfuzz (10-100x hız avantajı)
- **Review zone**: Eşik etrafında "inceleme gerekli" bölgesi

### 4. Akıllı Veri Doldurucu
- **Desen tespiti**: MCAR / MAR / MNAR buluşsal sınıflandırma
- **Yöntem önerisi**: Sütun tipi + desen + çarpıklık analizine dayalı otomatik öneri
- **Yöntemler**: Ortalama, Medyan, Mod, Sabit, Forward/Backward Fill, KNN

### 5. Audit Trail & Raporlama
- Her işlem JSONL formatında kayıt
- Oturum bazında metodolojik kalite raporu (JSON)
- Kalite skoru: 100 - (hata × 10) - (uyarı × 3)

---

## 🌐 i18n Kullanımı

```python
from core.i18n_manager import I18nManager

i18n = I18nManager(initial_language="tr")
print(i18n.t("menu_normalize"))  # → "Normalleştir"

i18n.set_language("en")
print(i18n.t("menu_normalize"))  # → "Normalize"

# Dil değişiminde UI güncelleme callback'i
def on_lang_change(new_lang):
    header.value = i18n.t("app_title")
    page.update()

i18n.on_language_change(on_lang_change)
```

---

## 📝 Lisans
MIT

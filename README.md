# CarDekho İkinci El Araç Fiyat Tahmini

CarDekho ikinci el araç ilanları üzerinde uçtan uca bir regresyon çalışması. Veri temizleme ve aykırı değer analizinden, veri sızıntısına karşı korumalı kategorik kodlamaya ve `RandomizedSearchCV` ile hiperparametre optimizasyonuna kadar tüm adımlar tek bir Jupyter Notebook içinde yürütülür. Hedef, bir aracın teknik ve ilan özelliklerinden satış fiyatını (`selling_price`) tahmin etmektir.

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-data%20wrangling-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-AdaBoost-F7931E?logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Veri Seti

Kaynak: Kaggle — CarDekho ikinci el araç ilanları
[used-cars-dataset-cardekho](https://www.kaggle.com/datasets/sukritchatterjee/used-cars-dataset-cardekho) · [cardekho-cleaned](https://www.kaggle.com/datasets/sharooqfarzeenak/cardekho-cleaned)

| | |
|---|---|
| Dosya | `17-cardekho.csv` |
| Ham boyut | **15.411 satır × 14 sütun** (indeks sütunu `Unnamed: 0` dahil) |
| Temizlik sonrası | **15.240 satır** |
| Hedef değişken | `selling_price` (sürekli) |
| Eksik değer | Yok (`isnull().sum()` tüm sütunlarda 0) |

**Özellikler**

| Sütun | Tip | Açıklama |
|---|---|---|
| `car_name` | kategorik (119 benzersiz) | Marka + model birleşimi |
| `brand` | kategorik (30 benzersiz) | Marka |
| `model` | kategorik (118 benzersiz) | Model |
| `vehicle_age` | sayısal | Araç yaşı (yıl) |
| `km_driven` | sayısal | Toplam kilometre |
| `seller_type` | kategorik (3) | Individual / Dealer / Trustmark Dealer |
| `fuel_type` | kategorik (5) | Petrol / Diesel / CNG / LPG / Electric |
| `transmission_type` | kategorik (2) | Manual / Automatic |
| `mileage` | sayısal | Yakıt tüketimi (km/l) |
| `engine` | sayısal | Motor hacmi (cc) |
| `max_power` | sayısal | Maksimum güç (bhp) |
| `seats` | sayısal | Koltuk sayısı |
| `selling_price` | sayısal | **Hedef** — satış fiyatı |

---

## Yöntem / İş Akışı

### 1. Veri Temizleme
- Anlamsız indeks sütunu `Unnamed: 0` kaldırıldı.
- **167 tam tekrar eden satır** silindi (`drop_duplicates`) → 15.411 ➜ 15.244.
- `seats = 0` olan 2 geçersiz kayıt, dağılımın modu olan **5** ile düzeltildi (silme yerine düzeltme tercih edildi).

### 2. Aykırı Değer Analizi
Dağılımlar `scatterplot` ve `boxplot` ile incelendi; temizlik hedef değişkene öncelik verilerek yapıldı:
- `selling_price ≥ 15.000.000` olan uç kayıtlar çıkarıldı (maksimum 39.500.000 idi) → 15.242 satır.
- `km_driven ≥ 1.000.000` olan fiziksel olarak anlamsız kayıtlar çıkarıldı (maksimum 3.800.000 idi) → **15.240 satır**.

### 3. Keşifsel Analiz
`selling_price` ile sayısal değişkenlerin korelasyonu:

| Değişken | Korelasyon |
|---|---|
| `max_power` | **+0.773** |
| `engine` | **+0.612** |
| `seats` | +0.135 |
| `km_driven` | −0.110 |
| `vehicle_age` | −0.259 |
| `mileage` | −0.319 |

Ayrıca `engine` ↔ `max_power` arasında yüksek çoklu doğrusallık (0.807) gözlendi.

### 4. Train/Test Ayrımı — Sızıntıya Karşı Önlem
Veri **önce** `train_test_split(test_size=0.3, random_state=15)` ile ayrıldı (10.668 eğitim / 4.572 test), kodlamalar **sonra** uygulandı. Böylece test setinin istatistikleri eğitim sürecine sızmaz.

### 5. Kategorik Kodlama — Kardinaliteye Göre Strateji
Tek bir kodlayıcı yerine sütunun benzersiz değer sayısına göre iki farklı yaklaşım seçildi:

- **Frequency Encoding** → `car_name`, `brand`, `model` (yüksek kardinalite).
  Frekanslar **yalnızca eğitim setinden** hesaplandı; test setinde görülmeyen kategoriler ortalama frekansla dolduruldu. Ordinal kodlama, bu sütunlarda doğal bir sıralama bulunmadığı için bilinçli olarak elendi.
- **One-Hot Encoding** → `seller_type`, `fuel_type`, `transmission_type` (düşük kardinalite), `drop="first"` ile kukla değişken tuzağından, `handle_unknown="ignore"` ile bilinmeyen kategori hatalarından korunularak.

Tüm dönüşümler `ColumnTransformer` (`remainder="passthrough"`) ile tek bir boru hattında toplandı. Sonuç: **16 özellikli** tamamen sayısal matris.

### 6. Modelleme ve Hiperparametre Optimizasyonu
`AdaBoostRegressor` üç aşamada geliştirildi:
1. Varsayılan ayarlarla temel model (baseline).
2. `RandomizedSearchCV` (`cv=5`, `scoring="r2"`) ile `n_estimators`, `learning_rate` ve `loss` araması.
3. Temel öğrenici `DecisionTreeRegressor` olarak açıkça tanımlanıp arama uzayına `estimator__max_depth` eklenerek derinlik optimizasyonu.

Değerlendirme metrikleri: **R²**, **MSE**, **MAE**.

---

## Sonuçlar

Test seti (4.572 kayıt) üzerindeki karşılaştırma:

| Model | R² | MAE | MSE |
|---|---:|---:|---:|
| AdaBoost (varsayılan) | 0.6525 | 321.845 | 187.341.821.862 |
| AdaBoost + RandomizedSearchCV | 0.6971 | 224.276 | 145.832.154.178 |
| **AdaBoost + DecisionTree(max_depth) + RandomizedSearchCV** | **0.8804** | **148.123** | **68.025.255.022** |

**En iyi hiperparametreler:** `n_estimators=100`, `loss="linear"`, `learning_rate=0.1`, `estimator__max_depth=5`

Öne çıkan bulgu: AdaBoost'un varsayılan temel öğrenicisi `max_depth=3` ile sınırlıdır. Temel öğrenicinin derinliğini de arama uzayına dahil etmek, yalnızca boosting parametrelerini aramaya kıyasla R²'yi **0.70'ten 0.88'e** taşımış, ortalama mutlak hatayı ise **%34 azaltmıştır**. Değer, modelin kendi hiperparametrelerinden çok, temel öğrenicinin kapasitesinde ortaya çıkmıştır.

> Not: Metrikler notebook'taki `r2_score(y_pred, y_test)` çağrım sırasıyla raporlanmıştır.

---

## Kurulum ve Çalıştırma

```bash
# 1. Depoyu klonlayın
git clone https://github.com/<kullanici-adi>/Cardekho-Data.git
cd Cardekho-Data

# 2. Sanal ortam oluşturun
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Bağımlılıkları kurun
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# 4. Notebook'u açın
jupyter notebook Cardekho_dataset_regression.ipynb
```

Notebook, `17-cardekho.csv` dosyasını çalışma dizininden okur; hücreleri baştan sona sırayla çalıştırmanız yeterlidir. Geliştirme ortamı: **Python 3.13.5**.

---

## Dosya Yapısı

```
Cardekho-Data/
├── Cardekho_dataset_regression.ipynb   # Uçtan uca analiz ve modelleme
├── 17-cardekho.csv                     # Ham veri seti (15.411 satır)
├── .gitignore
├── LICENSE
└── README.md
```

---

## Lisans

Bu proje **MIT Lisansı** ile lisanslanmıştır. Ayrıntılar için [LICENSE](LICENSE) dosyasına bakabilirsiniz.

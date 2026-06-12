# Eksperimen_SML_Siswa

Repository eksperimen preprocessing dataset California Housing untuk submission kelas Membangun Sistem Machine Learning.

## Struktur Repository

```
Eksperimen_SML_Siswa/
├── .github/
│   └── workflows/
│       └── preprocessing.yml       # GitHub Actions workflow (Advanced)
├── preprocessing/
│   ├── Eksperimen_Siswa.ipynb      # Notebook eksperimen EDA & preprocessing
│   ├── automate_Siswa.py           # Script otomatisasi preprocessing (Skilled)
│   └── housing_preprocessing/      # Output: data yang sudah diproses
│       ├── train.csv
│       └── test.csv
└── housing_raw.csv                 # Dataset mentah
```

## Dataset

**California Housing Dataset**
- Sumber: `sklearn.datasets.fetch_california_housing`
- Jumlah data: 20,640 baris × 9 kolom
- Task: Regresi (prediksi median harga rumah)

| Fitur | Deskripsi |
|-------|-----------|
| MedInc | Median income in block group |
| HouseAge | Median house age in block group |
| AveRooms | Average number of rooms per household |
| AveBedrms | Average number of bedrooms per household |
| Population | Block group population |
| AveOccup | Average household occupancy |
| Latitude | Block group latitude |
| Longitude | Block group longitude |
| **MedHouseVal** | **Target: Median house value (100k USD)** |

## Tahapan Preprocessing

1. **Data Loading** — Load dari sklearn, simpan sebagai CSV
2. **EDA** — Statistik deskriptif, missing values, duplikat, distribusi, korelasi, outlier
3. **Handle Missing Values** — Mean Imputation (SimpleImputer)
4. **Handle Outliers** — IQR Clipping
5. **Feature Scaling** — StandardScaler
6. **Train-Test Split** — 80:20, random_state=42

## Cara Menjalankan

### Notebook Eksperimen
```bash
cd preprocessing
jupyter notebook Eksperimen_Siswa.ipynb
```

### Script Otomatis
```bash
cd preprocessing
python automate_Siswa.py
```

### GitHub Actions
Workflow berjalan otomatis setiap push ke branch `main`.

# Analisis ISPU DKI Jakarta 2022

Proyek analisis **Indeks Standar Pencemaran Udara (ISPU)** DKI Jakarta Tahun 2022 menggunakan kerangka metodologi **CRISP-DM** (Cross-Industry Standard Process for Data Mining).

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Deskripsi Proyek

Dataset berisi data kualitas udara harian dari **5 stasiun pemantauan (DKI1–DKI5)** di Jakarta sepanjang tahun 2022, dengan 6 parameter polutan:

| Parameter | Keterangan |
|---|---|
| **PM10** | Partikel berdiameter ≤ 10 mikrometer |
| **PM2.5** | Partikel berdiameter ≤ 2.5 mikrometer |
| **SO2** | Sulfur Dioksida |
| **CO** | Karbon Monoksida |
| **O3** | Ozon |
| **NO2** | Nitrogen Dioksida |

**Target**: Klasifikasi kategori kualitas udara → `BAIK` / `SEDANG` / `TIDAK SEHAT`

---

## Struktur Proyek

```
Data-Mining/
│
├── notebook.ipynb              ← Analisis CRISP-DM lengkap (6 tahap)
├── requirements.txt            ← Daftar dependensi Python
│
├── dataset/
│   └── Filedata Indeks Standar Pencemaran Udara ISPU Tahun 2022.csv
│
├── dashboard/
│   └── dashboard.py               ← Dashboard Streamlit interaktif
│
└── model/                      ← Dibuat otomatis setelah training
    ├── best_model.pkl             ← Model ML terbaik (terlatih)
    ├── scaler.pkl                 ← StandardScaler
    ├── label_encoder.pkl          ← LabelEncoder
    └── model_info.json            ← Metadata & metrik model
```

---

## Cara Menjalankan

### Prasyarat
- Python **3.10** atau lebih baru
- Git

---

### 1. Clone Repository

```bash
git clone https://github.com/<username>/Data-Mining.git
cd Data-Mining
```

---

### 2. Buat Virtual Environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 3. Install Dependensi

```bash
pip install -r requirements.txt
```

---

### 4. Latih Model (Wajib — sebelum menjalankan dashboard)

**Opsi A: Gunakan script Python (cepat, tanpa Jupyter)**
```bash
python run_analysis.py
```

**Opsi B: Jalankan Notebook (lengkap + visualisasi)**

Buka `notebook.ipynb` di **VS Code** atau **Jupyter Lab**, lalu klik **Run All**.

```bash
# Jika menggunakan Jupyter Lab:
jupyter lab notebook.ipynb
```

Setelah selesai, folder `model/` akan berisi file `.pkl` yang diperlukan dashboard.

---

### 5. Jalankan Dashboard

```bash
streamlit run dashboard/dashboard.py
```

Buka browser dan akses: **http://localhost:8501**

---

## Dependensi Utama

| Package | Versi | Fungsi |
|---|---|---|
| `pandas` | ≥ 2.0 | Manipulasi data |
| `numpy` | ≥ 1.24 | Operasi numerik |
| `matplotlib` | ≥ 3.7 | Visualisasi statis |
| `seaborn` | ≥ 0.12 | Visualisasi statistik |
| `scikit-learn` | ≥ 1.3 | Model ML & preprocessing |
| `imbalanced-learn` | ≥ 0.11 | SMOTE augmentasi data |
| `statsmodels` | ≥ 0.14 | Analisis statistik (OLS trendline) |
| `streamlit` | ≥ 1.28 | Dashboard web interaktif |
| `plotly` | ≥ 5.18 | Chart interaktif |
| `joblib` | ≥ 1.3 | Serialisasi model |
| `ipykernel` | ≥ 6.0 | Kernel Jupyter Notebook |

---

## Temuan Utama

1. **Polutan dominan**: PM2.5 adalah pencemar kritis yang paling sering muncul (~75% hari)
2. **Bulan terburuk**: Juni–September (musim kemarau) → kualitas udara paling buruk
3. **Korelasi tinggi**: PM10 & PM2.5 berkorelasi sangat kuat (r > 0.85)
4. **Lokasi terburuk**: DKI4 paling sering mencatat status TIDAK SEHAT

---

## Dataset

Dataset ISPU DKI Jakarta Tahun 2022 tersedia di folder `dataset/`. Data bersumber dari **Open Data Jakarta** (data.jakarta.go.id).

---

##  Metodologi: CRISP-DM

```
1. Business Understanding  →  Rumusan masalah & tujuan
2. Data Understanding      →  EDA & identifikasi anomali
3. Data Preprocessing      →  Cleaning, SMOTE, encoding, scaling
4. Modelling               →  Decision Tree, Random Forest, KNN
5. Evaluation              →  Confusion Matrix, F1-Score, CV
6. Deployment              →  Streamlit Dashboard
```

---

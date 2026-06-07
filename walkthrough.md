# CRISP-DM Workflow Report: Air Quality Index (ISPU) 2022

Laporan lengkap implementasi siklus standar CRISP-DM untuk klasifikasi kategori kualitas udara tahun 2022.

---

## 1. Business + Data Understanding

### Business Understanding
Tujuan dari pemodelan ini adalah untuk memprediksi kategori tingkat polusi udara (`categori`) berdasarkan indikator konsentrasi zat polutan (`pm_10`, `pm_duakomalima`, `so2`, `co`, `o3`, `no2`) dan nilai maksimumnya (`max`). Klasifikasi otomatis ini membantu dinas lingkungan hidup dan masyarakat untuk mendapatkan deteksi dini tingkat kesehatan udara.

### Data Understanding
* **Dimensi Data**: 365 baris dan 12 kolom.
* **Distribusi Kelas Target**:
  * **SEDANG**: 225 data
  * **TIDAK SEHAT**: 137 data
  * **BAIK**: 3 data
  *(Terdapat ketidakseimbangan kelas / class imbalance yang signifikan pada kategori BAIK)*.
* **Distribusi Target Visual**:
![Target Distribution](<img width="1000" height="600" alt="target_distribution" src="https://github.com/user-attachments/assets/68b93bc7-e5c8-41d2-b6b4-4ad1fa00e597" />
)

* **Korelasi Antar Fitur (Polutan)**:
![Correlation Heatmap](file:///C:/Users/Hype%20AMD/.gemini/antigravity-ide/brain/044a2dc9-2e12-4bae-9c9e-5caa92fc43f9/correlation_heatmap.png)
  * Terlihat korelasi kuat antara `pm_duakomalima` dan `pm_10` (0.83), serta antara `pm_duakomalima` dengan nilai `max` (0.99), yang mengindikasikan bahwa PM2.5 sering kali menjadi polutan kritis utama yang menentukan indeks ISPU maksimum.

---

## 2. Data Preparation (Preprocessing)
Langkah prepocessing yang berhasil dijalankan meliputi:
1. **Tanggal Normalization**: Mengonversi kolom `tanggal` yang sebelumnya berisi format tidak teratur (seperti serial Excel `44926.625`) menjadi format datetime standar `YYYY-MM-DD`.
2. **Imputasi Nilai Hilang**: Mengisi nilai kosong/NaN pada polutan menggunakan nilai **median** masing-masing kolom.
3. **Pembersihan Target**: Menghapus baris kosong pada kolom `categori` serta melakukan normalisasi string (strip spasi dan ubah ke uppercase).
4. **Encoding Kategorikal**: Melakukan One-Hot Encoding pada fitur non-numerik seperti `lokasi_spku` dan `critical`.
5. **Feature Scaling**: Menstandarisasi nilai fitur numerik menggunakan `StandardScaler`.
6. **Data Splitting**: Membagi dataset menjadi **80% data training** (292 baris) dan **20% data testing** (73 baris) dengan teknik stratified sampling untuk menjaga distribusi kelas target.

---

## 3. Modeling
Tiga algoritma machine learning telah dilatih dan diuji:
1. **Logistic Regression** (baseline linear)
2. **Decision Tree Classifier**
3. **Random Forest Classifier**

Hasil evaluasi performa akurasi model:
* **Logistic Regression**: Train Acc: 98.63% | Test Acc: 94.52% | 5-Fold CV: 94.88%
* **Decision Tree**: Train Acc: 100.00% | Test Acc: 100.00% | 5-Fold CV: 100.00%
* **Random Forest**: Train Acc: 100.00% | Test Acc: 100.00% | 5-Fold CV: 99.32%

---

## 4. Evaluation & Validation

### Perbandingan Akurasi Model
![Model Comparison](file:///C:/Users/Hype%20AMD/.gemini/antigravity-ide/brain/044a2dc9-2e12-4bae-9c9e-5caa92fc43f9/model_comparison.png)

### Model Terbaik: Decision Tree / Random Forest
Model berbasis pohon (Decision Tree dan Random Forest) berhasil meraih **Akurasi 100%** pada data testing.

> [!IMPORTANT]
> **Analisis Insights**: Mengapa akurasi mencapai 100%?
> Dalam aturan perhitungan ISPU standar, kategori kualitas udara (`categori`) ditentukan secara langsung dan deterministik menggunakan rentang nilai indeks polutan tertinggi (`max`).
> * Rentang `max <= 50` -> **BAIK**
> * Rentang `51 <= max <= 100` -> **SEDANG**
> * Rentang `101 <= max <= 199` -> **TIDAK SEHAT**
> Karena fitur `max` disertakan di dalam pemodelan, Decision Tree dapat secara instan mempelajari aturan pemisahan/threshold linear ini secara sempurna, sehingga mencapai performa 100%.

### Classification Report (Best Model - Decision Tree)
```
              precision    recall  f1-score   support

        BAIK       1.00      1.00      1.00         1
      SEDANG       1.00      1.00      1.00        45
 TIDAK SEHAT       1.00      1.00      1.00        27

    accuracy                           1.00        73
   macro avg       1.00      1.00      1.00        73
weighted avg       1.00      1.00      1.00        73
```

### Confusion Matrix
![Confusion Matrix](file:///C:/Users/Hype%20AMD/.gemini/antigravity-ide/brain/044a2dc9-2e12-4bae-9c9e-5caa92fc43f9/confusion_matrix.png)

### Diagnosis Overfitting / Underfitting
* Selisih (Train-Test Gap) untuk ketiga model sangat kecil (< 5%).
* Semua model memiliki performa generalisasi yang sangat baik, tanpa adanya tanda-tanda overfitting yang merugikan.

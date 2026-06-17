import json

notebook_path = 'notebook.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update Cell 14 (markdown cell describing preprocessing steps)
cell_14 = nb['cells'][14]

cell_14['source'] = [
  "---\n",
  "## Tahap 3: Data Preprocessing\n",
  "\n",
  "Berdasarkan Data Understanding, ditemukan beberapa masalah yang perlu ditangani:\n",
  "\n",
  "| Masalah | Tindakan |\n",
  "|---|---|\n",
  "| Duplikasi data | Periksa dan hapus data duplikat jika ditemukan |\n",
  "| Tanggal Februari tertulis tahun 2020 | Perbaiki menjadi 2022 |\n",
  "| Tanggal baris terakhir berformat Excel serial number | Konversi ke format date |\n",
  "| Missing values pada kolom `critical` | Imputasi dengan nilai modus |\n",
  "| Nilai `0` pada `lokasi_spku` | Perlakukan sebagai NaN, imputasi modus |\n",
  "| Ketidakseimbangan kelas (BAIK hanya 2 baris) | SMOTE augmentasi |\n",
  "| Fitur kategorikal (`categori`) | Label Encoding |\n",
  "| Skala fitur berbeda | StandardScaler |\n",
  "\n",
  "### Langkah-langkah Preprocessing:\n",
  "1. **Pengecekan Duplikasi Data** — Periksa data duplikat dan hapus jika ditemukan\n",
  "2. **Fix Anomali Tanggal** — Perbaiki kesalahan penulisan tahun dan format Excel\n",
  "3. **Handle Missing Values** — Imputasi dengan modus/median\n",
  "4. **Feature Engineering** — Ekstrak fitur bulan dan musim dari tanggal\n",
  "5. **Label Encoding** — Encode variabel target ke integer\n",
  "6. **SMOTE Augmentasi** — Buat sampel sintetis untuk kelas minoritas (BAIK)\n",
  "7. **Train-Test Split** — 80:20 dengan stratified sampling\n",
  "8. **Standarisasi** — StandardScaler untuk normalisasi fitur"
]

# Update Cell 15 (code cell for Perbaikan Anomali Tanggal)
cell_15 = nb['cells'][15]

cell_15['source'] = [
  "# ============================================================\n",
  "# 3.1 Pengecekan Duplikasi Data & Perbaikan Anomali Tanggal\n",
  "# ============================================================\n",
  "\n",
  "df_clean = df.copy()\n",
  "\n",
  "# --- Pengecekan Duplikasi Data ---\n",
  "print('=== PENGECEKAN DUPLIKASI DATA ===')\n",
  "n_duplicates = df_clean.duplicated().sum()\n",
  "print(f'Jumlah data duplikat: {n_duplicates}')\n",
  "if n_duplicates > 0:\n",
  "    df_clean = df_clean.drop_duplicates().reset_index(drop=True)\n",
  "    print(f'Data duplikat berhasil dihapus! Sisa data: {len(df_clean)} baris')\n",
  "else:\n",
  "    print('Tidak ditemukan data duplikat dalam dataset.')\n",
  "print()\n",
  "\n",
  "print('=== SEBELUM PERBAIKAN TANGGAL ===')\n",
  "print('Sample tanggal Februari (periode 202202):')\n",
  "print(df_clean[df_clean['periode_data'] == 202202][['tanggal', 'periode_data']].head(3).to_string())\n",
  "print()\n",
  "\n",
  "# --- Fix 1: Tahun 2020 pada data Februari → ganti ke 2022 ---\n",
  "mask_feb = df_clean['periode_data'] == 202202\n",
  "df_clean.loc[mask_feb, 'tanggal'] = (\n",
  "    df_clean.loc[mask_feb, 'tanggal']\n",
  "    .astype(str)\n",
  "    .str.replace('2020-02', '2022-02', regex=False)\n",
  ")\n",
  "print(f'Fix 1: {mask_feb.sum()} baris Februari diperbaiki dari 2020 → 2022')\n",
  "\n",
  "# --- Fix 2: Excel serial date → format date ---\n",
  "def fix_excel_serial(val):\n",
  "    try:\n",
  "        val_float = float(str(val))\n",
  "        if val_float > 40000:  # Angka besar = kemungkinan Excel serial\n",
  "            from datetime import datetime, timedelta\n",
  "            excel_epoch = datetime(1899, 12, 30)\n",
  "            return (excel_epoch + timedelta(days=val_float)).strftime('%Y-%m-%d')\n",
  "        return str(val)\n",
  "    except (ValueError, TypeError):\n",
  "        return str(val)\n",
  "\n",
  "df_clean['tanggal'] = df_clean['tanggal'].astype(str).apply(fix_excel_serial)\n",
  "\n",
  "# --- Konversi ke datetime ---\n",
  "df_clean['tanggal'] = pd.to_datetime(df_clean['tanggal'], errors='coerce')\n",
  "\n",
  "print()\n",
  "print('=== SESUDAH PERBAIKAN TANGGAL ===')\n",
  "print(f'Tanggal tidak valid (NaT): {df_clean[\"tanggal\"].isnull().sum()}')\n",
  "print(f'Rentang tanggal: {df_clean[\"tanggal\"].min().date()} s/d {df_clean[\"tanggal\"].max().date()}')\n",
  "print()\n",
  "print('Sample tanggal Februari setelah perbaikan:')\n",
  "print(df_clean[df_clean['periode_data'] == 202202][['tanggal', 'periode_data']].head(3).to_string())"
]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Notebook updated successfully!')

import json

notebook_path = r'c:\Users\Lenovo LOQ\Documents\coding\data-mining\Data-Mining\notebook.ipynb'

with open(notebook_path, encoding='utf-8') as f:
    nb = json.load(f)

# New cell: save preprocessed dataset
new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "c21b-save-dataset",
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================\n",
        "# 3.7 Simpan Dataset Hasil Preprocessing\n",
        "# ============================================================\n",
        "\n",
        "import os\n",
        "\n",
        "# Pastikan direktori dataset tersedia\n",
        "os.makedirs('dataset', exist_ok=True)\n",
        "\n",
        "# --- Simpan df_clean (hasil preprocessing sebelum SMOTE) ---\n",
        "SAVE_PATH = 'dataset/ISPU_2022_preprocessed.csv'\n",
        "df_clean.to_csv(SAVE_PATH, index=False)\n",
        "\n",
        "print('=== DATASET HASIL PREPROCESSING BERHASIL DISIMPAN ===')\n",
        "print(f'   File     : {SAVE_PATH}')\n",
        "print(f'   Ukuran   : {os.path.getsize(SAVE_PATH) / 1024:.2f} KB')\n",
        "print(f'   Baris    : {df_clean.shape[0]}')\n",
        "print(f'   Kolom    : {df_clean.shape[1]}')\n",
        "print()\n",
        "print('=== KOLOM YANG TERSIMPAN ===')\n",
        "for col in df_clean.columns:\n",
        "    print(f'   - {col}')\n",
        "print()\n",
        "print('=== PREVIEW DATA TERSIMPAN ===')\n",
        "df_clean.head()"
    ]
}

# Find index of c21-split-scale and insert after it
insert_after = None
for i, c in enumerate(nb['cells']):
    if c.get('id', '') == 'c21-split-scale':
        insert_after = i
        break

if insert_after is None:
    print('ERROR: cell c21-split-scale not found!')
    exit(1)

nb['cells'].insert(insert_after + 1, new_cell)
print(f'Inserted new cell after index {insert_after} (c21-split-scale)')
print(f'Total cells now: {len(nb["cells"])}')

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Notebook saved successfully!')

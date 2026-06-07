import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Set style for plots
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Artifact directory path to save images
artifact_dir = r"C:\Users\Hype AMD\.gemini\antigravity-ide\brain\044a2dc9-2e12-4bae-9c9e-5caa92fc43f9"
os.makedirs(artifact_dir, exist_ok=True)

print("="*60)
print("             Siklus CRISP-DM: ISPU 2022 Dataset             ")
print("="*60)

# ==========================================
# 1. Business + Data Understanding
# ==========================================
print("\n--- 1. BUSINESS + DATA UNDERSTANDING ---")

file_path = r"c:\Users\Hype AMD\Documents\Data mining\Filedata Indeks Standar Pencemaran Udara ISPU Tahun 2022.csv"
df = pd.read_csv(file_path)

print(f"Dataset Shape: {df.shape[0]} baris, {df.shape[1]} kolom")
print("\nDataset Columns & Types:")
print(df.dtypes)

print("\nMissing Values Count per Column:")
print(df.isnull().sum())

# Kategori Air Quality Distribution
print("\nKategori Air Quality Distribution (Raw):")
print(df['categori'].value_counts(dropna=False))

# Plot target distribution
plt.figure()
sns.countplot(data=df, x='categori', order=df['categori'].value_counts().index, palette='viridis')
plt.title("Distribution of Air Quality Categories")
plt.xlabel("Category")
plt.ylabel("Count")
plt.tight_layout()
target_dist_path = os.path.join(artifact_dir, "target_distribution.png")
plt.savefig(target_dist_path)
plt.close()
print(f"Target distribution plot saved to: {target_dist_path}")

# ==========================================
# 2. Data Preparation (Preprocessing)
# ==========================================
print("\n--- 2. DATA PREPARATION ---")

# Normalisasi kolom tanggal (mengatasi serial number excel / format string)
def parse_date(x):
    if pd.isna(x) or str(x).strip() == '' or str(x).strip().lower() == 'nan':
        return pd.NaT
    val = str(x).strip()
    try:
        # Check if Excel serial float number
        val_f = float(val)
        return pd.to_datetime('1899-12-30') + pd.to_timedelta(val_f, unit='D')
    except ValueError:
        try:
            return pd.to_datetime(val)
        except:
            return pd.NaT

df['tanggal_parsed'] = df['tanggal'].apply(parse_date)
print("Sample parsed dates:")
print(df[['tanggal', 'tanggal_parsed']].head(5))

# Convert pollutant columns to numerical (coerce non-numeric to NaN)
pollutant_cols = ['pm_10', 'pm_duakomalima', 'so2', 'co', 'o3', 'no2', 'max']
for col in pollutant_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print("\nMissing values in pollutants after numeric coercion:")
print(df[pollutant_cols].isnull().sum())

# Impute missing values with Median
for col in pollutant_cols:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

# Clean Target column ('categori')
# Drop rows where 'categori' is null or invalid
df = df.dropna(subset=['categori'])
df['categori'] = df['categori'].str.strip().str.upper()

print("\nTarget categories after cleaning:")
print(df['categori'].value_counts())

# Clean 'critical' and 'lokasi_spku' columns
df['critical'] = df['critical'].fillna('UNSPECIFIED').str.strip().str.upper()
df['lokasi_spku'] = df['lokasi_spku'].fillna('UNSPECIFIED').astype(str).str.strip().str.upper()

# correlation heatmap for numerical attributes
plt.figure(figsize=(8, 6))
sns.heatmap(df[pollutant_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Air Pollutants")
plt.tight_layout()
corr_path = os.path.join(artifact_dir, "correlation_heatmap.png")
plt.savefig(corr_path)
plt.close()
print(f"Correlation heatmap saved to: {corr_path}")

# Encoding categorical columns using One-Hot encoding for features
categorical_cols = ['critical', 'lokasi_spku']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Select features (all numeric columns and encoded columns, except date/target/raw columns)
exclude_cols = ['categori', 'tanggal', 'tanggal_parsed', 'periode_data']
feature_cols = [col for col in df_encoded.columns if col not in exclude_cols]

X = df_encoded[feature_cols]
y = df['categori']

# Label encode target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)
classes_map = dict(zip(range(len(le.classes_)), le.classes_))
print(f"\nTarget Class Mapping: {classes_map}")

# Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set shape: {X_train_scaled.shape}")
print(f"Test set shape: {X_test_scaled.shape}")

# ==========================================
# 3. Modeling
# ==========================================
print("\n--- 3. MODELING ---")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

results = {}

for name, model in models.items():
    # Fit model
    model.fit(X_train_scaled, y_train)
    
    # Train accuracy
    train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
    # Test accuracy
    test_pred = model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, test_pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    cv_mean = cv_scores.mean()
    
    results[name] = {
        "model": model,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "cv_mean": cv_mean,
        "predictions": test_pred
    }
    
    print(f"\n[{name}]")
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy : {test_acc:.4f}")
    print(f"  5-Fold CV Mean: {cv_mean:.4f}")

# ==========================================
# 4. Evaluation & Validation
# ==========================================
print("\n--- 4. EVALUATION & VALIDATION ---")

# Plot accuracy comparisons
model_names = list(results.keys())
test_accuracies = [results[m]["test_accuracy"] for m in model_names]
cv_accuracies = [results[m]["cv_mean"] for m in model_names]

plt.figure(figsize=(10, 5))
x_idx = np.arange(len(model_names))
width = 0.35
plt.bar(x_idx - width/2, test_accuracies, width, label='Test Accuracy', color='b')
plt.bar(x_idx + width/2, cv_accuracies, width, label='5-Fold CV Accuracy', color='g')
plt.xticks(x_idx, model_names)
plt.ylabel('Accuracy')
plt.title('Model Performance Comparison')
plt.legend()
plt.tight_layout()
comparison_path = os.path.join(artifact_dir, "model_comparison.png")
plt.savefig(comparison_path)
plt.close()
print(f"Model comparison plot saved to: {comparison_path}")

# Select the best model based on Test Accuracy
best_model_name = max(results, key=lambda k: results[k]["test_accuracy"])
print(f"\nBest Model: {best_model_name}")

best_pred = results[best_model_name]["predictions"]
print(f"\nClassification Report for {best_model_name} (Test Set):")
print(classification_report(y_test, best_pred, target_names=le.classes_))

# Confusion Matrix for best model
cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f"Confusion Matrix - {best_model_name}")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
cm_path = os.path.join(artifact_dir, "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()
print(f"Confusion Matrix plot saved to: {cm_path}")

print("\n--- OVERFITTING / UNDERFITTING DIAGNOSIS ---")
for name in models.keys():
    diff = results[name]["train_accuracy"] - results[name]["test_accuracy"]
    print(f"[{name}]")
    print(f"  Train-Test Gap: {diff:.4f}")
    if diff > 0.10:
        print("  Status: Potential OVERFITTING (High Training Accuracy, Lower Testing Accuracy)")
    elif results[name]["train_accuracy"] < 0.60:
        print("  Status: Potential UNDERFITTING (Low Training Accuracy)")
    else:
        print("  Status: Good Generalization")

print("\n"+"="*60)
print("                  CRISP-DM Cycle Complete                   ")
print("="*60)

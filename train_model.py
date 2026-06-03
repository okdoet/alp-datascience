import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ─── 1. Load Dataset ─────────────────────────────────────────────────────────
print("Memuat dataset hantavirus...")
df = pd.read_csv('hantavirus_detection_dataset.csv')
print(f"  Data awal: {len(df)} baris")

# ─── 2. Drop Kolom Tidak Relevan & Hapus NaN ─────────────────────────────────
cols_to_drop = [
    'Patient_ID', 'Sample_Date', 'Disease_Onset_Date', 'Symptoms',
    'Hantavirus_Type', 'Disease_Severity', 'Immunological_Risk',
    'Renal_Risk', 'Liver_Injury_Score'
]
df = df.drop(columns=cols_to_drop)
df = df.dropna()
print(f"  Setelah drop kolom & NaN: {len(df)} baris")

# ─── 3. Label Encoding Fitur Kategorikal (simpan setiap encoder) ──────────────
# PENTING: encoder disimpan PER KOLOM agar app.py bisa decode balik label aslinya
categorical_cols = ['Gender', 'Region', 'Exposure_Type']
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le  # simpan encoder untuk tiap kolom

# ─── 4. IQR-3 Grouped (SEBELUM train-test split) ────────────────────────────
# Outlier dibuang per-grup (Positif / Negatif) agar distribusi tiap kelas
# tidak tercemar oleh nilai ekstrem dari kelas lain.
# Ini harus dilakukan SEBELUM split supaya:
#   a) cleaning tidak bocor info test set ke training
#   b) proporsi kelas tetap terjaga dengan benar

def remove_outliers_by_group(df, columns, target_col):
    """
    Grouped IQR-3: terapkan filter IQR 3× secara independen
    untuk tiap nilai unik pada kolom target.
    """
    df_clean = pd.DataFrame()
    for target_val in df[target_col].unique():
        group = df[df[target_col] == target_val].copy()
        for col in columns:
            Q1 = group[col].quantile(0.25)
            Q3 = group[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 3 * IQR
            upper = Q3 + 3 * IQR
            group = group[(group[col] >= lower) & (group[col] <= upper)]
        df_clean = pd.concat([df_clean, group])
    return df_clean.reset_index(drop=True)

numeric_columns = [
    'WBC_Count_K/uL', 'Platelet_Count_K/uL', 'CRP_mg/L',
    'ALT_U/L', 'AST_U/L', 'BUN_mg/dL', 'Creatinine_mg/dL'
]

df_cleaned = remove_outliers_by_group(df, numeric_columns, 'Hantavirus_Positive')
print(f"  Setelah Grouped IQR-3: {len(df_cleaned)} baris")
print(f"  Distribusi kelas: {df_cleaned['Hantavirus_Positive'].value_counts().to_dict()}")

# ─── 5. Pisahkan Fitur (X) dan Target (y) ────────────────────────────────────
X = df_cleaned.drop('Hantavirus_Positive', axis=1)
y = df_cleaned['Hantavirus_Positive']

# ─── 6. Train-Test Split (SETELAH cleaning) ──────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n  Train: {len(X_train)} baris  |  Test: {len(X_test)} baris")

# ─── 7. Standarisasi Fitur ────────────────────────────────────────────────────
# fit HANYA pada X_train, lalu transform keduanya (cegah data leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─── 8. Pelatihan Logistic Regression ────────────────────────────────────────
print("\nMelatih model Logistic Regression...")
log_reg_model = LogisticRegression(random_state=42, max_iter=1000)
log_reg_model.fit(X_train_scaled, y_train)

# ─── 9. Evaluasi ─────────────────────────────────────────────────────────────
y_pred = log_reg_model.predict(X_test_scaled)
print(f"\n  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"\n  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"\n  Classification Report:\n{classification_report(y_test, y_pred)}")

# ─── 10. Simpan Artefak ke File .pkl ─────────────────────────────────────────
print("\nMenyimpan artefak model...")
with open('log_reg_model.pkl', 'wb') as f:
    pickle.dump(log_reg_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

print("Selesai! File log_reg_model.pkl, scaler.pkl, dan encoders.pkl siap digunakan.")
print(f"\n  Fitur yang digunakan ({len(X.columns)} kolom):")
for col in X.columns:
    print(f"    - {col}")
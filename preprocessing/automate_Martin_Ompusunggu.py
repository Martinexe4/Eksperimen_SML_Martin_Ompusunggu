# Membuat file automate_Martin.py berdasarkan tahapan preprocessing di notebook eksperimen
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from joblib import dump, load

def automate_preprocessing(data_path, label_column, save_dir="model_automate", pca_group1_cols=None, pca_group2_cols=None,
                           pca_1_components=3, pca_2_components=3):
    # 1. Load data
    df = pd.read_csv(data_path)

    # 2. Handle duplicates
    df = df.drop_duplicates()

    # 3. Feature scaling
    scaler = MinMaxScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])
    dump(scaler, f"{save_dir}/scaler.joblib")

    # 4. Encode label if necessary
    if df[label_column].dtype != np.number and df[label_column].dtype != np.int64:
        le = LabelEncoder()
        df[label_column] = le.fit_transform(df[label_column])
        dump(le, f"{save_dir}/label_encoder.joblib")

    # 5. Split data
    X = df.drop(columns=[label_column])
    y = df[label_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

    # 6. Apply PCA Group 1
    if pca_group1_cols:
        pca_1 = PCA(n_components=pca_1_components, random_state=123)
        pca_1.fit(X_train[pca_group1_cols])
        dump(pca_1, f"{save_dir}/pca_1.joblib")
        X_train_pca1 = pca_1.transform(X_train[pca_group1_cols])
        X_test_pca1 = pca_1.transform(X_test[pca_group1_cols])
        pca1_df_train = pd.DataFrame(X_train_pca1, columns=[f"pc1_{i+1}" for i in range(pca_1_components)], index=X_train.index)
        pca1_df_test = pd.DataFrame(X_test_pca1, columns=[f"pc1_{i+1}" for i in range(pca_2_components)], index=X_test.index)
        X_train = X_train.drop(columns=pca_group1_cols).join(pca1_df_train)
        X_test = X_test.drop(columns=pca_group1_cols).join(pca1_df_test)

    # 7. Apply PCA Group 2
    if pca_group2_cols:
        pca_2 = PCA(n_components=pca_2_components, random_state=123)
        pca_2.fit(X_train[pca_group2_cols])
        dump(pca_2, f"{save_dir}/pca_2.joblib")
        X_train_pca2 = pca_2.transform(X_train[pca_group2_cols])
        X_test_pca2 = pca_2.transform(X_test[pca_group2_cols])
        pca2_df_train = pd.DataFrame(X_train_pca2, columns=[f"pc2_{i+1}" for i in range(pca_2_components)], index=X_train.index)
        pca2_df_test = pd.DataFrame(X_test_pca2, columns=[f"pc2_{i+1}" for i in range(pca_2_components)], index=X_test.index)
        X_train = X_train.drop(columns=pca_group2_cols).join(pca2_df_train)
        X_test = X_test.drop(columns=pca_group2_cols).join(pca2_df_test)

    return X_train, X_test, y_train, y_test

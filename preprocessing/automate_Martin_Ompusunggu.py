import pandas as pd
import numpy as np
import os
import sys
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OrdinalEncoder


def bin_bmi(bmi):
    if bmi < 18.5:
        return 1
    elif 18.5 <= bmi < 24.9:
        return 2
    elif 25 <= bmi < 29.9:
        return 4
    else:
        return 5

def automate_preprocessing_pipeline(data_path, save_csv_path, model_save_dir):
    # Buat folder output jika belum ada
    os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)

    # Load dataset
    df = pd.read_csv(data_path)

    # Buat kolom BMI_Category terlebih dahulu
    df["BMI_Category"] = df["BMI"].apply(bin_bmi)

    # Definisikan fitur
    target_col = "isDiabetes"

    numerical_features = ["BMI", "MentHlth", "PhysHlth"]
    ordinal_features = ["GenHlth", "Age", "Education", "Income", "BMI_Category"]
    binary_categorical_features = [
        "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "HeartDiseaseorAttack",
        "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
        "NoDocbcCost", "DiffWalk", "Sex"
    ]

    df_processed = df.copy()

    # Step 1: Manual Label Encoding (karena tidak kompatibel di pipeline)
    for feature in binary_categorical_features:
        encoder = LabelEncoder()
        df_processed[feature] = encoder.fit_transform(df_processed[feature])

    # Step 2: Pipeline untuk numerical + ordinal
    numerical_pipeline = Pipeline([
        ("scaler", MinMaxScaler())
    ])

    ordinal_pipeline = Pipeline([
        ("ordinal", OrdinalEncoder())
    ])

    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline, numerical_features),
        ("ord", ordinal_pipeline, ordinal_features)
    ])

    # Terapkan transformasi
    transformed = preprocessor.fit_transform(df_processed)

    # Gabungkan hasil dengan kolom target (ditempatkan di kolom terakhir)
    final_df = pd.DataFrame(transformed, columns=numerical_features + ordinal_features)
    final_df[target_col] = df[target_col].values

    # Simpan hasil akhir tanpa header
    final_df.to_csv(save_csv_path, index=False, header=False)
    print(f"✅ Preprocessing selesai. Dataset disimpan di: {save_csv_path}")


    if __name__ == "__main__":
        data_path = sys.argv[1]
        save_path = sys.argv[2]
        automate_preprocessing_pipeline(
            data_path=data_path,
            save_csv_path=save_path,
            model_save_dir="model_registry"
        )

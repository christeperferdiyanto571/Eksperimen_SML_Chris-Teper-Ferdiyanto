"""
automate_chris-teper-ferdiyanto.py
Automated preprocessing pipeline untuk California Housing Dataset.
Mengkonversi tahapan eksperimen pada notebook menjadi fungsi yang dapat dijalankan otomatis.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data() -> pd.DataFrame:
    """
    Load California Housing dataset dan simpan sebagai raw CSV.
    Returns:
        pd.DataFrame: Raw dataframe
    """
    logger.info("Loading California Housing dataset...")
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Simpan raw dataset
    raw_path = "housing_raw.csv"
    df.to_csv(raw_path, index=False)
    logger.info(f"Raw dataset saved to {raw_path}")

    return df


def perform_eda(df: pd.DataFrame) -> None:
    """
    Melakukan Exploratory Data Analysis dan mencetak hasilnya.
    Args:
        df: Input dataframe
    """
    logger.info("Performing EDA...")

    print("\n=== Dataset Shape ===")
    print(df.shape)

    print("\n=== Missing Values ===")
    print(df.isnull().sum())

    print("\n=== Duplicate Rows ===")
    print(f"Total duplicates: {df.duplicated().sum()}")

    print("\n=== Descriptive Statistics ===")
    print(df.describe())

    # Outlier detection
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
    print("\n=== Outlier Count per Column ===")
    print(outliers)

    logger.info("EDA completed")


def handle_missing_values(X: pd.DataFrame) -> pd.DataFrame:
    """
    Menangani missing values menggunakan mean imputation.
    Args:
        X: Feature dataframe
    Returns:
        pd.DataFrame: Dataframe tanpa missing values
    """
    logger.info("Handling missing values...")
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    logger.info(f"Missing values after imputation: {X_imputed.isnull().sum().sum()}")
    return X_imputed


def handle_outliers(X: pd.DataFrame) -> pd.DataFrame:
    """
    Menangani outlier menggunakan IQR clipping.
    Args:
        X: Feature dataframe
    Returns:
        pd.DataFrame: Dataframe dengan outlier yang sudah di-clip
    """
    logger.info("Handling outliers with IQR clipping...")
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    X_clipped = X.clip(lower=lower_bound, upper=upper_bound, axis=1)
    logger.info("Outlier handling completed")
    return X_clipped


def scale_features(X: pd.DataFrame) -> tuple:
    """
    Melakukan feature scaling menggunakan StandardScaler.
    Args:
        X: Feature dataframe
    Returns:
        tuple: (scaled_df, scaler_object)
    """
    logger.info("Scaling features with StandardScaler...")
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    logger.info("Feature scaling completed")
    return X_scaled, scaler


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> tuple:
    """
    Membagi data menjadi train dan test set.
    Args:
        X: Feature dataframe
        y: Target series
        test_size: Proporsi test set
        random_state: Random seed
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    logger.info(f"Splitting data with test_size={test_size}...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def save_preprocessed_data(X_train, X_test, y_train, y_test, output_dir: str = "housing_preprocessing") -> None:
    """
    Menyimpan data yang sudah diproses ke folder output.
    Args:
        X_train, X_test, y_train, y_test: Split datasets
        output_dir: Direktori output
    """
    os.makedirs(output_dir, exist_ok=True)

    train_df = X_train.copy()
    train_df['MedHouseVal'] = y_train.values

    test_df = X_test.copy()
    test_df['MedHouseVal'] = y_test.values

    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Train data saved to {train_path}")
    logger.info(f"Test data saved to {test_path}")


def run_preprocessing_pipeline() -> tuple:
    """
    Menjalankan seluruh pipeline preprocessing dari awal hingga akhir.
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    logger.info("Starting preprocessing pipeline...")

    # 1. Load data
    df = load_data()

    # 2. EDA
    perform_eda(df)

    # 3. Pisahkan fitur dan target
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']

    # 4. Handle missing values
    X = handle_missing_values(X)

    # 5. Handle outliers
    X = handle_outliers(X)

    # 6. Feature scaling
    X_scaled, scaler = scale_features(X)

    # 7. Train-test split
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)

    # 8. Simpan hasil
    save_preprocessed_data(X_train, X_test, y_train, y_test)

    logger.info("Preprocessing pipeline completed successfully!")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = run_preprocessing_pipeline()
    print("\n=== Preprocessing Summary ===")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape:  {y_test.shape}")

"""
Data loading and preprocessing utilities
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
import os

try:
    from .config import TRAIN_DATA_PATH, DESCRIPTION_DATA_PATH
except ImportError:
    # Fallback for direct execution
    DATA_DIR = "data"
    TRAIN_DATA_PATH = f"{DATA_DIR}/disease_dataAI2.csv"
    DESCRIPTION_DATA_PATH = f"{DATA_DIR}/disease_descriptionsAI.csv"


def load_training_data() -> pd.DataFrame:
    """Load training data from CSV file"""
    if not os.path.exists(TRAIN_DATA_PATH):
        raise FileNotFoundError(f"Training data not found at {TRAIN_DATA_PATH}")

    df = pd.read_csv(TRAIN_DATA_PATH)
    df = df.fillna('')  # Fill NaN with empty string
    return df


def load_disease_descriptions() -> Dict[str, str]:
    """Load disease descriptions from CSV file"""
    if not os.path.exists(DESCRIPTION_DATA_PATH):
        print(f"Warning: Description data not found at {DESCRIPTION_DATA_PATH}")
        return {}

    desc_df = pd.read_csv(DESCRIPTION_DATA_PATH)
    disease_dict = dict(zip(
        desc_df['Disease'].str.strip(),
        desc_df['Description'].str.strip()
    ))
    return disease_dict


def preprocess_symptoms(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Preprocess symptoms data into feature vectors

    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Labels array
        symptoms_list: Sorted list of unique symptoms
    """
    # Get symptom columns
    symptom_columns = [col for col in df.columns if 'Symptom' in col]

    # Collect all unique symptoms
    all_symptoms = set()
    for col in symptom_columns:
        unique_symps = df[col].unique()
        for symp in unique_symps:
            symp = str(symp).strip().lower()
            if symp and symp != '':
                all_symptoms.add(symp)

    symptoms_list = sorted(list(all_symptoms))
    print(f"Found {len(symptoms_list)} unique symptoms")

    # Create one-hot vectors
    X_data = []
    y_data = df['Disease'].values

    for index, row in df.iterrows():
        row_symptoms = [
            str(row[col]).strip().lower()
            for col in symptom_columns
            if str(row[col]).strip() != ''
        ]

        # Create feature vector
        feature_vector = [0] * len(symptoms_list)
        for symp in row_symptoms:
            if symp in symptoms_list:
                symp_idx = symptoms_list.index(symp)
                feature_vector[symp_idx] = 1

        X_data.append(feature_vector)

    X = np.array(X_data)
    y = np.array(y_data)  # Ensure y is numpy array

    return X, y, symptoms_list
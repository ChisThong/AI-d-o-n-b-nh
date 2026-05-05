"""
Configuration file for Disease Prediction AI Project
"""

# Data paths
DATA_DIR = "data"
MODEL_DIR = "models"

TRAIN_DATA_PATH = f"{DATA_DIR}/disease_dataAI2.csv"
DESCRIPTION_DATA_PATH = f"{DATA_DIR}/disease_descriptionsAI.csv"

# Model paths
MODEL_PATH = f"{MODEL_DIR}/disease_rf_model.pkl"
SYMPTOMS_LIST_PATH = f"{MODEL_DIR}/symptoms_list.pkl"
DISEASE_DESCRIPTIONS_PATH = f"{MODEL_DIR}/disease_descriptions.pkl"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.25  # 25% of remaining data after test split

# Random Forest parameters
N_ESTIMATORS = 100
"""
Machine Learning model utilities
"""

import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from typing import Tuple, List, Dict

try:
    from .config import (
        MODEL_PATH, SYMPTOMS_LIST_PATH, DISEASE_DESCRIPTIONS_PATH,
        RANDOM_STATE, TEST_SIZE, VAL_SIZE, N_ESTIMATORS
    )
except ImportError:
    # Fallback for direct execution
    MODEL_PATH = "models/disease_rf_model.pkl"
    SYMPTOMS_LIST_PATH = "models/symptoms_list.pkl"
    DISEASE_DESCRIPTIONS_PATH = "models/disease_descriptions.pkl"
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    VAL_SIZE = 0.25
    N_ESTIMATORS = 100


class DiseasePredictor:
    """Disease prediction model wrapper"""

    def __init__(self):
        self.model = None
        self.symptoms_list = None
        self.disease_descriptions = None

    def train(self, X: np.ndarray, y: np.ndarray, symptoms_list: List[str],
              disease_descriptions: Dict[str, str]) -> Dict[str, float]:
        """
        Train the model and return evaluation metrics

        Returns:
            metrics: Dictionary with train/val/test accuracies
        """
        print("Splitting data into train/validation/test sets...")

        # Split data: 60% train, 20% val, 20% test
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=VAL_SIZE, random_state=RANDOM_STATE
        )

        print(f"Data sizes: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")

        # Train model
        print("Training RandomForest model...")
        self.model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE
        )
        self.model.fit(X_train, y_train)

        # Evaluate
        metrics = {}

        y_train_pred = self.model.predict(X_train)
        metrics['train_accuracy'] = accuracy_score(y_train, y_train_pred)

        y_val_pred = self.model.predict(X_val)
        metrics['val_accuracy'] = accuracy_score(y_val, y_val_pred)

        y_test_pred = self.model.predict(X_test)
        metrics['test_accuracy'] = accuracy_score(y_test, y_test_pred)

        print(".2f")
        print(".2f")
        print(".2f")

        # Store metadata
        self.symptoms_list = symptoms_list
        self.disease_descriptions = disease_descriptions

        return metrics

    def predict(self, symptoms: List[str]) -> Tuple[str, str]:
        """
        Predict disease from symptoms list

        Returns:
            prediction: Predicted disease name
            description: Disease description if available
        """
        if self.model is None or self.symptoms_list is None:
            raise ValueError("Model not loaded. Call load() first.")

        # Create feature vector
        feature_vector = [0] * len(self.symptoms_list)
        normalized_symptoms = [sym.lower() for sym in self.symptoms_list]

        for symp in symptoms:
            symp = symp.strip().lower()
            if symp in normalized_symptoms:
                idx = normalized_symptoms.index(symp)
                feature_vector[idx] = 1

        # Predict
        X_input = np.array([feature_vector])
        prediction = self.model.predict(X_input)[0]

        # Get description
        description = self.disease_descriptions.get(prediction, "")

        return prediction, description

    def predict_top_k(self, symptoms: List[str], top_k: int = 3) -> List[Dict[str, object]]:
        """
        Predict the top K most likely diseases for the given symptoms.

        Returns:
            List of dictionaries with keys: disease, probability, description
        """
        if self.model is None or self.symptoms_list is None:
            raise ValueError("Model not loaded. Call load() first.")

        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("The loaded model does not support predict_proba().")

        feature_vector = [0] * len(self.symptoms_list)
        normalized_symptoms = [sym.lower() for sym in self.symptoms_list]

        for symp in symptoms:
            symp = symp.strip().lower()
            if symp in normalized_symptoms:
                idx = normalized_symptoms.index(symp)
                feature_vector[idx] = 1

        X_input = np.array([feature_vector])
        probabilities = self.model.predict_proba(X_input)[0]
        class_labels = self.model.classes_

        top_indices = np.argsort(probabilities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            disease = class_labels[idx]
            prob = float(probabilities[idx])
            description = self.disease_descriptions.get(disease, "")
            results.append({
                'disease': disease,
                'probability': prob,
                'description': description
            })

        return results

    def save(self):
        """Save model and metadata to disk"""
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.symptoms_list, SYMPTOMS_LIST_PATH)
        joblib.dump(self.disease_descriptions, DISEASE_DESCRIPTIONS_PATH)

        print(f"Model saved to {MODEL_PATH}")

    def load(self) -> bool:
        """Load model and metadata from disk"""
        try:
            self.model = joblib.load(MODEL_PATH)
            self.symptoms_list = joblib.load(SYMPTOMS_LIST_PATH)
            self.disease_descriptions = joblib.load(DISEASE_DESCRIPTIONS_PATH)
            return True
        except FileNotFoundError:
            return False
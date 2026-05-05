#!/usr/bin/env python3
"""
Train the disease prediction model
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import load_training_data, load_disease_descriptions, preprocess_symptoms
from src.model import DiseasePredictor


def main():
    print("=== Disease Prediction AI - Training ===")

    try:
        # Load data
        print("Loading training data...")
        df = load_training_data()
        disease_descriptions = load_disease_descriptions()

        # Preprocess
        print("Preprocessing data...")
        X, y, symptoms_list = preprocess_symptoms(df)

        # Train model
        predictor = DiseasePredictor()
        metrics = predictor.train(X, y, symptoms_list, disease_descriptions)

        # Save model
        print("Saving model...")
        predictor.save()

        print("\n[SUCCESS] Training completed successfully!")
        print(".2f")
        print(".2f")
        print(".2f")

    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
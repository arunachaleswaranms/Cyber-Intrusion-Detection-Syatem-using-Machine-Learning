# model_evaluation.py

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from data_preprocessing import load_data, preprocess_data

def load_model(model_path):
    """
    Load a trained model from a file.
    """
    return joblib.load(model_path)

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model using test data.
    """
    y_pred = model.predict(X_test)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nAccuracy Score:")
    print(accuracy_score(y_test, y_pred))

if __name__ == '__main__':
    train_path = 'KDDTrain+.csv'
    test_path = 'KDDTest+.csv'

    # Load and preprocess data
    train_df, test_df = load_data(train_path, test_path)
    X_train, X_test, y_train, y_test = preprocess_data(train_df, test_df)

    # Load trained models
    rf_model_path = 'random_forest_model.joblib'
    gb_model_path = 'gradient_boosting_model.joblib'

    rf_model = load_model(rf_model_path)
    gb_model = load_model(gb_model_path)

    # Evaluate Random Forest model
    print("Evaluating Random Forest model...")
    evaluate_model(rf_model, X_test, y_test)

    # Evaluate Gradient Boosting model
    print("\nEvaluating Gradient Boosting model...")
    evaluate_model(gb_model, X_test, y_test)

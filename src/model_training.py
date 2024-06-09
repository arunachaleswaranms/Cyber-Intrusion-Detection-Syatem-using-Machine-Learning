# model_training.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from data_preprocessing import load_data, preprocess_data

def train_random_forest(X_train, y_train):
    """
    Train a Random Forest classifier.
    """
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

def train_gradient_boosting(X_train, y_train):
    """
    Train a Gradient Boosting classifier.
    """
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    return gb_model

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

    # Train Random Forest model
    print("Training Random Forest model...")
    rf_model = train_random_forest(X_train, y_train)
    print("Random Forest model trained.")

    # Evaluate Random Forest model
    print("\nEvaluating Random Forest model...")
    evaluate_model(rf_model, X_test, y_test)

    # Train Gradient Boosting model
    print("Training Gradient Boosting model...")
    gb_model = train_gradient_boosting(X_train, y_train)
    print("Gradient Boosting model trained.")

    # Evaluate Gradient Boosting model
    print("\nEvaluating Gradient Boosting model...")
    evaluate_model(gb_model, X_test, y_test)

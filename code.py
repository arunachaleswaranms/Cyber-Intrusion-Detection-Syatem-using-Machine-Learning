# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from joblib import Parallel, delayed

# Data Preprocessing
def load_and_preprocess_data(train_path, test_path):
    # Load datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Combine train and test data for consistent preprocessing
    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    # Identify categorical and numerical features
    categorical_cols = combined_df.select_dtypes(include=['object']).columns
    numerical_cols = combined_df.select_dtypes(exclude=['object']).columns.drop('label')

    # Preprocess features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

    # Fit the preprocessor on the combined data
    X_combined = combined_df.drop('label', axis=1)
    y_combined = combined_df['label']
    X_combined = preprocessor.fit_transform(X_combined)

    # Separate combined data back into train and test sets
    X_train = X_combined[:len(train_df)]
    X_test = X_combined[len(train_df):]
    y_train = y_combined[:len(train_df)]
    y_test = y_combined[len(train_df):]

    return X_train, X_test, y_train, y_test

# Model Training
def train_model(model, X_train, y_train):
    # Train the model
    model.fit(X_train, y_train)
    return model

# Model Evaluation
def evaluate_model(model, X_test, y_test):
    # Predict on test set
    y_pred = model.predict(X_test)

    # Evaluate predictions
    results = {
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
        "Classification Report": classification_report(y_test, y_pred),
        "Accuracy Score": accuracy_score(y_test, y_pred)
    }
    return results

# Main function
def main():
    # File paths
    train_path = 'KDDTrain+.csv'
    test_path = 'KDDTest+.csv'

    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(train_path, test_path)

    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    # Train and evaluate models in parallel
    trained_models = Parallel(n_jobs=-1)(delayed(train_model)(model, X_train, y_train) for model in models.values())
    model_names = list(models.keys())
    evaluation_results = Parallel(n_jobs=-1)(delayed(evaluate_model)(model, X_test, y_test) for model in trained_models)

    # Display evaluation results
    for model_name, result in zip(model_names, evaluation_results):
        print(f"Results for {model_name}:")
        print("Confusion Matrix:")
        print(result["Confusion Matrix"])
        print("Classification Report:")
        print(result["Classification Report"])
        print("Accuracy Score:")
        print(result["Accuracy Score"])
        print("\n")

if __name__ == '__main__':
    main()

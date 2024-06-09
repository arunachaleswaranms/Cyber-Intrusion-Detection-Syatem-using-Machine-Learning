# data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(train_path, test_path):
    """
    Load train and test datasets from CSV files.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def preprocess_data(train_df, test_df):
    """
    Preprocess the train and test datasets by encoding categorical features and normalizing the data.
    """
    # Combine train and test data for consistent preprocessing
    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    # Encode categorical features
    categorical_cols = combined_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        combined_df[col] = le.fit_transform(combined_df[col])

    # Separate combined data back into train and test sets
    train_df = combined_df.iloc[:len(train_df)]
    test_df = combined_df.iloc[len(train_df):]

    # Split features and labels
    X_train = train_df.drop('label', axis=1)
    y_train = train_df['label']
    X_test = test_df.drop('label', axis=1)
    y_test = test_df['label']

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    train_path = 'KDDTrain+.csv'
    test_path = 'KDDTest+.csv'

    # Load data
    train_df, test_df = load_data(train_path, test_path)

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(train_df, test_df)

    print("Data preprocessing completed.")

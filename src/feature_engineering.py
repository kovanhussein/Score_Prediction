# src/feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_data(file_path):
    return pd.read_csv(file_path)

def engineer_features(df):
    # Convert date column to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    # Example feature engineering
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    
    # Dropping the original date column
    df = df.drop(columns=['date'])

    return df

def preprocess_data(df):
    # Separate features and target
    X = df.drop(columns=['sales'])
    y = df['sales']
    
    # Identify categorical and numerical columns
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(exclude=['object']).columns
    
    # Define the preprocessor
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Preprocess the features
    X_preprocessed = preprocessor.fit_transform(X)
    
    # Convert to dense array if it's sparse
    if hasattr(X_preprocessed, 'toarray'):
        X_preprocessed = X_preprocessed.toarray()
    
    return X_preprocessed, y

def save_preprocessed_data(X, y, feature_path, target_path):
    # Save the preprocessed features and target
    pd.DataFrame(X).to_csv(feature_path, index=False)
    y.to_csv(target_path, index=False)

def main():
    # Load cleaned data
    clean_data_path = './data/processed/cleaned_data.csv'
    df = load_data(clean_data_path)
    
    # Engineer features
    df_features = engineer_features(df)
    
    # Preprocess data
    X_preprocessed, y = preprocess_data(df_features)
    
    # Save preprocessed data
    feature_path = './data/processed/features.csv'
    target_path = './data/processed/target.csv'
    save_preprocessed_data(X_preprocessed, y, feature_path, target_path)

if __name__ == "__main__":
    main()

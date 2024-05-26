# src/feature_engineering.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    return pd.read_csv(file_path)

def engineer_features(df):
    df['date'] = pd.to_datetime(df['date'])
    # Example feature engineering
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    
    # Dropping the original date column
    df = df.drop(columns=['date'])

    return df

def normalize_features(df):
    scaler = StandardScaler()
    numerical_features = ['sales']
    
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    return df

def save_engineered_data(df, file_path):
    df.to_csv(file_path, index=False)

def main():
    # Load cleaned data
    clean_data_path = './data/processed/cleaned_data.csv'
    df = load_data(clean_data_path)
    
    # Engineer features
    df_features = engineer_features(df)
    
    # Normalize features
    df_normalized = normalize_features(df_features)
    
    # Save engineered data
    engineered_data_path = './data/processed/engineered_data.csv'
    save_engineered_data(df_normalized, engineered_data_path)

if __name__ == "__main__":
    main()

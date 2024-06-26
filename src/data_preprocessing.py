# src/data_preprocessing.py

import pandas as pd
import numpy as np

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])  # Ensure date column is in datetime format
    return df

def clean_data(df):
    # Handle missing values
    df = df.dropna()

    # Correct data types
    # Already handled date type conversion in load_data
    return df

def save_clean_data(df, file_path):
    df.to_csv(file_path, index=False)

def main():
    # Load raw data
    raw_data_path = './data/raw/train.csv'
    df = load_data(raw_data_path)
    
    # Clean data
    df_clean = clean_data(df)
    
    # Save cleaned data
    clean_data_path = './data/processed/cleaned_data.csv'
    save_clean_data(df_clean, clean_data_path)

if __name__ == "__main__":
    main()

# src/model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def load_data(feature_path, target_path):
    X = pd.read_csv(feature_path)
    y = pd.read_csv(target_path).squeeze("columns")  # Ensure y is loaded as a Series
    return X, y

def train_models(X_train, y_train):
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(),
        'Random Forest': RandomForestRegressor()
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        models[name] = model
    
    return models

def evaluate_models(models, X_test, y_test):
    results = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {'MSE': mse, 'R2': r2}
    
    return results

def main():
    # Load preprocessed data
    feature_path = './data/processed/features.csv'
    target_path = './data/processed/target.csv'
    X, y = load_data(feature_path, target_path)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    models = train_models(X_train, y_train)
    
    # Evaluate models
    results = evaluate_models(models, X_test, y_test)
    
    # Print results
    for name, metrics in results.items():
        print(f'{name}: MSE = {metrics["MSE"]}, R2 = {metrics["R2"]}')

if __name__ == "__main__":
    main()

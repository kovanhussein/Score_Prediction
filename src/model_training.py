# src/model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np

def load_data(feature_path, target_path):
    X = pd.read_csv(feature_path)
    y = pd.read_csv(target_path).squeeze("columns")  # Ensure y is loaded as a Series
    return X, y

def cross_validate_model(model, X, y, cv=5):
    mse_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    
    mse_mean = -mse_scores.mean()  # Convert negative MSE to positive
    r2_mean = r2_scores.mean()
    
    return {'MSE': mse_mean, 'R2': r2_mean}

def hyperparameter_tuning(X_train, y_train):
    param_grid_rf = {
        'n_estimators': [100, 200],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    rf = RandomForestRegressor()
    
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid_rf, 
                               cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(),
        'Random Forest': hyperparameter_tuning(X_train, y_train)  # Use tuned Random Forest
    }
    
    results = {}
    best_model = None
    best_score = float('inf')  # Initialize with a very large value
    
    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        
        # Cross-validate the model
        cv_results = cross_validate_model(model, X_train, y_train)
        
        # Evaluate on test set
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'Cross-Validated MSE': cv_results['MSE'],
            'Cross-Validated R2': cv_results['R2'],
            'Test MSE': mse,
            'Test R2': r2
        }
        
        # Select the best model based on Cross-Validated MSE
        if cv_results['MSE'] < best_score:
            best_score = cv_results['MSE']
            best_model = model
    
    return results, best_model

def main():
    # Load preprocessed data
    feature_path = './data/processed/features.csv'
    target_path = './data/processed/target.csv'
    X, y = load_data(feature_path, target_path)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and evaluate models
    results, best_model = train_and_evaluate_models(X_train, y_train, X_test, y_test)
    
    # Print results
    for name, metrics in results.items():
        print(f'{name}:')
        print(f'  Cross-Validated MSE = {metrics["Cross-Validated MSE"]}')
        print(f'  Cross-Validated R2 = {metrics["Cross-Validated R2"]}')
        print(f'  Test MSE = {metrics["Test MSE"]}')
        print(f'  Test R2 = {metrics["Test R2"]}')
    
    # Save the best model
    model_path = './models/best_model.joblib'
    joblib.dump(best_model, model_path)
    print(f'Best model saved to {model_path}')

if __name__ == "__main__":
    main()

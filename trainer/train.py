import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score,KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error
import xgboost as xgb
import os
import sys
import subprocess
import joblib

TRAINING_DATASET = "gs://specialization_black_friday/preprocessed_train_df.csv"
TRAINING_DATA_FILE = "preprocessed_train_df.csv"
MODEL_FILE = "model.bst"

model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, enable_categorical=True, tree_method='hist')

best_params = {'colsample_bytree': 0.9, 'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 200, 'subsample': 1}

if __name__ == '__main__':
    subprocess.check_call(['gsutil', 'cp', TRAINING_DATASET, TRAINING_DATA_FILE], stderr=sys.stdout)
    train_df = pd.read_csv(TRAINING_DATA_FILE)
    X = train_df.drop(columns=['User_ID', 'Product_ID','Purchase'])
    y = train_df['Purchase'].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    best_params = {'colsample_bytree': 0.9, 'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 300, 'subsample': 1}
    best_model = xgb.XGBRegressor(**best_params, objective='reg:squarederror', random_state=42)
    best_model.fit(X_train, y_train)
    
    best_model.save_model(MODEL_FILE)
    gcs_model_path = "gs://specialization-black-friday-demo-model/model.bst"
    subprocess.check_call(['gsutil', 'cp', MODEL_FILE, gcs_model_path],stderr=sys.stdout)
    
    
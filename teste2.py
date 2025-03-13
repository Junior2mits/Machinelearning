# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 21:51:09 2025

@author: utilisateur
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle  # Pour sauvegarder le modèle

# Modules pour la modélisation
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

#####################################
# 1. Lecture et exploration initiale #
#####################################

file_path = r'C:\Users\utilisateur\Downloads\Projetc\Non Life\dne.csv'
df = pd.read_csv(file_path)

print("=== Aperçu du dataset ===")
print(df.head())

##########################################
# 2. Nettoyage et prétraitement des données #
##########################################

cols_to_drop = ['_c39', 'policy_number', 'insured_zip', 'incident_location']
df.drop(columns=cols_to_drop, inplace=True)

df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'], errors='coerce')
df['incident_date'] = pd.to_datetime(df['incident_date'], errors='coerce')

df['authorities_contacted'].fillna(df['authorities_contacted'].mode()[0], inplace=True)

##########################################
# 3. Encodage des variables
##########################################

binary_cols = ['fraud_reported', 'insured_sex', 'property_damage', 'police_report_available']
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

df.rename(columns={'fraud_reported': 'target'}, inplace=True)

one_hot_cols = ['policy_state', 'policy_csl', 'insured_education_level', 
                'insured_occupation', 'insured_hobbies', 'insured_relationship',
                'incident_type', 'collision_type', 'incident_severity', 
                'authorities_contacted', 'incident_state', 'incident_city', 
                'auto_make', 'auto_model']

df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True)

##########################################
# 4. Séparation train/test et transformation des dates
##########################################

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for date_col in ['policy_bind_date', 'incident_date']:
    min_date = X_train[date_col].min()
    X_train.loc[:, date_col] = (X_train[date_col] - min_date).dt.days
    X_test.loc[:, date_col] = (X_test[date_col] - min_date).dt.days

##########################################
# 5. Application de SMOTE sur l'entraînement
##########################################

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

##########################################
# 6. Modélisation
##########################################

# --- 6.1 RandomForestClassifier ---
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)

# --- 6.2 XGBoost ---
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_resampled, y_train_resampled)

# 📌 Sauvegarde du modèle XGBoost
model_filename = "xgboost_fraud_model.pkl"
with open(model_filename, "wb") as file:
    pickle.dump(xgb_model, file)

print(f"✅ Modèle XGBoost sauvegardé sous {model_filename}")

# --- 6.3 Optimisation d'XGBoost avec GridSearchCV ---
param_grid = {
    'n_estimators': [100, 200, 300],  
    'max_depth': [3, 5, 7],  
    'learning_rate': [0.01, 0.1, 0.2],  
    'subsample': [0.7, 0.8, 1],  
    'colsample_bytree': [0.7, 0.8, 1]  
}

xgb_grid = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
grid_search = GridSearchCV(estimator=xgb_grid, param_grid=param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=2)
grid_search.fit(X_train_resampled, y_train_resampled)

print("\n=== Meilleurs hyperparamètres pour XGBoost ===")
print(grid_search.best_params_)

best_xgb = grid_search.best_estimator_

# 📌 Sauvegarde du meilleur modèle XGBoost optimisé
optimized_model_filename = "xgboost_fraud_model_optimized.pkl"
with open(optimized_model_filename, "wb") as file:
    pickle.dump(best_xgb, file)

print(f"✅ Modèle XGBoost optimisé sauvegardé sous {optimized_model_filename}")


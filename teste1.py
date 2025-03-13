# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 21:51:09 2025

@author: utilisateur
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# Chemin du fichier CSV (adapter si nécessaire)
file_path = r'C:\Users\utilisateur\Downloads\Projetc\Non Life\dne.csv'

# Lecture du fichier CSV
df = pd.read_csv(file_path)

print("=== Aperçu du dataset ===")
print(df.head())
print("\n=== Informations sur le dataset ===")
print(df.info())
print("\n=== Distribution de 'fraud_reported' ===")
print(df['fraud_reported'].value_counts())
print("\n=== Liste des colonnes ===")
print(df.columns)

##########################################
# 2. Nettoyage et prétraitement des données #
##########################################

# Suppression des colonnes inutiles
cols_to_drop = ['_c39', 'policy_number', 'insured_zip', 'incident_location']
df.drop(columns=cols_to_drop, inplace=True)

# Conversion des colonnes de dates en format datetime
df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'], errors='coerce')
df['incident_date'] = pd.to_datetime(df['incident_date'], errors='coerce')

# Gestion des valeurs manquantes dans 'authorities_contacted' (remplacer par la modalité la plus fréquente)
df['authorities_contacted'].fillna(df['authorities_contacted'].mode()[0], inplace=True)

##########################################
# 3. Encodage des variables
##########################################

# Encodage des colonnes binaires incluant la variable cible
binary_cols = ['fraud_reported', 'insured_sex', 'property_damage', 'police_report_available']
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

# Renommer la variable cible pour plus de clarté
df.rename(columns={'fraud_reported': 'target'}, inplace=True)

# One-hot encoding pour les colonnes catégorielles
one_hot_cols = ['policy_state', 'policy_csl', 'insured_education_level', 
                'insured_occupation', 'insured_hobbies', 'insured_relationship',
                'incident_type', 'collision_type', 'incident_severity', 
                'authorities_contacted', 'incident_state', 'incident_city', 
                'auto_make', 'auto_model']

df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True)

print("\n=== Informations après encodage ===")
print(df.info())

##########################################
# 4. Séparation train/test et transformation des dates
##########################################

# Séparation des features (X) et de la cible (y)
X = df.drop('target', axis=1)
y = df['target']

# Division en ensembles d'entraînement et de test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pour éviter le data leakage, on transforme les dates en nombre de jours en se basant sur l'ensemble d'entraînement
for date_col in ['policy_bind_date', 'incident_date']:
    # Calculer la date minimale sur l'ensemble d'entraînement
    min_date = X_train[date_col].min()
    # Transformer la colonne dans X_train et X_test en jours écoulés depuis min_date
    X_train.loc[:, date_col] = (X_train[date_col] - min_date).dt.days
    X_test.loc[:, date_col] = (X_test[date_col] - min_date).dt.days

##########################################
# 5. Application de SMOTE sur l'entraînement
##########################################

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print("\n=== Distribution des classes après SMOTE sur l'entraînement ===")
print(np.bincount(y_train_resampled))

##########################################
# 6. Modélisation
##########################################

# --- 6.1 RandomForestClassifier ---
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)
y_pred_rf = rf_model.predict(X_test)

print("\n=== Random Forest - Rapport de Classification ===")
print(classification_report(y_test, y_pred_rf))
print("=== Matrice de Confusion ===")
print(confusion_matrix(y_test, y_pred_rf))
print("=== AUC-ROC ===")
print(roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1]))

# --- 6.2 XGBoost ---
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_resampled, y_train_resampled)
y_pred_xgb = xgb_model.predict(X_test)

print("\n=== XGBoost - Rapport de Classification ===")
print(classification_report(y_test, y_pred_xgb))
print("=== Matrice de Confusion ===")
print(confusion_matrix(y_test, y_pred_xgb))
print("=== AUC-ROC ===")
print(roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1]))

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
y_pred_best_xgb = best_xgb.predict(X_test)
y_proba_best_xgb = best_xgb.predict_proba(X_test)[:, 1]

print("\n=== XGBoost Optimisé - Rapport de Classification ===")
print(classification_report(y_test, y_pred_best_xgb))
print("=== Matrice de Confusion ===")
print(confusion_matrix(y_test, y_pred_best_xgb))
print("=== AUC-ROC ===")
print(roc_auc_score(y_test, y_proba_best_xgb))

##########################################
# 7. Visualisations
##########################################

# 7.1 Importance des features avec Random Forest
importances = rf_model.feature_importances_
feature_names = X.columns
feat_importance = pd.DataFrame({"Feature": feature_names, "Importance": importances})
feat_importance = feat_importance.sort_values(by="Importance", ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feat_importance, palette="viridis")
plt.title("Top 10 des features les plus importantes")
plt.show()

# 7.2 Visualisations supplémentaires
# Pour visualiser certaines distributions originales, on relit le fichier CSV d'origine
df_viz = pd.read_csv(file_path)

# On complète les valeurs manquantes pour les colonnes de réclamations par la médiane
claim_features = ['total_claim_amount', 'vehicle_claim', 'injury_claim', 'property_claim']
for feature in claim_features:
    df_viz[feature].fillna(df_viz[feature].median(), inplace=True)

# Distribution des réclamations selon la variable cible (ici, 'fraud_reported' sous forme 'Y'/'N')
for feature in claim_features:
    plt.figure(figsize=(8, 6))
    sns.kdeplot(df_viz[df_viz['fraud_reported'] == 'N'][feature].dropna(), label='Non Fraudulent', fill=True)
    sns.kdeplot(df_viz[df_viz['fraud_reported'] == 'Y'][feature].dropna(), label='Fraudulent', fill=True)
    plt.title(f'Distribution de {feature}')
    plt.legend()
    plt.show()

# 7.3 Visualisations avec features importantes dans le DataFrame transformé
# On sélectionne ici quelques variables d'intérêt (adapter les noms selon votre jeu de données)
important_features = [
    'vehicle_claim', 'total_claim_amount', 'injury_claim', 'property_claim',
    'incident_date', 'policy_bind_date', 'policy_annual_premium'
]

# Vérifier que ces features existent dans df
available_features = [feat for feat in important_features if feat in df.columns]
print("Features disponibles pour visualisation :", available_features)

# Distribution (KDE plots) selon la classe cible (target codée en 0 et 1)
plt.figure(figsize=(16, 12))
for i, feature in enumerate(available_features, 1):
    plt.subplot(3, 2, i)
    sns.kdeplot(df[df['target'] == 0][feature], label='Non Fraudulent', shade=True)
    sns.kdeplot(df[df['target'] == 1][feature], label='Fraudulent', shade=True)
    plt.title(f"Distribution de {feature} par classe")
    plt.legend()
plt.tight_layout()
plt.show()

# Boxplots des features importantes selon la classe
plt.figure(figsize=(16, 12))
for i, feature in enumerate(available_features, 1):
    plt.subplot(3, 2, i)
    sns.boxplot(x='target', y=feature, data=df)
    plt.title(f"Boxplot de {feature} par classe")
plt.tight_layout()
plt.show()

# Heatmap des corrélations entre les features importantes
plt.figure(figsize=(10, 8))
corr = df[available_features].corr()
sns.heatmap(corr, cmap='coolwarm', annot=True, fmt=".2f", linewidths=0.5)
plt.title("Heatmap des corrélations entre les features importantes")
plt.show()

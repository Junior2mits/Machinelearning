# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 12:00:17 2025

@author: utilisateur
"""

import pandas as pd

# Lire le fichier CSV
#file_path = 'dne.csv'
file_path = r'C:\Users\utilisateur\Downloads\Projetc\Non Life\dne.csv'

df = pd.read_csv(file_path)

# Afficher les premières lignes
print(df.head())

#%%
print(df.info())

#%%
print(df['fraud_reported'].value_counts())

#%%
print(df.columns)

#%%

import pandas as pd

# Suppression des colonnes inutiles
df.drop(columns=['_c39', 'policy_number', 'insured_zip', 'incident_location'], inplace=True)

# Conversion des dates en format datetime
df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'], errors='coerce')
df['incident_date'] = pd.to_datetime(df['incident_date'], errors='coerce')

# Gérer les valeurs manquantes dans 'authorities_contacted' (remplacer par la valeur la plus fréquente)
df['authorities_contacted'].fillna(df['authorities_contacted'].mode()[0], inplace=True)

# Vérification après nettoyage
print(df.info())

#%%
from sklearn.preprocessing import LabelEncoder

# 1. Label encoding pour les colonnes binaires
label_cols = ['fraud_reported', 'insured_sex', 'property_damage', 'police_report_available']
le = LabelEncoder()

for col in label_cols:
    df[col] = le.fit_transform(df[col])

# 2. One-hot encoding pour les colonnes catégorielles
one_hot_cols = ['policy_state', 'policy_csl', 'insured_education_level', 
                'insured_occupation', 'insured_hobbies', 'insured_relationship',
                'incident_type', 'collision_type', 'incident_severity', 
                'authorities_contacted', 'incident_state', 'incident_city', 
                'auto_make', 'auto_model']

df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True)

# ✅ Vérification après encodage
print(df.info())

#%%
from imblearn.over_sampling import SMOTE

# 1. Séparation des features et de la cible
X = df.drop('fraud_reported', axis=1)
y = df['fraud_reported']

# 2. Conversion des dates en nombre de jours depuis la date minimale
X['policy_bind_date'] = (X['policy_bind_date'] - X['policy_bind_date'].min()).dt.days
X['incident_date'] = (X['incident_date'] - X['incident_date'].min()).dt.days

# 3. Application de SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 4. Vérification après SMOTE
import numpy as np
print(f"Distribution des classes après SMOTE : {np.bincount(y_resampled)}")

#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# 1. Division des données en train et test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 2. Entraînement du modèle Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 3. Prédiction sur le jeu de test
y_pred = model.predict(X_test)

# 4. Évaluation du modèle
print("✅ Classification Report :")
print(classification_report(y_test, y_pred))

print("\n✅ Matrice de confusion :")
print(confusion_matrix(y_test, y_pred))

print("\n✅ AUC-ROC :")
print(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

#%%
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# 1. Entraînement du modèle XGBoost
model_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model_xgb.fit(X_train, y_train)

# 2. Prédiction sur le jeu de test
y_pred_xgb = model_xgb.predict(X_test)

# 3. Évaluation du modèle
print("✅ Classification Report :")
print(classification_report(y_test, y_pred_xgb))

print("\n✅ Matrice de confusion :")
print(confusion_matrix(y_test, y_pred_xgb))

print("\n✅ AUC-ROC :")
print(roc_auc_score(y_test, model_xgb.predict_proba(X_test)[:, 1]))

#%%
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

# Définir la grille des hyperparamètres à tester
param_grid = {
    'n_estimators': [100, 200, 300],  
    'max_depth': [3, 5, 7],  
    'learning_rate': [0.01, 0.1, 0.2],  
    'subsample': [0.7, 0.8, 1],  
    'colsample_bytree': [0.7, 0.8, 1]  
}

# Initialiser le modèle
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Lancer la recherche par grille
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Meilleurs hyperparamètres
print("Meilleurs hyperparamètres : ", grid_search.best_params_)

# Évaluer le modèle optimisé
best_xgb = grid_search.best_estimator_
y_pred = best_xgb.predict(X_test)
y_proba = best_xgb.predict_proba(X_test)[:, 1]

# Rapport de classification
print("\n✅ Classification Report :\n", classification_report(y_test, y_pred))

# Matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
print("\n✅ Matrice de confusion :\n", conf_matrix)

# AUC-ROC
auc_roc = roc_auc_score(y_test, y_proba)
print("\n✅ AUC-ROC :\n", auc_roc)

#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Récupérer l'importance des features
importances = model.feature_importances_
feature_names = X.columns  # Remplace X par le DataFrame utilisé pour entraîner le modèle

# Créer un DataFrame pour trier les features par importance
feat_importance = pd.DataFrame({"Feature": feature_names, "Importance": importances})
feat_importance = feat_importance.sort_values(by="Importance", ascending=False).head(10)

# Afficher le graphique
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feat_importance, palette="viridis")
plt.title("Top 10 des features les plus importantes")
plt.show()

#%%
features = ['total_claim_amount', 'vehicle_claim', 'injury_claim', 'property_claim']

for feature in features:
    df.fillna({feature: df[feature].median()}, inplace=True)


for feature in features:
    plt.figure(figsize=(8, 6))
    
    plot_empty = True
    
    if len(df[df['fraud_reported'] == 'N'][feature].dropna()) > 0:
        sns.kdeplot(df[df['fraud_reported'] == 'N'][feature], label='Non Fraudulent', fill=True)
        plot_empty = False
    
    if len(df[df['fraud_reported'] == 'Y'][feature].dropna()) > 0:
        sns.kdeplot(df[df['fraud_reported'] == 'Y'][feature], label='Fraudulent', fill=True)
        plot_empty = False
    
    plt.title(f'Distribution de {feature}')
    
    # Affiche la légende uniquement si le plot n'est pas vide
    if not plot_empty:
        plt.legend()
    
    plt.show()




#%%

import matplotlib.pyplot as plt
import seaborn as sns

# Supposons que ton DataFrame s'appelle `df` et la cible `target`
important_features = [
    'Insured_hobbies_chess', 'vehicle_claim', 'total_claim_amount',
    'Insured_hobbies_cross-fit', 'property_claim', 'incident_date',
    'injury_claim', 'policy_bind_date', 'policy_annual_premium',
    'incident_severity_Minor Damage'
]

# 🎯 Distribution des variables importantes selon la cible
plt.figure(figsize=(16, 12))
for i, feature in enumerate(important_features[:6], 1):  # On prend les 6 premières
    plt.subplot(3, 2, i)
    sns.kdeplot(df[df['target'] == 0][feature], label='Non Fraudulent', shade=True)
    sns.kdeplot(df[df['target'] == 1][feature], label='Fraudulent', shade=True)
    plt.title(f"Distribution de {feature} par classe")
    plt.legend()

plt.tight_layout()
plt.show()

# 📦 Boxplots des variables importantes
plt.figure(figsize=(16, 12))
for i, feature in enumerate(important_features[:6], 1):
    plt.subplot(3, 2, i)
    sns.boxplot(x='target', y=feature, data=df)
    plt.title(f"Boxplot de {feature} par classe")

plt.tight_layout()
plt.show()

# 🔥 Heatmap des corrélations entre les features importantes
plt.figure(figsize=(10, 8))
corr = df[important_features].corr()
sns.heatmap(corr, cmap='coolwarm', annot=True, fmt=".2f", linewidths=0.5)
plt.title("Heatmap des corrélations entre les features importantes")
plt.show()



import streamlit as st
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

# 🎯 Charger le modèle entraîné
model_path = "xgboost_fraud_model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# 🌟 Fonction pour faire la prédiction
def predict_fraud(input_data):
    input_df = pd.DataFrame([input_data])

    # Obtenir les features attendues par le modèle
    expected_features = model.get_booster().feature_names

    # Réorganiser les colonnes et ajouter les colonnes manquantes avec 0
    input_df = input_df.reindex(columns=expected_features, fill_value=0)

    # Debug : afficher les features avant la prédiction
    print("Features attendues par le modèle :", expected_features)
    print("Features fournies par l'utilisateur :", input_df.columns.tolist())

    # Faire la prédiction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[:, 1][0]
    return prediction, probability

# 🎨 Personnalisation du style
st.markdown("""
    <style>
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        font-size: 16px;
        border-radius: 10px;
        padding: 8px 20px;
        margin-top: 10px;
    }
    .stSuccess {
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }
    .stError {
        background-color: #FF4B4B;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# 🚗 Interface utilisateur Streamlit
st.title("Détection de Fraude en Assurance Auto 🚗")
st.write("Remplissez le formulaire ci-dessous pour vérifier si une réclamation est frauduleuse ou non.")

# 📝 Champs du formulaire
policy_annual_premium = st.number_input("Prime annuelle de la police d'assurance", min_value=0.0)
total_claim_amount = st.number_input("Montant total de la réclamation", min_value=0.0)
vehicle_claim = st.number_input("Montant du sinistre véhicule", min_value=0.0)
injury_claim = st.number_input("Montant du sinistre corporel", min_value=0.0)
property_claim = st.number_input("Montant du sinistre matériel", min_value=0.0)
incident_severity = st.selectbox("Gravité de l'incident", ["Minor Damage", "Major Damage", "Total Loss"])
authorities_contacted = st.selectbox("Autorités contactées", ["Police", "None", "Fire", "Other"])
insured_hobbies_chess = st.checkbox("L'assuré aime jouer aux échecs")
insured_hobbies_crossfit = st.checkbox("L'assuré pratique le CrossFit")

# 🧠 Prédiction
if st.button("Analyser la réclamation"):
    # Préparer les données d'entrée
    input_data = {
        "policy_annual_premium": policy_annual_premium,
        "total_claim_amount": total_claim_amount,
        "vehicle_claim": vehicle_claim,
        "injury_claim": injury_claim,
        "property_claim": property_claim,
        "incident_severity_Minor Damage": 1 if incident_severity == "Minor Damage" else 0,
        "incident_severity_Major Damage": 1 if incident_severity == "Major Damage" else 0,
        "incident_severity_Total Loss": 1 if incident_severity == "Total Loss" else 0,
        "authorities_contacted_Police": 1 if authorities_contacted == "Police" else 0,
        "authorities_contacted_Fire": 1 if authorities_contacted == "Fire" else 0,
        "authorities_contacted_Other": 1 if authorities_contacted == "Other" else 0,
        "insured_hobbies_chess": 1 if insured_hobbies_chess else 0,
        "insured_hobbies_cross-fit": 1 if insured_hobbies_crossfit else 0,
    }
    
    try:
        # Prédiction
        prediction, probability = predict_fraud(input_data)

        # ✅ Affichage du résultat
        if prediction == 1:
            st.error(f"⚠️ Fraude détectée avec une probabilité de {probability:.2f}")
        else:
            st.success(f"✅ Réclamation légitime avec une probabilité de {1 - probability:.2f}")
    except Exception as e:
        st.error(f"❌ Erreur lors de la prédiction : {e}")

# 🔄 Réinitialisation du formulaire
if st.button("Réinitialiser"):
    st.experimental_rerun()

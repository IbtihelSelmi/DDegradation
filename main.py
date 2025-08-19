from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import numpy as np
import joblib

# Charger le modèle sauvegardé
model = joblib.load('model_knn.pkl')


app = Flask(__name__)

def safe_get(key, cast_type=str):
    try:
        return cast_type(request.form[key])
    except (KeyError, ValueError):
        return None

# Charger les données depuis le fichier CSV
data = pd.read_csv('projet.csv', encoding='utf-8', on_bad_lines="skip", sep=';')

# Définir les colonnes à sélectionner
columns_to_select = [
    'num_dosss', 'anne_doss', 'date_of_admission', 'Age', 'Gender',
    'Marital_Status', 'obesity', 'frailty', 'HTA', 'diabetis',
    'complications_degeneratives', 'Heart_disease', 'Cancer_hémato',
    'Cancer_solide', 'Chronicdisease', 'mental_health', 'Anxiety', 'mMRC',
    'charlson_comorbidity', 'tobacco_use', 'alcohol_use', 'medication',
    'ventilatory_support', 'ARDS', 'coma', 'glasgow_coma_scale',
    'heart_rate', 'SBP', 'DBP', 'MBP', 'PF', 'LOS', 'PF_ratio',
    'Urine_output', 'Lactates', 'fever', 'SOFA_score', 'SAPSII', 'Death', 'ventilation', 'FiO2', 'SpO2',
    'Discharge'
]

# Vérifier que toutes les colonnes existent
missing_columns = [col for col in columns_to_select if col not in data.columns]
if missing_columns:
    print("Colonnes manquantes :", missing_columns)
else:
    data_selected = data[columns_to_select].fillna(0)
    data_selected.to_csv('C:\\Users\\DELL\\Desktop\\AI\\degradation\\data_selectionnee_clean.csv',
                         sep=';', index=False, encoding='utf-8')
    print(" ✅  ✅ Le fichier nettoyé a été enregistré avec succès.")

# Selection et nettoyage : convertir les colonnes importantes en valeurs numériques
cols_to_clean = [
    'Age', 'Gender', 'heart_rate', 'SBP', 'DBP', 'Lactates',
    'SOFA_score', 'HTA', 'SpO2', 'SAPSII', 'glasgow_coma_scale', 'FiO2'
]

for col in cols_to_clean:
    data[col] = pd.to_numeric(data[col], errors='coerce')

#  Supprimer les lignes avec des valeurs manquantes dans les colonnes clés
data = data.dropna(subset=cols_to_clean)

# Préparer les variables pour l'entraînement
X = data[cols_to_clean]
y = data['Death']

# Normalisation
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)


#  Diviser les données en train et test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le modèle KNN
# Vérifier si le modèle a déjà été entraîné et sauvegardé
model_path = 'model_knn.pkl'
if not os.path.exists(model_path):
    # Entraîner le modèle KNN une seule fois
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)  # Sauvegarder le modèle entraîné
else:
    model = joblib.load(model_path)  # Charger le modèle sauvegardé
# ---------------------------------------------------------------
# Entraînement et évaluation du modèle Random Forest (pour comparaison)
# from sklearn.ensemble import RandomForestClassifier

# rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_model.fit(X_train, y_train)

# rf_pred = rf_model.predict(X_test)
# rf_accuracy = accuracy_score(y_test, rf_pred)

# print("✅ Précision du modèle Random Forest :", round(rf_accuracy * 100, 2), "%")
# ---------------------------------------------------------------

# Évaluer la précision du modèle KNN
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

#  Route pour la page d'accueil
@app.route('/')
def index():
    return render_template('index.html', accuracy=round(accuracy * 100, 2))

# Route pour la prédiction
@app.route('/predict', methods=['POST'])
def predict():
    def safe_get(field, cast_type, default_value=0):
        try:
            value = request.form.get(field, '')
            return cast_type(value) if value else default_value
        except:
            return default_value

    age = safe_get('age', int)
    gender = safe_get('gender', int)
    heart_rate = safe_get('heart_rate', int)
    sbp = safe_get('SBP', int)
    dbp = safe_get('DBP', int)
    lactates = safe_get('lactates', float)
    sofa_score = safe_get('sofa_score', int)
    hta = safe_get('HTA', int)
    spo2 = safe_get('SpO2', float)
    sapsii = safe_get('SAPSII', int)
    glasgow_coma_scale = safe_get('glasgow_coma_scale', int)
    fio2 = safe_get('FiO2', float)

    # ajout des inputs dans une liste
    input_data = [[age, gender, heart_rate, sbp, dbp, lactates, sofa_score, hta, spo2, sapsii, glasgow_coma_scale, fio2]]

    # Les inputs arrivant de l'interface
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        prediction_text = "🔴 🔴 Le patient présente un risque élevé de dégradation aiguë."
    else:
        prediction_text = "🟢 🟢 Risque faible de dégradation aiguë."

    return render_template('index.html', prediction=prediction_text, accuracy=round(accuracy * 100, 2))

# Lancer l'application Flask
if __name__ == '__main__':
    app.run(debug=True)

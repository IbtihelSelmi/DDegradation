import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

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
    print("✅ Le fichier nettoyé a été enregistré avec succès.")

# Nettoyage : convertir les colonnes importantes en valeurs numériques
cols_to_clean = [
    'Age', 'Gender', 'heart_rate', 'SBP', 'DBP', 'Lactates',
    'SOFA_score', 'HTA', 'SpO2', 'SAPSII', 'glasgow_coma_scale', 'FiO2'
]

for col in cols_to_clean:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Supprimer les lignes avec des valeurs manquantes dans les colonnes clés
data = data.dropna(subset=cols_to_clean)

# Préparer les variables pour l'entraînement
X = data[cols_to_clean]
y = data['Death']

# Normalisation
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Diviser les données en train et test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le modèle KNN
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Créer le dossier "degradation" s'il n'existe pas
os.makedirs('degradation', exist_ok=True)

# Sauvegarder le modèle
joblib.dump(model, 'degradation/train_model.pkl')

# Évaluer la précision du modèle
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("✅ Précision du modèle KNN :", round(accuracy * 100, 2), "%")

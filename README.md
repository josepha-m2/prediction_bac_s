# 🎓 Prédiction des Résultats du Bac

Ce projet utilise le Machine Learning pour prédire les résultats d'élèves au baccalauréat à partir de différentes données (notes, caractéristiques, etc.).

---

## 📌 Objectif

L'objectif est de :

* Analyser les performances des élèves
* Construire un modèle de prédiction
* Évaluer la précision du modèle

---

## 🧰 Technologies utilisées

* Python 3.x
* pandas
* numpy
* scikit-learn
* matplotlib / seaborn
* streamlit

---

## 📂 Structure du projet

```
prediction_bac_s/
│── bac_serie_S_dataset.csv # Dataset              # 
│── Pretraitement.py   # Nettoyage et préparation des données
│── Traitement.py      # Entraînement du modèle
│── streamlit_app.py   # Interface utilisateur
│── requirements.txt   # Dépendances
│── README.md
```

---

## ⚙️ Installation

1. Cloner le projet :

```bash
git clone https://github.com/josepha-m2/prediction_bac_s.git
cd prediction_bac_s
```

2. Créer un environnement virtuel :

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

3. Installer les dépendances :

```bash
pip install -r requirements.txt
```

---

## ▶️ Exécution

### 1. Prétraitement des données

```bash
python Pretraitement.py
```

➡️ Ce script :

* charge les données brutes (CSV)
* nettoie les valeurs manquantes
* prépare les variables pour le modèle

---

### 2. Entraînement du modèle

```bash
python Traitement.py
```

➡️ Ce script :

* utilise les données prétraitées
* entraîne un modèle de Machine Learning
* génère les prédictions

---

### 3. Lancer l’application web

```bash
streamlit run streamlit_app.py
```

➡️ Cette application permet :

* d’interagir avec le modèle
* de tester des prédictions
* de visualiser les résultats

---

## 📊 Fonctionnalités

* Chargement et nettoyage des données
* Analyse exploratoire
* Entraînement du modèle
* Prédiction des résultats
* Interface interactive avec Streamlit

---

## 📈 Modèles utilisés

* Régression logistique

---

## 🤝 Contribution

Les contributions sont les bienvenues :

1. Fork
2. Commit
3. Pull Request

---

## 📜 Licence

Projet open-source

---

## 👤 Auteur

* Josepha M2

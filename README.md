# 🎓 Application de Prédiction de Réussite au BAC S

Cette application utilise l'intelligence artificielle (Régression Logistique) pour estimer les chances de réussite d'un élève au Baccalauréat Scientifique (S) en fonction de son profil et de ses résultats scolaires en classe de Première et Terminale.

## 🚀 Fonctionnalités
- **Interface intuitive** : Formulaire par étapes pour une saisie fluide des données.
- **Analyse du profil** : Prise en compte de l'âge, du sexe et de la moyenne de Première.
- **Évaluation scientifique** : Calcul automatique d'un score basé sur les notes en Mathématiques, Physique et SVT des deux semestres.
- **Suivi de progression** : Analyse de l'évolution entre le semestre 1 et le semestre 2.
- **Indicateurs de risque** : Affichage d'un statut (Favorable, Zone critique, Risque élevé) basé sur la probabilité calculée.

## 🛠️ Stack Technique
- **Langage** : Python
- **Framework Web** : [Streamlit](https://streamlit.io/)
- **Machine Learning** : Scikit-learn (Régression Logistique)
- **Traitement de données** : Pandas & Numpy
- **Sérialisation** : Joblib

## 📁 Structure du Projet
```text
├── Predict_20260420.py      # Script principal de l'application Streamlit
├── requirements.txt         # Dépendances du projet
├── ens.jpeg                 # Logo de l'établissement/application
├── Traitement(model_logit).pkl # Modèle entraîné
├── Traitement(scaler).pkl      # Normaliseur de données
├── Traitement(features).pkl    # Noms des variables d'entrée
└── Traitement(threshold).pkl   # Seuil de décision optimal

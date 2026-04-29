import streamlit as st
import pandas as pd
import numpy as np
import joblib
import hashlib
import time

# ─── CONFIG ───
st.set_page_config(page_title="Sécurité | Prédiction BAC S", layout="centered")

# ─── SÉCURITÉ & AUTHENTIFICATION ───

# Simulation d'un stockage sécurisé (SHA-256)
# Le hash correspond à "josepha@2026"
USER_AUTH = {
    "Josepha": "03ec38d890e05980b6104f994eb562e32e19152a1a2914538e2da349e2283af6"
}

def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_auth():
    """Gère l'interface de connexion"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        st.markdown("<h2 style='text-align:center;'>🔐Connexion</h2>", unsafe_allow_html=True)
        
        # Initialisation du compteur anti-bruteforce
        if "attempts" not in st.session_state:
            st.session_state.attempts = 0
        
        if st.session_state.attempts >= 5:
            st.error("⚠️ Trop de tentatives. Accès bloqué temporairement.")
            return False

        with st.form("login_form"):
            user = st.text_input("Utilisateur")
            pw = st.text_input("Mot de passe", type="password")
            submit = st.form_submit_button("Se connecter")

            if submit:
                # Anti-Injection : On vérifie les clés du dictionnaire, pas de requêtes brutes
                hashed_pw = hash_password(pw)
                if user in USER_AUTH and USER_AUTH[user] == hashed_pw:
                    st.session_state.authenticated = True
                    st.session_state.attempts = 0
                    st.rerun()
                else:
                    st.session_state.attempts += 1
                    st.error(f"Identifiants invalides ({st.session_state.attempts}/5)")
        return False
    return True

# ─── STYLE ───
st.markdown("""
<style>
.main { background-color: #F8F9FA; }
.stButton>button { width: 100%; border-radius: 10px; background-color: #2196F3; color: white; }
.step-box { padding: 25px; border-radius: 15px; background-color: white; box-shadow: 0 4px 10px rgba(0,0,0,0.08); }
</style>
""", unsafe_allow_html=True)

# ─── EXECUTION PRINCIPALE ───
if check_auth():
    # Bouton de déconnexion dans la sidebar
    if st.sidebar.button("Déconnexion"):
        st.session_state.authenticated = False
        st.rerun()

    # ─── LOAD MODEL ───
    @st.cache_resource
    def load_models():
        try:
            model = joblib.load('model_logit.pkl')
            scaler = joblib.load('scaler.pkl')
            features_names = joblib.load('features.pkl')
            threshold = joblib.load('threshold.pkl')
            return model, scaler, features_names, threshold
        except:
            return None, None, None, None

    model, scaler, features_names, threshold = load_models()
    
    if model is None:
        st.error("⚠️ Fichiers modèles (.pkl) introuvables dans le repository.")
        st.stop()

    # ─── STATE DU FORMULAIRE ───
    if 'step' not in st.session_state:
        st.session_state.step = 1

    def next_step(): st.session_state.step += 1
    def prev_step(): st.session_state.step -= 1

    # ─── HEADER ───
    col1, col2 = st.columns([1, 4])
    with col1:
        try: st.image("ens.jpeg", width=100)
        except: pass
    with col2:
        st.markdown("<h2>Réussite scolaire - BAC S</h2>", unsafe_allow_html=True)

    st.progress(st.session_state.step / 3)
    st.write("---")

    # ─── CONTENU DU FORMULAIRE (Ton code original condensé) ───
    with st.container():
        #st.markdown('<div class="step-box">', unsafe_allow_html=True)

        if st.session_state.step == 1:
            st.subheader("👤 Profil")
            sexe = st.selectbox("Sexe", ["Masculin", "Féminin"])
            age = st.number_input("Âge", min_value=17, max_value=20, value=17)
            moy = st.slider("Moyenne Première", 10.0, 20.0, 10.0, step=0.25)
            st.session_state.sexe = 0 if sexe == "Masculin" else 1
            st.session_state.retard = 1 if age > 18 else 0
            st.session_state.moy_premiere = moy
            st.button("Suivant", on_click=next_step)

        elif st.session_state.step == 2:
            st.subheader("📚 Notes scientifiques")
            c1, c2 = st.columns(2)
            with c1:
                m1 = st.number_input("Math S1", 0.0, 20.0, 10.0)
                p1 = st.number_input("Physique S1", 0.0, 20.0, 10.0)
                s1 = st.number_input("SVT S1", 0.0, 20.0, 10.0)
            with c2:
                m2 = st.number_input("Math S2", 0.0, 20.0, 10.0)
                p2 = st.number_input("Physique S2", 0.0, 20.0, 10.0)
                s2 = st.number_input("SVT S2", 0.0, 20.0, 10.0)
            
            st.session_state.score_scientifique = ((m1+p1+s1)/3 + (m2+p2+s2)/3) / 2
            st.session_state.progression = (m2+p2+s2)/3 - (m1+p1+s1)/3
            
            col_b1, col_b2 = st.columns(2)
            col_b1.button("Précédent", on_click=prev_step)
            col_b2.button("Suivant", on_click=next_step)

        elif st.session_state.step == 3:
            st.subheader("🎯 Résultat")
            absenteisme = st.number_input("Absentéisme (%)", 0.0, 100.0, 5.0)

            if st.button("📊 Prédire"):
                data = pd.DataFrame([[
                    st.session_state.score_scientifique, absenteisme,
                    st.session_state.progression, st.session_state.moy_premiere,
                    st.session_state.sexe, st.session_state.retard
                ]], columns=features_names)

                scaled = scaler.transform(data)
                prob = model.predict_proba(scaled)[0, 1]

                if prob >= threshold:
                    st.success(f"✅ ADMIS (Probabilité: {prob:.2%})")
                    st.balloons()
                else:
                    st.error(f"❌ ÉCHEC (Probabilité: {prob:.2%})")

            st.button("🔄 Recommencer", on_click=lambda: st.session_state.update(step=1))

        st.markdown('</div>', unsafe_allow_html=True)

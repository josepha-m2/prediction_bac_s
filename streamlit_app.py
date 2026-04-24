import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ─── CONFIG ───
st.set_page_config(page_title="Prédiction BAC S", layout="centered")

# ─── STYLE ───
st.markdown("""
<style>
.main { background-color: #F8F9FA; }
.stButton>button {
    width: 100%;
    border-radius: 10px;
    background-color: #2196F3;
    color: white;
}
.step-box {
    padding: 25px;
    border-radius: 15px;
    background-color: white;
    box-shadow: 0 4px 10px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# ─── LOAD MODEL ───
@st.cache_resource
def load_models():
    model = joblib.load('model_logit.pkl')
    scaler = joblib.load('scaler.pkl')
    features_names = joblib.load('features.pkl')
    threshold = joblib.load('threshold.pkl')
    return model, scaler, features_names, threshold

try:
    model, scaler, features_names, threshold = load_models()
except:
    st.error("⚠️ Modèle introuvable")
    st.stop()

# ─── STATE ───
if 'step' not in st.session_state:
    st.session_state.step = 1

def next_step(): st.session_state.step += 1
def prev_step(): st.session_state.step -= 1

# ─── HEADER ───
col1, col2 = st.columns([1, 4])

with col1:
    try:
        st.image("ens.jpeg", width=100)
    except:
        pass

with col2:
    st.markdown("""
    <h2>Réussite scolaire - BAC S</h2>
    <p style='color:gray;'>Modèle de régression logistique</p>
    """, unsafe_allow_html=True)

# Barre de progression
st.progress(st.session_state.step / 3)

st.write("---")

# ─── FORM ───
with st.container():
    st.markdown('<div class="step-box">', unsafe_allow_html=True)

    # ─── STEP 1 ───
    if st.session_state.step == 1:
        st.subheader("👤 Profil")

        sexe = st.selectbox("Sexe", ["Masculin", "Féminin"])
        age = st.number_input("Âge", min_value=17, max_value=20, value=17)
        moy = st.slider("Moyenne Première", 10.0, 20.0, 10.0, step=0.25)

        st.session_state.sexe = 0 if sexe == "Masculin" else 1
        st.session_state.retard = 1 if age > 18 else 0
        st.session_state.moy_premiere = moy

        st.button("Suivant", on_click=next_step)

    # ─── STEP 2 ───
    elif st.session_state.step == 2:
        st.subheader("📚 Notes scientifiques")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Semestre 1**")
            m1 = st.number_input("Math S1", 0.0, 20.0, 10.0, step=0.25, format="%.2f")
            p1 = st.number_input("Physique S1", 0.0, 20.0, 10.0, step=0.25, format="%.2f")
            s1 = st.number_input("SVT S1", 0.0, 20.0, 10.0, step=0.25, format="%.2f")

        with col2:
            st.markdown("**Semestre 2**")
            m2 = st.number_input("Math S2", 0.0, 20.0, 10.0, step=0.25, format="%.2f")
            p2 = st.number_input("Physique S2", 0.0, 20.0, 10.0, step=0.25, format="%.2f")
            s2 = st.number_input("SVT S2", 0.0, 20.0, 10.0, step=0.25, format="%.2f")

        # Calculs
        moy_s1 = (m1 + p1 + s1) / 3
        moy_s2 = (m2 + p2 + s2) / 3

        st.session_state.score_scientifique = (moy_s1 + moy_s2) / 2
        st.session_state.progression = moy_s2 - moy_s1

        st.info(f"Score scientifique : {st.session_state.score_scientifique:.2f}")
        st.info(f"Progression S1 → S2 : {st.session_state.progression:.2f}")

        c1, c2 = st.columns(2)
        c1.button("Précédent", on_click=prev_step)
        c2.button("Suivant", on_click=next_step)

    # ─── STEP 3 ───
    elif st.session_state.step == 3:
        st.subheader("🎯 Résultat")

        absenteisme = st.number_input("Absentéisme (%)", 0.0, 100.0, 5.0)

        if st.button("📊 Prédire"):

            data = pd.DataFrame([[
                st.session_state.score_scientifique,
                absenteisme,
                st.session_state.progression,
                st.session_state.moy_premiere,
                st.session_state.sexe,
                st.session_state.retard
            ]], columns=features_names)

            scaled = scaler.transform(data)
            prob = model.predict_proba(scaled)[0, 1]

            st.markdown("---")

            if prob >= threshold:
                color = "green"
                result = "✅ ADMIS"
                st.balloons()
            else:
                color = "red"
                result = "❌ ÉCHEC"

            st.markdown(f"<h1 style='text-align:center; color:{color};'>{result}</h1>", unsafe_allow_html=True)

            if prob > 0.65:
                color_stat = "green"
                statut = "Favorable"
            elif prob > 0.35:
                color_stat = "orange"
                statut = "Zone critique"
            else:
                color_stat = "red"
                statut = "Risque élevé"

            st.markdown(
                f"<h3 style='text-align:center;'>Statut : <span style='color:{color_stat};'>{statut}</span></h3>",
                unsafe_allow_html=True
            )

        st.button("🔄 Recommencer", on_click=lambda: st.session_state.update(step=1))

    st.markdown('</div>', unsafe_allow_html=True)

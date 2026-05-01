import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection    import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing      import StandardScaler
from sklearn.linear_model       import LogisticRegression
from sklearn.metrics            import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    ConfusionMatrixDisplay
)
from sklearn.calibration        import calibration_curve
from sklearn.model_selection import learning_curve
import statsmodels.api          as sm
from scipy                      import stats
import joblib

C = dict(admis='#2ECC71', echec='#E74C3C', primary='#2C3E50',
         accent='#3498DB', gold='#F39C12', bg='#F8F9FA', grey='#95A5A6')

print("=" * 70)
print("  PHASE 3 — RÉGRESSION LOGISTIQUE ")
print("  Projet : Réussite BAC Série S — LRR")
print("=" * 70)

print("\n📥 [0] Chargement du dataset préprocessé (Phase 2)...")
df = pd.read_csv('Pretraitement_dataset.csv')
print(f"   ✔ {df.shape[0]} obs × {df.shape[1]} variables chargées")

print("\n🔍 [1] Sélection des variables (stratégie post-VIF)...")

FEATURES = [
    'score_scientifique',       # synthèse pondérée officielle
    'taux_absenteisme',         # comportement scolaire
    'progression_s1_s2',        # dynamique S1→S2
    'moyenne_generale_premiere', # performance Première
    'sexe',                     # démographique
    'retard_scolaire',          # âge relatif
]
TARGET = 'reussite_bac'

X = df[FEATURES].copy()
y = df[TARGET].copy()

print(f"   ✔ {len(FEATURES)} prédicteurs sélectionnés : {FEATURES}")
print(f"   ✔ Distribution cible : Admis={y.sum()} ({y.mean()*100:.1f}%) | Échec={len(y)-y.sum()}")

print("\n✂️  [2] Split Train/Test stratifié (80% / 20%)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"   ✔ Train : {len(X_train)} obs | Test : {len(X_test)} obs")
print(f"   ✔ Taux réussite Train : {y_train.mean()*100:.1f}% | Test : {y_test.mean()*100:.1f}%")

print("\n📏 [3] Standardisation (fit sur Train uniquement — évite le data leakage)...")
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)       # transform seulement !
print("   ✔ Standardisation sans fuite d'information")

print("\n📐 [4] Régression Logistique — Statsmodels (inférence économétrique)...")

X_train_sm = sm.add_constant(X_train_sc)
X_test_sm  = sm.add_constant(X_test_sc)

logit_sm   = sm.Logit(y_train, X_train_sm)
result_sm  = logit_sm.fit(method='bfgs', maxiter=200, disp=False)

print("\n" + "=" * 70)
print(result_sm.summary2().as_text())
print("=" * 70)

coef_names = ['Constante'] + FEATURES
params     = result_sm.params
pvalues    = result_sm.pvalues
conf       = result_sm.conf_int()

or_table = pd.DataFrame({
    'Variable'  : coef_names,
    'β (coeff)' : params.values.round(4),
    'Odds Ratio': np.exp(params.values).round(4),
    'OR IC 2.5%': np.exp(conf.iloc[:, 0].values).round(4),
    'OR IC 97.5%':np.exp(conf.iloc[:, 1].values).round(4),
    'p-value'   : pvalues.values.round(4),
    'Sig.'      : ['***' if p < 0.001 else '**' if p < 0.01
                   else '*' if p < 0.05 else '.' if p < 0.1
                   else '' for p in pvalues.values]
})
print("\n📊 TABLEAU DES ODDS RATIOS :")
print(or_table.to_string(index=False))

llf  = result_sm.llf
llnull = result_sm.llnull
mcfadden_r2 = 1 - (llf / llnull)
print("\n🤖 [5] Régression Logistique — Sklearn (évaluation prédictive)...")

logit_sk = LogisticRegression(C=1.0, solver='lbfgs', max_iter=500, random_state=42)
logit_sk.fit(X_train_sc, y_train)

y_pred       = logit_sk.predict(X_test_sc)
y_prob       = logit_sk.predict_proba(X_test_sc)[:, 1]
y_prob_train = logit_sk.predict_proba(X_train_sc)[:, 1]

print("\n   ── RAPPORT DE CLASSIFICATION (Test) ──")
print(classification_report(y_test, y_pred, target_names=['Échec', 'Admis']))

auc_roc = roc_auc_score(y_test, y_prob)
ap      = average_precision_score(y_test, y_prob)

print(f"   AUC-ROC                 : {auc_roc:.4f}")
print(f"   Average Precision (AP)  : {ap:.4f}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(logit_sk, X_train_sc, y_train, cv=cv, scoring='roc_auc')
print(f"\n   Validation croisée 5-fold AUC-ROC :")
print(f"   Scores   : {[round(s,4) for s in cv_scores]}")
print(f"   Moyenne  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

print("\n📈 Learning Curve (détection overfitting/underfitting)...")

train_sizes, train_scores, val_scores = learning_curve(
    logit_sk,
    X_train_sc,
    y_train,
    cv=5,
    scoring='roc_auc',
    train_sizes=np.linspace(0.1, 1.0, 10),
    shuffle=True,
    random_state=42
)

train_mean = train_scores.mean(axis=1)
val_mean   = val_scores.mean(axis=1)

plt.figure(figsize=(8,6))

plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Train AUC')
plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation AUC')

plt.title("Learning Curve - Logit Model")
plt.xlabel("Taille des données d'entraînement")
plt.ylabel("AUC-ROC")
plt.grid(alpha=0.3)
plt.legend()

plt.savefig("Traitement(learning_curve).png", dpi=150)
plt.close()

print("✔ Learning curve sauvegardée : Traitement(learning_curve).png")
print("\n🎯 [6] Recherche du seuil optimal (Youden Index)...")
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
youden_idx  = np.argmax(tpr - fpr)
best_thresh = thresholds[youden_idx]
print(f"   ✔ Seuil optimal (Youden) : {best_thresh:.4f}")
print(f"   Sensibilité (TPR)       : {tpr[youden_idx]:.4f}")
print(f"   Spécificité (1-FPR)     : {1-fpr[youden_idx]:.4f}")

y_pred_opt = (y_prob >= best_thresh).astype(int)
print(f"\n   Rapport avec seuil optimal ({best_thresh:.2f}) :")
print(classification_report(y_test, y_pred_opt, target_names=['Échec', 'Admis']))

print("\n📊 [7] Génération des visualisations...")

fig = plt.figure(figsize=(20, 16), facecolor=C['bg'])
fig.suptitle(
    "Phase 3 — Modélisation Logit | BAC Série S | LRR",
    fontsize=16, fontweight='bold', color=C['primary'], y=0.99
)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(fpr, tpr, color=C['accent'], lw=2.5, label=f'Logit (AUC = {auc_roc:.3f})')
ax1.plot([0,1],[0,1], 'k--', lw=1.2, alpha=0.5, label='Aléatoire')
ax1.scatter(fpr[youden_idx], tpr[youden_idx], color=C['gold'], s=100,
            zorder=5, label=f'Seuil optimal ({best_thresh:.2f})')
ax1.set_xlabel('Taux faux positifs (FPR)', fontsize=9)
ax1.set_ylabel('Taux vrais positifs (TPR)', fontsize=9)
ax1.set_title('Courbe ROC', fontweight='bold')
ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
cm = confusion_matrix(y_test, y_pred_opt)
cmd = ConfusionMatrixDisplay(cm, display_labels=['Échec', 'Admis'])
cmd.plot(ax=ax2, colorbar=False, cmap='Blues')
ax2.set_title('Matrice de Confusion\n(seuil optimal)', fontweight='bold')

ax3 = fig.add_subplot(gs[0, 2])
or_vals = or_table[or_table['Variable'] != 'Constante']
colors_or = [C['admis'] if v > 1 else C['echec'] for v in or_vals['Odds Ratio']]
bars = ax3.barh(or_vals['Variable'], or_vals['Odds Ratio'] - 1,
                left=1, color=colors_or, alpha=0.85, edgecolor='white')
ax3.axvline(1, color='black', lw=1.5, ls='--')
for i, (v, p) in enumerate(zip(or_vals['Odds Ratio'], or_vals['Sig.'])):
    ax3.text(v + 0.02, i, f'{v:.2f}{p}', va='center', fontsize=8, fontweight='bold')
ax3.set_title('Odds Ratios\n(vert > 1 = facteur de réussite)', fontweight='bold')
ax3.set_xlabel('OR')
ax3.grid(axis='x', alpha=0.3)

ax5 = fig.add_subplot(gs[1, 0])
prec, rec, _ = precision_recall_curve(y_test, y_prob)
ax5.plot(rec, prec, color=C['accent'], lw=2.5, label=f'AP = {ap:.3f}')
ax5.axhline(y_test.mean(), color='grey', ls='--', lw=1.2, label='Baseline')
ax5.set_xlabel('Rappel (Recall)')
ax5.set_ylabel('Précision')
ax5.set_title('Courbe Précision-Rappel', fontweight='bold')
ax5.legend(fontsize=8); ax5.grid(alpha=0.3)

ax6 = fig.add_subplot(gs[1, 1])
fold_labels = [f'Fold {i+1}' for i in range(5)]
bars_cv = ax6.bar(fold_labels, cv_scores, color=C['accent'], alpha=0.8, edgecolor='white')
ax6.axhline(cv_scores.mean(), color=C['gold'], lw=2, ls='--',
            label=f'Moy = {cv_scores.mean():.3f}')
for bar, v in zip(bars_cv, cv_scores):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
             f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax6.set_ylim(0.7, 1.02)
ax6.set_ylabel('AUC-ROC')
ax6.set_title('Validation Croisée 5-Fold\n(Stabilité du modèle)', fontweight='bold')
ax6.legend(fontsize=8); ax6.grid(axis='y', alpha=0.3)

ax9 = fig.add_subplot(gs[1, 2])
score_range = np.linspace(df['score_scientifique'].min(), df['score_scientifique'].max(), 300)
other_means  = np.zeros((300, len(FEATURES)))
other_means[:, 0] = score_range
other_means[:, 1] = df['taux_absenteisme'].mean()
other_means[:, 2] = df['progression_s1_s2'].mean()
other_means[:, 3] = df['moyenne_generale_premiere'].mean()
other_means[:, 4] = df['sexe'].mean()
other_means[:, 5] = df['retard_scolaire'].mean()
other_means_sc = scaler.transform(other_means)
prob_curve = logit_sk.predict_proba(other_means_sc)[:,1]

ax9.plot(score_range, prob_curve, color=C['accent'], lw=2.5)
ax9.axhline(0.5, color=C['gold'], ls='--', lw=1.5, label='Seuil 0.05')
ax9.scatter(df['score_scientifique'], y + np.random.normal(0, 0.02, len(y)),
            alpha=0.15, c=[C['admis'] if v==1 else C['echec'] for v in y.values], s=20)
ax9.set_xlabel('Score Scientifique (/20)')
ax9.set_ylabel('P(Admis)')
ax9.set_title('Courbe Logistique\nScore → P(Réussite)', fontweight='bold')
ax9.legend(fontsize=8); ax9.grid(alpha=0.3)

plt.savefig('Traitement(analyse).png',
            dpi=150, bbox_inches='tight', facecolor=C['bg'])
plt.close()
print("   ✔ Figure Phase 3 sauvegardée → Traitement(analyse).png")

print("\n💾 [8] Export des prédictions et résultats...")

X_full_sc = scaler.transform(X)
df_results = df.copy()
df_results['prob_admis']    = logit_sk.predict_proba(X_full_sc)[:,1].round(4)
df_results['prediction']    = (df_results['prob_admis'] >= best_thresh).astype(int)
df_results['correct']       = (df_results['prediction'] == df_results[TARGET]).astype(int)
df_results['zone_risque']   = pd.cut(df_results['prob_admis'],
    bins=[0, 0.35, 0.65, 1.0],
    labels=['🔴 Risque élevé', '🟡 Zone critique', '🟢 Favorable'])

df_results.to_csv('Traitement(bac_predictions).csv', index=False)
print("   ✔ bac_predictions.csv exporté")

print("\n   Distribution des zones de risque (jeu complet) :")
print(df_results['zone_risque'].value_counts().to_string())

print("\n" + "=" * 70)
print("  SYNTHÈSE ÉCONOMÉTRIQUE FINALE")
print("=" * 70)

print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │                INDICATEURS DE PERFORMANCE DU MODÈLE             │
  ├──────────────────────────────────┬──────────────────────────────┤
  │  AUC-ROC                         │  {auc_roc:.4f}                     │
  │  Validation croisée (AUC moy.)   │  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}         │
  │  Seuil optimal (Youden)          │  {best_thresh:.4f}                     │
  └──────────────────────────────────┴──────────────────────────────┘
""")

sig_vars = or_table[(or_table['Variable'] != 'Constante') &
                    (or_table['p-value'] < 0.05)].sort_values('Odds Ratio', ascending=False)

print("  Variables statistiquement significatives (p < 0.05) :")
for _, row in sig_vars.iterrows():
    direction = "↑ Facteur de RÉUSSITE" if row['Odds Ratio'] > 1 else "↓ Facteur de RISQUE"
    print(f"   • {row['Variable']:30s}  OR={row['Odds Ratio']:.3f}  {row['Sig.']:3s}  → {direction}")

joblib.dump(logit_sk, 'model_logit.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(FEATURES, 'features.pkl')
joblib.dump(best_thresh, 'threshold.pkl')

print("✅ Modèle sauvegardé")

print("✅ Phase 3 terminée!")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

COLORS = {
    'admis':   '#2ECC71',
    'echec':   '#E74C3C',
    'primary': '#2C3E50',
    'accent':  '#3498DB',
    'bg':      '#F8F9FA',
}

print("=" * 70)
print("  PHASE 2 — PRÉTRAITEMENT & FEATURE ENGINEERING")
print("  Projet : Réussite scolaire — LRR")
print("=" * 70)

print("\n📥 [1] Chargement du dataset brut...")
df_raw = pd.read_csv('bac_serie_S_dataset.csv')

print(f"   ✔ Dimensions brutes : {df_raw.shape[0]} observations × {df_raw.shape[1]} variables")
print("\n   Aperçu des premières lignes :")
print(df_raw.head(3).to_string(index=False))
print("\n   Types de données :")
print(df_raw.dtypes.to_string())

print("\n🧹 [2] Nettoyage des données...")
df = df_raw.copy()

df.drop(columns=['id_eleve', 'annee'], inplace=True)
print("   ✔ Colonnes supprimées : id_eleve, annee")

df['sexe']         = df['sexe'].map({'F': 1, 'M': 0})
print("   ✔ Encodage : Sexe (F=1, M=0) | Réussite déjà binaire (1=Admis, 0=Échec)")

note_cols = [c for c in df.columns if 'notes_' in c or 'moyenne_' in c]
invalid_count = 0
for col in note_cols:
    mask = (df[col] < 0) | (df[col] > 20)
    n_inv = mask.sum()
    if n_inv > 0:
        print(f"   ⚠ {col} : {n_inv} valeur(s) hors [0,20] → remplacées par NaN")
        df.loc[mask, col] = np.nan
    invalid_count += n_inv
if invalid_count == 0:
    print("   ✔ Toutes les notes sont dans [0, 20]")

missing_before = df.isnull().sum().sum()
print(f"\n   Valeurs manquantes détectées : {missing_before}")
if missing_before > 0:
    for col in df.select_dtypes(include='number').columns:
        if df[col].isnull().any():
            med = df[col].median()
            df[col].fillna(med, inplace=True)
            print(f"   ✔ Imputation médiane → {col} : médiane = {med:.2f}")
else:
    print("   ✔ Aucune valeur manquante — aucune imputation nécessaire")

print("\n📈 [3] Feature Engineering — Variables dérivées à haute valeur prédictive...")

df['math_moy']     = (df['notes_math_semestre_1']     + df['notes_math_semestre_2'])     / 2
df['physique_moy'] = (df['notes_physique_semestre_1'] + df['notes_physique_semestre_2']) / 2
df['svt_moy']      = (df['notes_SVT_semestre_1']      + df['notes_SVT_semestre_2'])      / 2
print("   ✔ Moyennes annuelles : math_moy | physique_moy | svt_moy")

df['score_scientifique'] = (
    (df['math_moy'] * 6) + (df['physique_moy'] * 6) + (df['svt_moy'] * 6)
) / 18
print("   ✔ Score scientifique pondéré (Math×6 + Physique×6 + SVT×6) / 18")

somme_S2 = df['notes_math_semestre_2'] + df['notes_physique_semestre_2'] + df['notes_SVT_semestre_2']
somme_S1 = df['notes_math_semestre_1'] + df['notes_physique_semestre_1'] + df['notes_SVT_semestre_1']
df['progression_s1_s2'] = somme_S2 - somme_S1
print("   ✔ Indicateur de progression : Σ(S2) - Σ(S1)")

df['retard_scolaire'] = df['age'] - 18
print("   ✔ Retard scolaire : Âge - 18")

raw_note_cols = [
    'notes_math_semestre_1', 'notes_math_semestre_2',
    'notes_physique_semestre_1', 'notes_physique_semestre_2',
    'notes_SVT_semestre_1', 'notes_SVT_semestre_2',
    'age'
]
df.drop(columns=raw_note_cols, inplace=True)

print(f"\n   DataFrame après feature engineering : {df.shape[0]} obs × {df.shape[1]} variables")
print("   Variables retenues :", list(df.columns))

print("\n🔬 [4] Calcul du Variance Inflation Factor (VIF)...")
target    = 'reussite_bac'
num_cols  = [c for c in df.columns if c != target]

X_vif = df[num_cols]

vif_data = pd.DataFrame({
    'Variable': num_cols,
    'VIF': [
        variance_inflation_factor(X_vif.values, i)
        for i in range(X_vif.shape[1])
    ]
}).sort_values('VIF', ascending=False).reset_index(drop=True)

vif_data['Interprétation'] = vif_data['VIF'].apply(
    lambda v: '🔴 CRITIQUE (> 10)' if v > 10
    else ('🟠 Modéré (5-10)' if v > 5
    else '🟢 Acceptable (< 5)')
)

print("\n   ╔══ TABLEAU DES VIF ═══════════════════════════════════════════╗")
print(vif_data.to_string(index=False))
print("   ╚══════════════════════════════════════════════════════════════╝")

critical_vif = vif_data[vif_data['VIF'] > 10]
if not critical_vif.empty:
    print("\n   ⚠ STRATÉGIE DE RÉDUCTION DE MULTICOLINÉARITÉ :")
    print("   Variables critiques :", list(critical_vif['Variable']))
    print("   → Option 1 : Supprimer les moyennes matière (math_moy, physique_moy,")
    print("     svt_moy) et ne conserver que le score_scientifique (synthèse pondérée).")
    print("   → Option 2 : ACP pour créer un composant orthogonal 'Performance S'.")
    print("   → Option 3 : Régularisation Ridge/Lasso pour neutraliser la collinéarité.")
    print("   → Choix recommandé : conserver score_scientifique + progression_s1_s2")
    print("     + moyenne_generale_premiere → modèle parcimonieux & stable.")
else:
    print("\n   ✔ Aucune variable avec VIF > 10 — multicolinéarité sous contrôle.")

print("\n📊 [5] Analyse exploratoire ciblée — Génération des graphiques...")

fig = plt.figure(figsize=(18, 14), facecolor=COLORS['bg'])
fig.suptitle(
    "Analyse pretraitement — BAC Série S | LRR",
    fontsize=16, fontweight='bold', color=COLORS['primary'], y=0.98
)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

admis = df[df[target] == 1]
echec = df[df[target] == 0]
labels_box = ['Échec (0)', 'Admis (1)']

ax1 = fig.add_subplot(gs[0, 0])
bp = ax1.boxplot(
    [echec['score_scientifique'], admis['score_scientifique']],
    patch_artist=True,
    labels=labels_box,
    medianprops=dict(color='white', linewidth=2.5)
)
for patch, color in zip(bp['boxes'], [COLORS['echec'], COLORS['admis']]):
    patch.set_facecolor(color); patch.set_alpha(0.8)
ax1.set_title('Score Scientifique\nvs Réussite', fontweight='bold')
ax1.set_ylabel('Score pondéré (/20)')
ax1.grid(axis='y', alpha=0.4)

ax2 = fig.add_subplot(gs[0, 1])
bp2 = ax2.boxplot(
    [echec['taux_absenteisme'], admis['taux_absenteisme']],
    patch_artist=True,
    labels=labels_box,
    medianprops=dict(color='white', linewidth=2.5)
)
for patch, color in zip(bp2['boxes'], [COLORS['echec'], COLORS['admis']]):
    patch.set_facecolor(color); patch.set_alpha(0.8)
ax2.set_title('Taux d\'Absentéisme\nvs Réussite', fontweight='bold')
ax2.set_ylabel('Taux (%)')
ax2.grid(axis='y', alpha=0.4)

ax3 = fig.add_subplot(gs[0, 2])
bp3 = ax3.boxplot(
    [echec['progression_s1_s2'], admis['progression_s1_s2']],
    patch_artist=True,
    labels=labels_box,
    medianprops=dict(color='white', linewidth=2.5)
)
for patch, color in zip(bp3['boxes'], [COLORS['echec'], COLORS['admis']]):
    patch.set_facecolor(color); patch.set_alpha(0.8)
ax3.set_title('Progression S1→S2\nvs Réussite', fontweight='bold')
ax3.set_ylabel('Δ Score (S2 - S1)')
ax3.axhline(0, color='grey', linestyle='--', alpha=0.6)
ax3.grid(axis='y', alpha=0.4)

ax4 = fig.add_subplot(gs[1, 0:2])
admis['score_scientifique'].plot.kde(ax=ax4, color=COLORS['admis'],
    linewidth=2.5, label='Admis')
echec['score_scientifique'].plot.kde(ax=ax4, color=COLORS['echec'],
    linewidth=2.5, label='Échec')
ax4.fill_between(
    np.linspace(df['score_scientifique'].min(), df['score_scientifique'].max(), 300),
    0, 0, alpha=0)
ax4.set_title('Densité KDE — Score Scientifique par Groupe', fontweight='bold')
ax4.set_xlabel('Score scientifique (/20)')
ax4.set_ylabel('Densité')
ax4.legend(fontsize=10)
ax4.grid(alpha=0.3)

ax5 = fig.add_subplot(gs[1, 2])
admis['taux_absenteisme'].plot.kde(ax=ax5, color=COLORS['admis'],
    linewidth=2.5, label='Admis')
echec['taux_absenteisme'].plot.kde(ax=ax5, color=COLORS['echec'],
    linewidth=2.5, label='Échec')
ax5.set_title('Densité KDE\nAbsentéisme', fontweight='bold')
ax5.set_xlabel('Taux (%)')
ax5.set_ylabel('Densité')
ax5.legend(fontsize=10)
ax5.grid(alpha=0.3)

ax6 = fig.add_subplot(gs[2, :])
corr_matrix = df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, ax=ax6,
    mask=mask,
    annot=True, fmt='.2f', annot_kws={'size': 8},
    cmap='RdYlGn', center=0, vmin=-1, vmax=1,
    linewidths=0.5, linecolor='white',
    cbar_kws={'shrink': 0.6, 'label': 'r de Pearson'}
)
ax6.set_title(
    'Matrice de Corrélation de Pearson (post-standardisation)\n'
    '→ Corrélations avec reussite_bac en dernière colonne/ligne',
    fontweight='bold', fontsize=11
)
ax6.tick_params(axis='x', rotation=30)
ax6.tick_params(axis='y', rotation=0)

plt.savefig('Pretraitement(analyse).png',
            dpi=160, bbox_inches='tight', facecolor=COLORS['bg'])
plt.close()
print("   ✔ Figure sauvegardée → Pretraitement(analyse).png")

print("\n🎯 [6] Corrélations Pearson avec reussite_bac (classées) :")
corr_target = df.corr()[target].drop(target).sort_values(key=abs, ascending=False)
for var, r in corr_target.items():
    bar = '█' * int(abs(r) * 20)
    signe = '+' if r > 0 else '-'
    print(f"   {signe}{abs(r):.3f}  {bar:<20}  {var}")

print("\n💾 [8] Export du dataset préprocessé...")

df_final = df.copy()


df_final.to_csv('Pretraitement_dataset.csv',         index=False)

print(f"   ✔ Pretraitement_dataset.csv       ({df_final.shape[0]} obs × {df_final.shape[1]} vars) — brut ingéniéré")

print("\n" + "=" * 70)
print("  SYNTHÈSE STATISTIQUE — VARIABLES OPTIMISABLES POUR LA RÉUSSITE")
print("=" * 70)

df_desc = df.groupby(target)[
    ['score_scientifique', 'taux_absenteisme',
     'progression_s1_s2', 'moyenne_generale_premiere',
     'math_moy', 'physique_moy', 'svt_moy', 'retard_scolaire']
].mean().T.rename(columns={0: 'Échec (moy)', 1: 'Admis (moy)'})

df_desc['Δ (Admis - Échec)'] = df_desc['Admis (moy)'] - df_desc['Échec (moy)']
df_desc['|Δ| relatif (%)']   = (df_desc['Δ (Admis - Échec)'] / df_desc['Échec (moy)'].abs() * 100).round(1)
df_desc = df_desc.sort_values('|Δ| relatif (%)', ascending=False, key=abs)
df_desc = df_desc.round(2)

print(df_desc.to_string())


print("\n✅ Données prêtes pour la régression logistique !\n")

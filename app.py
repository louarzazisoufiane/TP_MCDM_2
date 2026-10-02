"""Application Streamlit du devoir libre.

Lancement:
    pip install -r requirements.txt
    streamlit run app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from analyse_devoir_libre import (
    MAPS, fuzzy_topsis, gaussian_nb_predict, knn_predict, measures, minmax,
    multinomial_logistic_predict, split_stratified, standardize, topsis,
)

st.set_page_config(page_title="ABC multi-attributs", page_icon="📦", layout="wide")

REQUIRED = ["Risk", "Demand fluctuation", "Average stock", "Daily usage", "Unit cost",
            "Lead time", "Consignment stock", "Unit size"]


def prepare_data(source, global_weights):
    """Transforme, agrège et classe un inventaire, sans modifier la source."""
    df = source.copy()
    missing = set(REQUIRED).difference(df.columns)
    if missing:
        raise ValueError("Colonnes manquantes : " + ", ".join(sorted(missing)))
    if df[REQUIRED].isna().any().any():
        raise ValueError("Le fichier comporte des valeurs manquantes dans les colonnes requises.")
    numeric_columns = ["Average stock", "Daily usage", "Unit cost", "Lead time"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if df[numeric_columns].isna().any().any():
        raise ValueError("Les colonnes Average stock, Daily usage, Unit cost et Lead time doivent être numériques.")
    for col, mapping in MAPS.items():
        score_col = col.replace(" ", "_") + "_score"
        df[score_col] = df[col].map(mapping)
        if df[score_col].isna().any():
            invalid = sorted(df.loc[df[score_col].isna(), col].astype(str).unique())
            raise ValueError(f"Valeur non reconnue dans '{col}' : {', '.join(invalid)}")
    df["Daily_usage_norm"] = minmax(df["Daily usage"])
    df["Average_stock_urgency"] = 1 - minmax(df["Average stock"])
    df["Lead_time_norm"] = minmax(df["Lead time"])
    df["Unit_cost_norm"] = minmax(df["Unit cost"])
    df["Criticality"] = .78 * df["Risk_score"] + .22 * df["Demand_fluctuation_score"]
    df["Demand"] = .71 * df["Daily_usage_norm"] + .29 * df["Average_stock_urgency"]
    df["Supply"] = .75 * df["Lead_time_norm"] + .25 * df["Consignment_stock_score"]
    df["Cost"] = df["Unit_cost_norm"]
    df["Space"] = df["Unit_size_score"]
    criteria = ["Criticality", "Demand", "Supply", "Cost", "Space"]
    weight_vector = np.asarray([global_weights[x] for x in criteria])
    score, _, weighted = topsis(df[criteria].to_numpy(), weight_vector)
    df[["TOPSIS_" + c for c in criteria]] = weighted
    df["TOPSIS_score"] = score
    df["TOPSIS_rank"] = df["TOPSIS_score"].rank(method="first", ascending=False).astype(int)
    a_cut, b_cut = int(round(.2 * len(df))), int(round(.5 * len(df)))
    df["Classe"] = np.select([df.TOPSIS_rank <= a_cut, df.TOPSIS_rank <= b_cut], ["A", "B"], default="C")
    df["Fuzzy_TOPSIS_score"] = fuzzy_topsis(df[criteria].to_numpy(), weight_vector)
    df["Fuzzy_rank"] = df["Fuzzy_TOPSIS_score"].rank(method="first", ascending=False).astype(int)
    df["Fuzzy_Classe"] = np.select([df.Fuzzy_rank <= a_cut, df.Fuzzy_rank <= b_cut], ["A", "B"], default="C")
    return df, criteria


def train_models(df, test_fraction, seed, k):
    features = ["Risk_score", "Demand_fluctuation_score", "Average_stock_urgency",
                "Daily_usage_norm", "Unit_cost_norm", "Lead_time_norm",
                "Consignment_stock_score", "Unit_size_score"]
    x, y = df[features].to_numpy(), df.Classe.to_numpy()
    classes = np.array(["A", "B", "C"])
    tr, te = split_stratified(y, test_fraction=test_fraction, seed=seed)
    xtr, xte = standardize(x[tr], x[te])
    predicted = {
        "Régression logistique": multinomial_logistic_predict(xtr, y[tr], xte, classes),
        "Naive Bayes gaussien": gaussian_nb_predict(xtr, y[tr], xte, classes),
        f"k-NN (k={k})": knn_predict(xtr, y[tr], xte, classes, k=k),
    }
    metrics = pd.DataFrame([
        {"Modèle": name, "Accuracy": a, "Précision macro": p, "Rappel macro": r, "F1 macro": f}
        for name, pred in predicted.items() for a, p, r, f in [measures(y[te], pred, classes)]
    ])
    # Identifiants conservés afin de pouvoir contrôler chaque prédiction test dans l'interface.
    test_results = df.iloc[te][REQUIRED + ["Classe", "TOPSIS_score"]].copy()
    test_results.insert(0, "Ligne_source", te + 1)
    for name, pred in predicted.items():
        test_results["Prédit — " + name] = pred
    return metrics, test_results, y[te], predicted


@st.cache_data(show_spinner=False)
def default_data():
    return pd.read_csv(Path(__file__).parent / "inventory_data.csv")


st.title("📦 Analyse ABC multi-attributs")
st.caption("Classification d'inventaire par TOPSIS, ABC, apprentissage automatique et extension floue.")

with st.sidebar:
    st.header("Données et paramètres")
    uploaded = st.file_uploader("Importer un CSV", type="csv", help="Le fichier doit contenir les huit colonnes de l'énoncé.")
    st.markdown("**Poids des critères globaux**")
    raw_weights = {}
    for label in ["Criticality", "Demand", "Supply", "Cost", "Space"]:
        raw_weights[label] = st.number_input(label, min_value=0.0, value=1.0, step=.1)
    st.divider()
    st.markdown("**Validation des modèles**")
    test_fraction = st.slider("Part réservée au test", .15, .40, .20, .05)
    random_seed = st.number_input("Graine aléatoire", min_value=0, value=42, step=1)
    neighbors = st.slider("Nombre de voisins k", 3, 31, 11, 2)

try:
    data = pd.read_csv(uploaded) if uploaded else default_data()
except Exception as exc:
    st.error(f"Impossible de lire le fichier : {exc}")
    st.stop()

total = sum(raw_weights.values())
if total == 0:
    st.error("Au moins un poids doit être strictement positif.")
    st.stop()
weights = {name: value / total for name, value in raw_weights.items()}
try:
    result, criteria = prepare_data(data, weights)
except ValueError as exc:
    st.error(str(exc))
    st.info("Colonnes attendues : " + ", ".join(REQUIRED))
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["Vue d'ensemble", "TOPSIS & ABC", "Machine learning", "Méthode"])

with tab1:
    st.subheader("Inventaire chargé")
    st.caption(f"Source active : **{uploaded.name if uploaded else 'inventory_data.csv (fichier fourni)'}**. Tous les calculs affichés utilisent ce fichier.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Articles", len(result))
    c2.metric("Classe A", int((result.Classe == "A").sum()))
    c3.metric("Classe B", int((result.Classe == "B").sum()))
    c4.metric("Classe C", int((result.Classe == "C").sum()))
    st.dataframe(data, use_container_width=True, height=270)
    st.subheader("Poids effectivement utilisés")
    st.bar_chart(pd.DataFrame({"Poids": weights}))

with tab2:
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Classement TOPSIS")
        display = ["TOPSIS_rank", "TOPSIS_score", "Classe", "Fuzzy_TOPSIS_score", "Fuzzy_Classe"] + REQUIRED
        st.dataframe(result.sort_values("TOPSIS_rank")[display], use_container_width=True, height=440)
    with right:
        st.subheader("Répartition ABC")
        st.bar_chart(result.Classe.value_counts().reindex(["A", "B", "C"]))
        agreement = (result.Classe == result.Fuzzy_Classe).mean()
        st.metric("Accord TOPSIS / fuzzy", f"{agreement:.1%}")
        st.caption("Seuils : A = premiers 20 %, B = 30 % suivants, C = 50 % restants.")
    st.download_button("Télécharger la base enrichie", result.to_csv(index=False).encode("utf-8"),
                       "inventory_data_enriched.csv", "text/csv")

with tab3:
    st.subheader("Classification automatique des classes ABC")
    st.write("Les modèles sont entraînés maintenant sur l'inventaire actif; le sous-ensemble de test est écarté de l'apprentissage.")
    signature = (pd.util.hash_pandas_object(data, index=True).sum(), tuple(weights.values()),
                 test_fraction, int(random_seed), neighbors)
    if st.button("Entraîner et évaluer les modèles", type="primary"):
        with st.spinner("Entraînement sur les données importées et évaluation sur le jeu de test…"):
            st.session_state["ml_results"] = train_models(result, test_fraction, int(random_seed), neighbors)
            st.session_state["ml_signature"] = signature
    saved = st.session_state.get("ml_results")
    if saved is not None and st.session_state.get("ml_signature") == signature:
        metrics, test_results, actual, predictions = saved
        train_n = len(result) - len(test_results)
        st.caption(f"Apprentissage : {train_n} articles — test indépendant : {len(test_results)} articles. Graine : {int(random_seed)}.")
        st.dataframe(metrics.style.format({c: "{:.4f}" for c in metrics.columns[1:]}), use_container_width=True)
        chosen = st.selectbox("Modèle à inspecter", list(predictions), key="selected_model")
        confusion = pd.crosstab(pd.Series(actual, name="Classe réelle"),
                               pd.Series(predictions[chosen], name="Classe prédite"), dropna=False)
        confusion = confusion.reindex(index=["A", "B", "C"], columns=["A", "B", "C"], fill_value=0)
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.markdown("**Matrice de confusion**")
            st.dataframe(confusion, use_container_width=True)
        with col_b:
            st.markdown("**Prédictions du jeu de test**")
            st.dataframe(test_results, use_container_width=True, height=250)
        st.download_button("Télécharger les prédictions de test", test_results.to_csv(index=False).encode("utf-8"),
                           "predictions_test.csv", "text/csv")
        st.caption("Les huit attributs élémentaires encodés sont les entrées; les rangs et scores TOPSIS sont exclus afin d'éviter une fuite directe de la cible.")
    else:
        st.info("Cliquez sur « Entraîner et évaluer les modèles ». Toute modification du fichier, des poids ou des paramètres invalide les anciens résultats.")

with tab4:
    st.subheader("Règles de décision")
    st.markdown("""
Les variables qualitatives sont converties avec les scores normalisés de la table 1. Les critères agrégés sont :

* Criticality = 0,78 × Risk + 0,22 × Demand fluctuation ;
* Demand = 0,71 × Daily usage + 0,29 × urgence du stock moyen ;
* Supply = 0,75 × Lead time + 0,25 × Consignment stock.

Le stock moyen est inversé après normalisation min–max : moins de stock signifie davantage d'urgence. Coût et taille sont conservés comme critères propres. TOPSIS normalise ensuite la matrice, applique les poids choisis, et mesure la proximité de la solution idéale.

L'extension fuzzy TOPSIS remplace une note x par le nombre triangulaire `(max(0, x−0,10), x, min(1, x+0,10))`, afin de représenter l'incertitude des évaluations.
""")

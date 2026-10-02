"""Analyse ABC multi-attributs: TOPSIS, ABC, apprentissage et extension floue.

Exécution: python3 analyse_devoir_libre.py
Les fichiers produits sont inventory_data_enriched.csv, model_metrics.csv et rapport_devoir_libre.md.
Seules les bibliothèques Python usuelles (numpy/pandas) sont nécessaires.
"""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "inventory_data.csv"

# Table 1: les valeurs sont déjà normalisées par la somme des valeurs assignées.
MAPS = {
    "Risk": {"High": .47, "Normal": .35, "Low": .18},
    "Demand fluctuation": {"Increasing": .36, "Stable": .28, "Unknown": .20,
                           "Decreasing": .16, "Ending": .00},
    "Consignment stock": {"No": .80, "Yes": .20},
    "Unit size": {"Large": .53, "Medium": .31, "Small": .13},
}


def minmax(x):
    x = np.asarray(x, dtype=float)
    span = x.max() - x.min()
    # Un critère constant ne discrimine pas les articles; on lui attribue donc 0.
    return np.zeros_like(x) if span == 0 else (x - x.min()) / span


def topsis(matrix, weights):
    """TOPSIS pour des critères tous orientés 'plus élevé = plus prioritaire'."""
    normalized = matrix / np.sqrt((matrix ** 2).sum(axis=0))
    weighted = normalized * weights
    ideal_best, ideal_worst = weighted.max(axis=0), weighted.min(axis=0)
    d_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    d_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))
    return d_worst / (d_best + d_worst), normalized, weighted


def fuzzy_topsis(matrix, weights, spread=.10):
    """Extension triangulaire: chaque note x devient (x-spread, x, x+spread).

    Les idéaux sont les sommets observés; la distance est la distance vertex à 3 composantes.
    """
    lo = np.maximum(0, matrix - spread) * weights
    mid = matrix * weights
    hi = np.minimum(1, matrix + spread) * weights
    best = np.stack((lo.max(0), mid.max(0), hi.max(0)), axis=-1)
    worst = np.stack((lo.min(0), mid.min(0), hi.min(0)), axis=-1)
    tri = np.stack((lo, mid, hi), axis=-1)
    db = np.sqrt(((tri - best) ** 2).mean(axis=2).sum(axis=1))
    dw = np.sqrt(((tri - worst) ** 2).mean(axis=2).sum(axis=1))
    return dw / (db + dw)


def split_stratified(y, test_fraction=.20, seed=SEED):
    rng = np.random.default_rng(seed)
    train, test = [], []
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        n_test = round(len(idx) * test_fraction)
        test.extend(idx[:n_test]); train.extend(idx[n_test:])
    return np.asarray(train), np.asarray(test)


def standardize(train, test):
    mean, std = train.mean(0), train.std(0)
    return (train - mean) / np.where(std == 0, 1, std), (test - mean) / np.where(std == 0, 1, std)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def multinomial_logistic_predict(x_train, y_train, x_test, classes, lr=.08, epochs=3500, l2=.002):
    """Régression logistique multinomiale apprise par descente de gradient."""
    xx = np.c_[np.ones(len(x_train)), x_train]
    xt = np.c_[np.ones(len(x_test)), x_test]
    yy = np.eye(len(classes))[np.searchsorted(classes, y_train)]
    w = np.zeros((xx.shape[1], len(classes)))
    for _ in range(epochs):
        p = softmax(xx @ w)
        gradient = xx.T @ (p - yy) / len(xx)
        gradient[1:] += l2 * w[1:]
        w -= lr * gradient
    return classes[(xt @ w).argmax(1)]


def gaussian_nb_predict(x_train, y_train, x_test, classes):
    """Naive Bayes gaussien, avec lissage de variance."""
    logp = []
    for c in classes:
        z = x_train[y_train == c]
        mean, var = z.mean(0), z.var(0) + 1e-7
        logp.append(np.log(len(z) / len(x_train)) - .5 *
                    (np.log(2 * np.pi * var) + (x_test - mean) ** 2 / var).sum(1))
    return classes[np.argmax(np.column_stack(logp), axis=1)]


def knn_predict(x_train, y_train, x_test, classes, k=11):
    pred = []
    for row in x_test:
        near = y_train[np.argsort(((x_train - row) ** 2).sum(1))[:k]]
        pred.append(classes[np.argmax([(near == c).sum() for c in classes])])
    return np.asarray(pred)


def measures(actual, predicted, classes):
    accuracy = (actual == predicted).mean()
    scores = []
    for c in classes:
        tp = ((actual == c) & (predicted == c)).sum()
        fp = ((actual != c) & (predicted == c)).sum()
        fn = ((actual == c) & (predicted != c)).sum()
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        scores.append((precision, recall, f1))
    return accuracy, *np.mean(scores, axis=0)


def main():
    df = pd.read_csv(INPUT)
    # Scores qualitatifs demandés (Table 1).
    for column, mapping in MAPS.items():
        df[column.replace(" ", "_") + "_score"] = df[column].map(mapping)

    # Tous les critères élémentaires sont alignés: élevé = besoin de contrôle élevé.
    df["Daily_usage_norm"] = minmax(df["Daily usage"])
    df["Average_stock_urgency"] = 1 - minmax(df["Average stock"])
    df["Lead_time_norm"] = minmax(df["Lead time"])
    df["Unit_cost_norm"] = minmax(df["Unit cost"])

    # Table 2, puis deux critères physiques/économiques indépendants.
    df["Criticality"] = .78 * df["Risk_score"] + .22 * df["Demand_fluctuation_score"]
    df["Demand"] = .71 * df["Daily_usage_norm"] + .29 * df["Average_stock_urgency"]
    df["Supply"] = .75 * df["Lead_time_norm"] + .25 * df["Consignment_stock_score"]
    df["Cost"] = df["Unit_cost_norm"]
    df["Space"] = df["Unit_size_score"]
    criteria = ["Criticality", "Demand", "Supply", "Cost", "Space"]
    weights = np.full(5, .20)  # aucune pondération inter-critères n'est prescrite
    score, norm, weighted = topsis(df[criteria].to_numpy(), weights)
    df[["TOPSIS_" + c for c in criteria]] = weighted
    df["TOPSIS_score"] = score
    df["TOPSIS_rank"] = df["TOPSIS_score"].rank(method="first", ascending=False).astype(int)
    # Seuils Pareto en effectif: 20 %, 30 %, 50 % => 140, 210, 350 articles.
    df["Classe"] = np.select([df.TOPSIS_rank <= 140, df.TOPSIS_rank <= 350], ["A", "B"], default="C")
    fuzzy = fuzzy_topsis(df[criteria].to_numpy(), weights)
    df["Fuzzy_TOPSIS_score"] = fuzzy
    df["Fuzzy_rank"] = df["Fuzzy_TOPSIS_score"].rank(method="first", ascending=False).astype(int)
    df["Fuzzy_Classe"] = np.select([df.Fuzzy_rank <= 140, df.Fuzzy_rank <= 350], ["A", "B"], default="C")

    feature_cols = ["Risk_score", "Demand_fluctuation_score", "Average_stock_urgency",
                    "Daily_usage_norm", "Unit_cost_norm", "Lead_time_norm",
                    "Consignment_stock_score", "Unit_size_score"]
    x, y = df[feature_cols].to_numpy(), df["Classe"].to_numpy()
    classes = np.array(["A", "B", "C"])
    tr, te = split_stratified(y)
    xtr, xte = standardize(x[tr], x[te])
    predictions = {
        "Régression logistique multinomiale": multinomial_logistic_predict(xtr, y[tr], xte, classes),
        "Naive Bayes gaussien": gaussian_nb_predict(xtr, y[tr], xte, classes),
        "k-plus-proches-voisins (k=11)": knn_predict(xtr, y[tr], xte, classes),
    }
    metrics = pd.DataFrame(
        [{"Modèle": name, "Accuracy": a, "Précision_macro": p, "Rappel_macro": r, "F1_macro": f}
         for name, pred in predictions.items() for a, p, r, f in [measures(y[te], pred, classes)]])
    metrics.to_csv(ROOT / "model_metrics.csv", index=False, float_format="%.4f")
    df.to_csv(ROOT / "inventory_data_enriched.csv", index=False, float_format="%.6f")

    counts = df.Classe.value_counts().reindex(classes)
    agreement = (df.Classe == df.Fuzzy_Classe).mean()
    metrics_table = "| Modèle | Accuracy | Précision macro | Rappel macro | F1 macro |\n|---|---:|---:|---:|---:|\n" + "\n".join(
        f"| {row['Modèle']} | {row['Accuracy']:.4f} | {row['Précision_macro']:.4f} | {row['Rappel_macro']:.4f} | {row['F1_macro']:.4f} |"
        for _, row in metrics.iterrows())
    report = f"""# Devoir libre — analyse ABC multi-attributs

## 1. Données et variables

La base contient **{len(df)} articles** et aucune valeur manquante. Les variables qualitatives sont `Risk`, `Demand fluctuation`, `Consignment stock` et `Unit size`. Les variables quantitatives sont `Average stock`, `Daily usage`, `Unit cost` et `Lead time`.

## 2. Codage des variables qualitatives

Les scores normalisés de la table 1 ont été utilisés : risque High/Normal/Low = 0,47/0,35/0,18 ; fluctuation Increasing/Stable/Unknown/Decreasing/Ending = 0,36/0,28/0,20/0,16/0 ; consignation No/Yes = 0,80/0,20 ; taille Large/Medium/Small = 0,53/0,31/0,13. Cette étape rend des échelles hétérogènes comparables et permet les opérations de pondération et de distance de TOPSIS.

Pour les mesures quantitatives, une normalisation min–max est appliquée. `Average stock` est inversé : un stock faible accroît l'urgence. Les autres mesures sont orientées de sorte qu'une valeur élevée signifie une priorité élevée.

## 3. Critères agrégés

* Criticality = 0,78 × Risk + 0,22 × Demand fluctuation.
* Demand = 0,71 × Daily usage + 0,29 × Average-stock urgency.
* Supply = 0,75 × Lead time + 0,25 × Consignment score.
* Cost = Unit-cost normalized ; Space = Unit-size score.

Les trois premières formules reprennent exactement les poids internes de la table 2. Coût et encombrement sont conservés comme critères autonomes, afin de ne pas abandonner deux des huit attributs. Faute de pondérations entre les cinq critères globaux dans l'énoncé, chacun reçoit 0,20 : c'est une hypothèse neutre à valider avec les décideurs.

## 4–7. TOPSIS et ABC

La matrice décisionnelle contient les cinq critères précédents, tous à maximiser. TOPSIS effectue une normalisation vectorielle, applique le vecteur de poids (0,20, …, 0,20), puis calcule `C_i = D_i− / (D_i+ + D_i−)`. Les articles sont triés selon `TOPSIS_score` décroissant.

Les seuils ABC sont fondés sur les proportions de l'énoncé, par rang : A = rang 1–140 (20 %), B = 141–350 (30 %), C = 351–700 (50 %). Ainsi A contient les articles à suivre le plus étroitement, même lorsque la somme des scores ne suit pas exactement une loi 80/15/5. Résultat : A = {counts['A']}, B = {counts['B']}, C = {counts['C']}.

Le fichier `inventory_data_enriched.csv` ajoute les scores, critères, poids TOPSIS, rangs et `Classe` à la base originale.

## 8–9. Apprentissage automatique

Les labels sont `Classe`; les huit attributs élémentaires numérisés sont les variables explicatives. Le partage est stratifié 80/20 (560 apprentissage, 140 test, graine 42). Aucun score TOPSIS ni rang n'est introduit dans les prédicteurs, afin d'éviter une fuite directe de la cible.

{metrics_table}

Les indicateurs sont l'accuracy et les moyennes macro de précision, rappel et F1 : la moyenne macro accorde la même importance aux trois classes, malgré leur taille différente. Les résultats sont reproductibles dans `model_metrics.csv`.

## 10. Généralisation floue

Une version fuzzy TOPSIS est aussi calculée. Chaque note normalisée x devient le nombre triangulaire `(max(0,x−0,10), x, min(1,x+0,10))`; les cinq poids sont appliqués, puis les distances vertex aux idéaux flous observés donnent `Fuzzy_TOPSIS_score`. Les classes floues utilisent les mêmes seuils de rang. Elles coïncident avec les classes TOPSIS classiques pour **{agreement:.1%}** des articles. Cette formulation peut être enrichie en remplaçant la largeur fixe 0,10 par des jugements linguistiques d'experts (faible, moyen, élevé) et des poids flous.
"""
    (ROOT / "rapport_devoir_libre.md").write_text(report, encoding="utf-8")
    print("Fichiers créés:", "inventory_data_enriched.csv, model_metrics.csv, rapport_devoir_libre.md")
    print(metrics.to_string(index=False, float_format=lambda z: f"{z:.4f}"))
    print("Classes:", counts.to_dict(), "accord flou:", f"{agreement:.1%}")


if __name__ == "__main__":
    main()

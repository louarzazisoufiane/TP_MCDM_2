# Devoir libre — analyse ABC multi-attributs

## 1. Données et variables

La base contient **700 articles** et aucune valeur manquante. Les variables qualitatives sont `Risk`, `Demand fluctuation`, `Consignment stock` et `Unit size`. Les variables quantitatives sont `Average stock`, `Daily usage`, `Unit cost` et `Lead time`.

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

Les seuils ABC sont fondés sur les proportions de l'énoncé, par rang : A = rang 1–140 (20 %), B = 141–350 (30 %), C = 351–700 (50 %). Ainsi A contient les articles à suivre le plus étroitement, même lorsque la somme des scores ne suit pas exactement une loi 80/15/5. Résultat : A = 140, B = 210, C = 350.

Le fichier `inventory_data_enriched.csv` ajoute les scores, critères, poids TOPSIS, rangs et `Classe` à la base originale.

## 8–9. Apprentissage automatique

Les labels sont `Classe`; les huit attributs élémentaires numérisés sont les variables explicatives. Le partage est stratifié 80/20 (560 apprentissage, 140 test, graine 42). Aucun score TOPSIS ni rang n'est introduit dans les prédicteurs, afin d'éviter une fuite directe de la cible.

| Modèle | Accuracy | Précision macro | Rappel macro | F1 macro |
|---|---:|---:|---:|---:|
| Régression logistique multinomiale | 0.9571 | 0.9590 | 0.9397 | 0.9469 |
| Naive Bayes gaussien | 0.7500 | 0.7767 | 0.6929 | 0.7152 |
| k-plus-proches-voisins (k=11) | 0.7786 | 0.7558 | 0.7262 | 0.7373 |

Les indicateurs sont l'accuracy et les moyennes macro de précision, rappel et F1 : la moyenne macro accorde la même importance aux trois classes, malgré leur taille différente. Les résultats sont reproductibles dans `model_metrics.csv`.

## 10. Généralisation floue

Une version fuzzy TOPSIS est aussi calculée. Chaque note normalisée x devient le nombre triangulaire `(max(0,x−0,10), x, min(1,x+0,10))`; les cinq poids sont appliqués, puis les distances vertex aux idéaux flous observés donnent `Fuzzy_TOPSIS_score`. Les classes floues utilisent les mêmes seuils de rang. Elles coïncident avec les classes TOPSIS classiques pour **87.1%** des articles. Cette formulation peut être enrichie en remplaçant la largeur fixe 0,10 par des jugements linguistiques d'experts (faible, moyen, élevé) et des poids flous.

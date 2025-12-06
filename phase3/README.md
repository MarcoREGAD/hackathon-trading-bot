# Stratégie de Trading IA Légère - Phase 3

## 📋 Vue d'ensemble

Ce projet implémente une stratégie de trading basée sur un modèle d'IA ultra-léger pour le trading Forex. Le système utilise une régression logistique entraînée offline, dont les poids sont chargés en runtime sans dépendances lourdes.

### Architecture

```
Phase 3/
├── features.py              # Module de feature engineering
├── train_model.py           # Script d'entraînement (offline)
├── bot_trade.py            # Bot de trading (runtime ultra-léger)
├── forex_model_weights.json # Poids du modèle (généré)
├── main.py                 # Point d'entrée du hackathon
└── requirement.txt         # Dépendances
```

## 🎯 Objectifs

- **Offline (training)** : Utiliser scikit-learn pour entraîner un modèle
- **Runtime (bot)** : Ultra-léger, uniquement stdlib Python (json, math, statistics)
- **Prédiction** : Probabilité que le prochain mouvement soit haussier
- **Décision** : Convertir la probabilité en allocation de capital (Asset A, Asset B, Cash)

## 🚀 Installation et Setup

### 1. Installer les dépendances

```bash
cd phase3
pip install -r requirement.txt
```

Ou en utilisant le script de setup :

```bash
./setup_env.sh
```

### 2. Entraîner le modèle (offline)

```bash
python train_model.py data/asset_a_b_train.csv
```

Cela va :
- Charger les données historiques
- Calculer 5 features : `ret_1`, `ret_5`, `ret_10`, `vol_10`, `ma_ratio_5_20`
- Entraîner une régression logistique (split temporel 70/30)
- Exporter les poids dans `forex_model_weights.json`

**Sortie attendue :**
```
Chargement des données depuis data/asset_a_b_train.csv...
Nombre de lignes: 2522
Calcul des features...
Shape de X: (2502, 5)

Split temporel:
  Train: 1751 samples
  Test: 751 samples

Entraînement du modèle...
Accuracy Train: 0.5234
Accuracy Test: 0.5127

✅ Modèle exporté dans forex_model_weights.json
```

### 3. Exécuter le bot

```bash
./main.py data/asset_a_b_train.csv
```

Avec graphique :

```bash
./main.py data/asset_a_b_train.csv --show-graph
```

## 📊 Features Engineering

Le module `features.py` calcule 5 features simples et robustes :

| Feature | Description | Formule |
|---------|-------------|---------|
| `ret_1` | Retour sur 1 période | `close_t / close_{t-1} - 1` |
| `ret_5` | Retour sur 5 périodes | `close_t / close_{t-5} - 1` |
| `ret_10` | Retour sur 10 périodes | `close_t / close_{t-10} - 1` |
| `vol_10` | Volatilité locale | `std(returns_1bar)` sur 10 périodes |
| `ma_ratio_5_20` | Ratio moyennes mobiles | `MA5 / MA20 - 1` |

### Fenêtre requise

- **Minimum** : 20 bougies (WINDOW_MAX)
- **Recommandé** : 50+ bougies pour des features stables

## 🤖 Logique du Bot

### 1. Chargement du modèle (une seule fois)

```python
# Au démarrage, charge les poids depuis forex_model_weights.json
_COEF = [w1, w2, w3, w4, w5]
_INTERCEPT = b
```

### 2. Prédiction (à chaque tick)

```python
def predict_proba_up(features):
    z = intercept + sum(w_i * x_i)
    return sigmoid(z)  # P(hausse) entre 0 et 1
```

### 3. Conversion en position

```python
def proba_to_position(p_up):
    if p_up > 0.55:    # Signal haussier
        return +position (long Asset A)
    elif p_up < 0.45:  # Signal baissier
        return -position (réduire Asset A)
    else:
        return 0.0      # Neutre
```

### 4. Allocation du capital

- **P(hausse) > 0.55** : Augmenter Asset A (jusqu'à 60% max)
- **P(hausse) < 0.45** : Réduire Asset A, favoriser Asset B et Cash
- **0.45 ≤ P(hausse) ≤ 0.55** : Position neutre (1/3 chacun)

## ⚙️ Configuration

Dans `bot_trade.py`, vous pouvez ajuster :

```python
# Seuils de décision
UPPER_THRESHOLD = 0.55  # Seuil pour position longue
LOWER_THRESHOLD = 0.45  # Seuil pour position courte

# Taille maximale de position
MAX_POSITION_ASSET_A = 0.6  # 60% max sur Asset A

# Historique minimal requis
MIN_HISTORY_LENGTH = 20  # Nombre de bougies minimum
```

## 📈 Dépendances

### Offline (entraînement)
- pandas
- numpy
- scikit-learn

### Runtime (bot)
- **Aucune dépendance externe** (uniquement stdlib Python)
- json
- math
- statistics

## 🔍 Validation

### Test du modèle

```python
# Dans train_model.py, split temporel
Train: 70% des données historiques
Test: 30% des données les plus récentes

Métriques affichées :
- Accuracy (train et test)
- Classification report (précision, recall, F1-score)
```

### Test du bot

Le bot affiche des logs tous les 100 epochs :

```
Epoch 100: P(hausse)=0.523, Position=0.146, Asset A=0.407
Epoch 200: P(hausse)=0.487, Position=-0.051, Asset A=0.318
```

## 🎓 Algorithme Complet

```
OFFLINE (train_model.py):
1. Charger CSV historique
2. Créer label: y = 1 si close_{t+1} > close_t
3. Pour chaque fenêtre temporelle:
   - Calculer features [ret_1, ret_5, ret_10, vol_10, ma_ratio_5_20]
4. Split temporel (70% train, 30% test)
5. Entraîner LogisticRegression
6. Exporter coef + intercept → forex_model_weights.json

RUNTIME (bot_trade.py):
1. Charger poids depuis JSON (une fois)
2. À chaque tick:
   a. Mettre à jour historique
   b. Calculer features
   c. Prédire P(hausse) = sigmoid(w·x + b)
   d. Convertir en position
   e. Allouer capital entre Asset A, Asset B, Cash
3. Retourner décision
```

## 🛡️ Contraintes Respectées

✅ **Runtime ultra-léger** : Pas de scikit-learn en production  
✅ **Déterministe** : Pas de random, résultats reproductibles  
✅ **Pas de réseau** : Tout en local  
✅ **Pas d'I/O lourde** : Un seul fichier JSON chargé au démarrage  
✅ **Gestion mémoire** : Historique incrémental, pas de structure gigantesque  
✅ **Pas de look-ahead bias** : Features calculées sans regarder le futur

## 📝 Notes Importantes

1. **Split temporel obligatoire** : Ne jamais mélanger les données avec shuffle pour du trading
2. **Alignement features** : L'ordre des features doit être identique entre training et runtime
3. **Gestion des cas limites** : Le bot retourne une position neutre si pas assez d'historique
4. **Normalisation** : Les allocations sont toujours normalisées pour sommer à 1.0

## 🐛 Troubleshooting

### Erreur : "Fichier forex_model_weights.json non trouvé"
→ Exécutez d'abord `python train_model.py data/asset_a_b_train.csv`

### Erreur : "Nombre de features incorrect"
→ Vérifiez que `features.py` génère bien 5 features (même ordre qu'au training)

### Accuracy trop basse (<0.51)
→ Normal pour du trading, le marché est proche d'une marche aléatoire. Essayez d'ajouter plus de features ou d'ajuster les seuils.

## 📚 Ressources

- Documentation scikit-learn : https://scikit-learn.org/
- Régression logistique : https://en.wikipedia.org/wiki/Logistic_regression
- Technical indicators : https://www.investopedia.com/

---

**Bon trading ! 🚀📈**

# Grid Search - Guide d'utilisation

Guide rapide pour optimiser n'importe quelle stratégie de trading avec le grid search automatique.

## 🚀 Quick Start

```bash
# 1. Configurer votre stratégie dans .env
cp .env.example .env

# 2. Lancer le grid search
python3 grid_search.py data/asset_a_test.csv

# 3. Utiliser les meilleurs paramètres
cp .env.optimized .env
```

## 📋 Comment ça marche

### 1. **Définir les paramètres dans `.env.example`**

Le grid search détecte **automatiquement** tous les paramètres de votre stratégie :

```env
# .env.example
SHORT_WINDOW=40        # Détecté comme int
LONG_WINDOW=50         # Détecté comme int
TREND_GAIN=2.0         # Détecté comme float (à cause du .0)
SMOOTH_ALPHA=0.3       # Détecté comme float
DD_HARD=0.20           # Détecté comme float
```

### 2. **Utiliser les paramètres dans `bot_trade.py`**

```python
import os
from dotenv import load_dotenv

load_dotenv()

# Charger les paramètres depuis .env
SHORT_WINDOW = int(os.getenv('SHORT_WINDOW', 40))
LONG_WINDOW = int(os.getenv('LONG_WINDOW', 50))
TREND_GAIN = float(os.getenv('TREND_GAIN', 2.0))

def make_decision(epoch: int, price: float):
    # Utiliser SHORT_WINDOW, LONG_WINDOW, etc.
    pass
```

### 3. **Lancer le grid search**

```bash
python3 grid_search.py data/asset_a_test.csv
```

Le script va :
- ✅ Lire `.env.example` et détecter les paramètres
- ✅ Générer automatiquement des ranges intelligents
- ✅ Tester toutes les combinaisons en parallèle
- ✅ Créer `.env.optimized` avec les meilleurs paramètres

## ⚙️ Configuration personnalisée (optionnel)

Pour contrôler précisément les valeurs testées :

```bash
# Créer un fichier de config personnalisé
cp grid_search_config.py.example grid_search_config.py
```

Éditer `grid_search_config.py` :

```python
GRID_CONFIG = {
    'SHORT_WINDOW': [20, 30, 40, 50],           # Valeurs exactes à tester
    'LONG_WINDOW': [40, 50, 60, 70, 80],
    'TREND_GAIN': [1.0, 1.5, 2.0, 2.5, 3.0],
    'SMOOTH_ALPHA': [0.1, 0.2, 0.3, 0.4, 0.5],
    'DD_HARD': [0.10, 0.15, 0.20, 0.25],
}
```

## 📊 Résultats

Le grid search génère :
- `grid_search_results.csv` : Tous les résultats détaillés
- `.env.optimized` : Configuration optimale prête à l'emploi

### Exemple de sortie :

```
🏆 TOP 10 DES MEILLEURES COMBINAISONS

#1
  SHORT_WINDOW = 35
  LONG_WINDOW = 55
  TREND_GAIN = 2.5
  SMOOTH_ALPHA = 0.3
  DD_HARD = 0.2
  Sharpe Ratio: 1.8234
  Rendement cumulé: 45.23%
```

## 🎯 Workflow complet

```bash
# 1. Développer votre stratégie
vim bot_trade.py

# 2. Définir les paramètres configurables
vim .env.example

# 3. Tester la stratégie manuellement
python3 main.py data/asset_a_test.csv

# 4. Optimiser avec grid search
python3 grid_search.py data/asset_a_test.csv

# 5. Utiliser la config optimale
cp .env.optimized .env
python3 main.py data/asset_a_test.csv --show-graph
```

## 💡 Tips

- **Types importants** : Utilisez `.0` pour les floats (`2.0` pas `2`)
- **Noms clairs** : Utilisez des noms de variables en MAJUSCULES
- **Valeurs par défaut** : Mettez des valeurs raisonnables dans `.env.example`
- **Performance** : Le grid search utilise tous vos CPU cores automatiquement

## 🔧 Pour une nouvelle stratégie

1. ✏️ Écrivez votre algo dans `bot_trade.py`
2. 📝 Ajoutez les paramètres dans `.env.example`
3. 🚀 Lancez `python3 grid_search.py data/your_data.csv`
4. ✅ **C'est tout !** Le grid search s'adapte automatiquement

Aucune modification du code de grid search nécessaire ! 🎉

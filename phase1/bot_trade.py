# bot_trade.py

import os
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# ÉTAT GLOBAL DU BOT
_price_history = []
_current_weight = 0.5
_max_price_so_far = None

# HYPERPARAMÈTRES DE LA STRATÉGIE (chargés depuis .env)
DD_WINDOW = int(os.getenv('DD_WINDOW', 100))          # nombre d'epochs avant d'activer la détection de drawdown
DD_THRESHOLD = float(os.getenv('DD_THRESHOLD', 0.24)) # drawdown de 24% par rapport au plus haut -> régime "crash"
SAFE_WEIGHT = float(os.getenv('SAFE_WEIGHT', 0.25))   # allocation à Asset A en cas de crash
FULL_WEIGHT = float(os.getenv('FULL_WEIGHT', 1.0))    # allocation normale à Asset A
SMOOTHING = float(os.getenv('SMOOTHING', 0.5))        # vitesse de convergence vers la cible (0 = lent, 1 = instantané)


def _reset_state():
    """
    Réinitialise l'état interne du bot.
    Appelé automatiquement quand epoch == 0 (nouveau backtest / nouveau dataset).
    """
    global _price_history, _current_weight, _max_price_so_far
    _price_history = []
    _current_weight = 0.5
    _max_price_so_far = None


def make_decision(epoch: int, price: float):
    """
    Décide de l'allocation entre 'Asset A' et 'Cash'.

    Stratégie (long-only avec protection de krach)
    ---------------------------------------------
    - Par défaut, on reste pleinement investi dans Asset A (FULL_WEIGHT = 1.0),
      ce qui maximise le PnL quand le sous-jacent a un drift positif.
    - On surveille le plus haut prix observé jusqu'ici et on calcule le drawdown
      actuel par rapport à ce pic.
    - Une fois qu'on a suffisamment d'historique (DD_WINDOW points), si le
      drawdown dépasse DD_THRESHOLD (e.g. 24%), on réduit progressivement
      l'exposition vers SAFE_WEIGHT (par ex. 0.25).
    - En dehors de ces phases de krach, on remonte progressivement vers FULL_WEIGHT.

    Les changements de poids sont lissés via SMOOTHING pour limiter les frais.

    Parameters
    ----------
    epoch : int
        Index temporel courant.
    price : float
        Prix actuel de 'Asset A'.

    Returns
    -------
    dict
        {'Asset A': w_asset, 'Cash': w_cash} avec w_asset + w_cash == 1.0
    """
    global _price_history, _current_weight, _max_price_so_far

    # Nouveau run / nouveau dataset : on reset l'état
    if epoch == 0:
        _reset_state()

    # Mise à jour de l'historique de prix
    price = float(price)
    _price_history.append(price)

    # Mise à jour du plus haut historique
    if _max_price_so_far is None or price > _max_price_so_far:
        _max_price_so_far = price

    # Premier point : on reste neutre à 50/50 pour démarrer
    if len(_price_history) == 1:
        asset_weight = _current_weight
    else:
        # Tant qu'on n'a pas assez d'historique, on se comporte comme un buy & hold
        if len(_price_history) < DD_WINDOW or _max_price_so_far <= 0:
            target_weight = FULL_WEIGHT
        else:
            # Drawdown courant (ex: 0.2 = -20%)
            drawdown = 1.0 - price / _max_price_so_far

            # Détection de krach
            if drawdown >= DD_THRESHOLD:
                # Gros crash : on se met en mode "safe"
                target_weight = SAFE_WEIGHT
            else:
                # Régime normal : full Asset A
                target_weight = FULL_WEIGHT

        # Lissage vers la cible pour limiter le turnover et les frais
        _current_weight = _current_weight + SMOOTHING * (target_weight - _current_weight)

        # Sécurité numérique : on borne entre 0 et 1
        if _current_weight < 0.0:
            _current_weight = 0.0
        elif _current_weight > 1.0:
            _current_weight = 1.0

        asset_weight = float(_current_weight)

    cash_weight = 1.0 - asset_weight

    return {
        "Asset A": asset_weight,
        "Cash": cash_weight,
    }

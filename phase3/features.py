"""
Module de feature engineering pour la stratégie de trading Forex.
Calcule des features simples et robustes à partir de séries de prix de clôture.
"""

import math
from statistics import stdev


# Taille minimale de fenêtre requise pour calculer toutes les features
WINDOW_MAX = 20


def compute_features_from_close_series(closes: list[float]) -> list[float]:
    """
    Calcule un vecteur de features à partir d'une série de prix de clôture.
    
    Args:
        closes: Liste de prix de clôture ordonnée dans le temps (ancienne -> récente).
                Doit contenir au moins WINDOW_MAX éléments.
    
    Returns:
        Liste de features [ret_1, ret_5, ret_10, vol_10, ma_ratio_5_20]
    
    Features:
        - ret_1: Retour sur 1 période
        - ret_5: Retour sur 5 périodes
        - ret_10: Retour sur 10 périodes
        - vol_10: Volatilité (écart-type des retours 1-bar) sur 10 périodes
        - ma_ratio_5_20: Ratio (MA5 / MA20) - 1
    """
    if len(closes) < WINDOW_MAX:
        raise ValueError(f"La série doit contenir au moins {WINDOW_MAX} éléments, reçu {len(closes)}")
    
    # Prendre les dernières valeurs nécessaires
    current = closes[-1]
    
    # Retours simples
    ret_1 = (current / closes[-2] - 1) if len(closes) >= 2 else 0.0
    ret_5 = (current / closes[-6] - 1) if len(closes) >= 6 else 0.0
    ret_10 = (current / closes[-11] - 1) if len(closes) >= 11 else 0.0
    
    # Volatilité locale : écart-type des retours 1-bar sur les 10 dernières périodes
    if len(closes) >= 11:
        returns_1bar = []
        for i in range(-10, 0):
            ret = closes[i] / closes[i-1] - 1
            returns_1bar.append(ret)
        vol_10 = stdev(returns_1bar) if len(returns_1bar) > 1 else 0.0
    else:
        vol_10 = 0.0
    
    # Moyennes mobiles
    ma_5 = sum(closes[-5:]) / 5 if len(closes) >= 5 else current
    ma_20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else current
    
    # Ratio des moyennes mobiles
    ma_ratio_5_20 = (ma_5 / ma_20 - 1) if ma_20 != 0 else 0.0
    
    # Retour dans un ordre fixe
    features = [
        ret_1,
        ret_5,
        ret_10,
        vol_10,
        ma_ratio_5_20,
    ]
    
    return features


def compute_features_from_prices_dict(history: list[dict]) -> list[float]:
    """
    Helper pour extraire les closes d'un historique sous forme de dict et calculer les features.
    
    Args:
        history: Liste de dicts contenant au minimum une clé 'priceA' ou similaire.
    
    Returns:
        Liste de features calculées.
    """
    # Adapter selon la structure de données du projet
    # Ici on suppose que history contient des dicts avec 'priceA'
    closes = [h.get('priceA', 0.0) for h in history]
    
    if len(closes) < WINDOW_MAX:
        # Retourner des features neutres si pas assez de données
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    
    return compute_features_from_close_series(closes)

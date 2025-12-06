"""
Module de feature engineering pour la stratégie de trading Forex.
Calcule des features simples et efficaces à partir de séries de prix de clôture.
"""

import math
from statistics import stdev, mean


# Taille minimale de fenêtre requise pour calculer toutes les features
WINDOW_MAX = 30


def calculate_rsi(prices: list[float], period: int = 14) -> float:
    """Calcule le Relative Strength Index (RSI)."""
    if len(prices) < period + 1:
        return 50.0
    
    deltas = [prices[i] - prices[i-1] for i in range(-period, 0)]
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]
    
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_features_from_close_series(closes: list[float]) -> list[float]:
    """
    Calcule un vecteur de features optimisées à partir d'une série de prix.
    
    Args:
        closes: Liste de prix de clôture ordonnée dans le temps (ancienne -> récente).
    
    Returns:
        Liste de 10 features essentielles
    """
    if len(closes) < WINDOW_MAX:
        raise ValueError(f"La série doit contenir au moins {WINDOW_MAX} éléments, reçu {len(closes)}")
    
    current = closes[-1]
    
    # ==================== RETOURS (3) ====================
    ret_1 = (current / closes[-2] - 1) if len(closes) >= 2 else 0.0
    ret_5 = (current / closes[-6] - 1) if len(closes) >= 6 else 0.0
    ret_10 = (current / closes[-11] - 1) if len(closes) >= 11 else 0.0
    
    # ==================== VOLATILITÉ (2) ====================
    if len(closes) >= 11:
        returns_10 = [(closes[i] / closes[i-1] - 1) for i in range(-10, 0)]
        vol_10 = stdev(returns_10) if len(returns_10) > 1 else 0.0
    else:
        vol_10 = 0.0
    
    if len(closes) >= 21:
        returns_20 = [(closes[i] / closes[i-1] - 1) for i in range(-20, 0)]
        vol_20 = stdev(returns_20) if len(returns_20) > 1 else 0.0
    else:
        vol_20 = 0.0
    
    # ==================== MOYENNES MOBILES (3) ====================
    ma_5 = sum(closes[-5:]) / 5 if len(closes) >= 5 else current
    ma_10 = sum(closes[-10:]) / 10 if len(closes) >= 10 else current
    ma_20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else current
    
    ma_ratio_5_20 = (ma_5 / ma_20 - 1) if ma_20 != 0 else 0.0
    price_to_ma10 = (current / ma_10 - 1) if ma_10 != 0 else 0.0
    
    # ==================== RSI (1) ====================
    rsi_14 = calculate_rsi(closes, period=14)
    rsi_normalized = (rsi_14 - 50) / 50  # Normaliser entre -1 et 1
    
    # ==================== MOMENTUM (1) ====================
    momentum_10 = (current / closes[-11] - 1) if len(closes) >= 11 and closes[-11] != 0 else 0.0
    
    # Total: 10 features essentielles
    features = [
        ret_1,              # 0 - Retour court terme
        ret_5,              # 1 - Retour moyen terme
        ret_10,             # 2 - Retour long terme
        vol_10,             # 3 - Volatilité récente
        vol_20,             # 4 - Volatilité tendance
        ma_ratio_5_20,      # 5 - Tendance des moyennes
        price_to_ma10,      # 6 - Distance à la moyenne
        rsi_normalized,     # 7 - Force relative
        momentum_10,        # 8 - Momentum
        ret_1 * vol_10,     # 9 - Interaction retour/volatilité
    ]
    
    return features


def compute_features_from_prices_dict(history: list[dict]) -> list[float]:
    """Helper pour extraire les closes et calculer les features."""
    closes = [h.get('priceA', 0.0) for h in history]
    
    if len(closes) < WINDOW_MAX:
        return [0.0] * 10
    
    return compute_features_from_close_series(closes)

"""
Module de feature engineering pour la stratégie de trading Forex.
Calcule des features avancées à partir de séries de prix de clôture.
"""

import math
from statistics import stdev, mean


# Taille minimale de fenêtre requise pour calculer toutes les features
WINDOW_MAX = 50


def _rsi(prices: list[float], period: int = 14) -> float:
    """
    Calcule le Relative Strength Index (RSI).
    RSI = 100 - (100 / (1 + RS)), où RS = moyenne des gains / moyenne des pertes
    """
    if len(prices) < period + 1:
        return 50.0  # Neutre par défaut
    
    changes = [prices[i] - prices[i-1] for i in range(-period, 0)]
    gains = [max(c, 0) for c in changes]
    losses = [abs(min(c, 0)) for c in changes]
    
    avg_gain = mean(gains) if gains else 0.0
    avg_loss = mean(losses) if losses else 0.0
    
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _ema(prices: list[float], period: int) -> float:
    """Calcule l'Exponential Moving Average (EMA)."""
    if len(prices) < period:
        return prices[-1] if prices else 0.0
    
    multiplier = 2.0 / (period + 1)
    ema = sum(prices[-period:]) / period  # Démarrer avec SMA
    
    for price in prices[-period:]:
        ema = (price - ema) * multiplier + ema
    
    return ema


def _macd(prices: list[float]) -> tuple[float, float, float]:
    """
    Calcule le MACD (Moving Average Convergence Divergence).
    Returns: (macd_line, signal_line, histogram)
    """
    if len(prices) < 26:
        return 0.0, 0.0, 0.0
    
    ema_12 = _ema(prices, 12)
    ema_26 = _ema(prices, 26)
    macd_line = ema_12 - ema_26
    
    # Signal line (EMA 9 du MACD) - simplification
    signal_line = macd_line * 0.8  # Approximation
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def _bollinger_bands(prices: list[float], period: int = 20) -> tuple[float, float, float]:
    """
    Calcule les Bollinger Bands.
    Returns: (lower_band, middle_band, upper_band)
    """
    if len(prices) < period:
        mid = prices[-1] if prices else 0.0
        return mid, mid, mid
    
    recent = prices[-period:]
    middle_band = mean(recent)
    std = stdev(recent) if len(recent) > 1 else 0.0
    
    upper_band = middle_band + (2 * std)
    lower_band = middle_band - (2 * std)
    
    return lower_band, middle_band, upper_band


def _momentum(prices: list[float], period: int = 10) -> float:
    """Calcule le momentum (rate of change)."""
    if len(prices) < period + 1:
        return 0.0
    return (prices[-1] / prices[-(period+1)] - 1)


def _atr(prices: list[float], period: int = 14) -> float:
    """Calcule l'Average True Range (volatilité)."""
    if len(prices) < period + 1:
        return 0.0
    
    ranges = []
    for i in range(-period, 0):
        true_range = abs(prices[i] - prices[i-1])
        ranges.append(true_range)
    
    return mean(ranges) if ranges else 0.0


def compute_features_from_close_series(closes: list[float]) -> list[float]:
    """
    Calcule un vecteur de features avancées à partir d'une série de prix de clôture.
    
    Args:
        closes: Liste de prix de clôture ordonnée dans le temps (ancienne -> récente).
                Doit contenir au moins WINDOW_MAX éléments.
    
    Returns:
        Liste de features (25+ features ultra-optimisées)
    
    Features:
        - Retours multiples périodes: ret_1, ret_3, ret_5, ret_10, ret_20
        - Volatilités adaptatives: vol_5, vol_10, vol_20, atr_14, vol_ratio
        - Moyennes mobiles: ma_ratio_5_20, ma_ratio_10_30, ema_ratio_12_26
        - Indicateurs techniques: rsi_14, macd, macd_signal, macd_hist
        - Bollinger Bands: bb_position, bb_width
        - Momentum avancé: momentum_5, momentum_10, roc_14
        - Tendance: price_to_ma_20, price_to_ma_50, trend_slope, trend_strength
    """
    if len(closes) < WINDOW_MAX:
        raise ValueError(f"La série doit contenir au moins {WINDOW_MAX} éléments, reçu {len(closes)}")
    
    current = closes[-1]
    
    # === RETOURS MULTIPLES PÉRIODES ===
    ret_1 = (current / closes[-2] - 1) if len(closes) >= 2 else 0.0
    ret_3 = (current / closes[-4] - 1) if len(closes) >= 4 else 0.0
    ret_5 = (current / closes[-6] - 1) if len(closes) >= 6 else 0.0
    ret_10 = (current / closes[-11] - 1) if len(closes) >= 11 else 0.0
    ret_20 = (current / closes[-21] - 1) if len(closes) >= 21 else 0.0
    
    # === VOLATILITÉS ADAPTATIVES ===
    returns_5 = [closes[i] / closes[i-1] - 1 for i in range(-5, 0)] if len(closes) >= 6 else [0.0]
    vol_5 = stdev(returns_5) if len(returns_5) > 1 else 0.0
    
    returns_10 = [closes[i] / closes[i-1] - 1 for i in range(-10, 0)] if len(closes) >= 11 else [0.0]
    vol_10 = stdev(returns_10) if len(returns_10) > 1 else 0.0
    
    returns_20 = [closes[i] / closes[i-1] - 1 for i in range(-20, 0)] if len(closes) >= 21 else [0.0]
    vol_20 = stdev(returns_20) if len(returns_20) > 1 else 0.0
    
    atr_14 = _atr(closes, 14)
    
    # Ratio de volatilité court-terme / long-terme (détecte les changements de régime)
    vol_ratio = (vol_5 / vol_20) if vol_20 > 0 else 1.0
    
    # === MOYENNES MOBILES ===
    ma_5 = mean(closes[-5:]) if len(closes) >= 5 else current
    ma_10 = mean(closes[-10:]) if len(closes) >= 10 else current
    ma_20 = mean(closes[-20:]) if len(closes) >= 20 else current
    ma_30 = mean(closes[-30:]) if len(closes) >= 30 else current
    ma_50 = mean(closes[-50:]) if len(closes) >= 50 else current
    
    ma_ratio_5_20 = (ma_5 / ma_20 - 1) if ma_20 != 0 else 0.0
    ma_ratio_10_30 = (ma_10 / ma_30 - 1) if ma_30 != 0 else 0.0
    
    ema_12 = _ema(closes, 12)
    ema_26 = _ema(closes, 26)
    ema_ratio_12_26 = (ema_12 / ema_26 - 1) if ema_26 != 0 else 0.0
    
    price_to_ma_20 = (current / ma_20 - 1) if ma_20 != 0 else 0.0
    price_to_ma_50 = (current / ma_50 - 1) if ma_50 != 0 else 0.0
    
    # === INDICATEURS TECHNIQUES ===
    rsi_14 = (_rsi(closes, 14) - 50) / 50  # Normaliser entre -1 et 1
    
    macd_line, macd_signal, macd_hist = _macd(closes)
    # Normaliser MACD par le prix pour le rendre comparable
    macd_norm = macd_line / current if current != 0 else 0.0
    macd_signal_norm = macd_signal / current if current != 0 else 0.0
    macd_hist_norm = macd_hist / current if current != 0 else 0.0
    
    # === BOLLINGER BANDS ===
    bb_lower, bb_middle, bb_upper = _bollinger_bands(closes, 20)
    bb_width = ((bb_upper - bb_lower) / bb_middle) if bb_middle != 0 else 0.0
    # Position du prix dans les bandes (0 = bande basse, 1 = bande haute)
    bb_position = ((current - bb_lower) / (bb_upper - bb_lower)) if (bb_upper - bb_lower) != 0 else 0.5
    
    # === MOMENTUM AVANCÉ ===
    momentum_5 = _momentum(closes, 5)
    momentum_10 = _momentum(closes, 10)
    
    # Rate of Change sur 14 périodes
    roc_14 = (current / closes[-15] - 1) if len(closes) >= 15 else 0.0
    
    # === TENDANCE AVANCÉE (price action) ===
    # Calcul de la pente de régression linéaire sur les 20 dernières périodes
    if len(closes) >= 20:
        recent_20 = closes[-20:]
        x_vals = list(range(len(recent_20)))
        mean_x = mean(x_vals)
        mean_y = mean(recent_20)
        
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_vals, recent_20))
        denominator = sum((x - mean_x) ** 2 for x in x_vals)
        
        trend_slope = numerator / denominator if denominator != 0 else 0.0
        trend_slope_norm = trend_slope / mean_y if mean_y != 0 else 0.0
        
        # Force de la tendance (R²)
        ss_tot = sum((y - mean_y) ** 2 for y in recent_20)
        ss_res = sum((y - (mean_y + trend_slope * (x - mean_x))) ** 2 for x, y in zip(x_vals, recent_20))
        trend_strength = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    else:
        trend_slope_norm = 0.0
        trend_strength = 0.0
    
    # === FEATURES SUPPLÉMENTAIRES ===
    # Détection de breakout: prix actuel vs max/min récent
    max_20 = max(closes[-20:]) if len(closes) >= 20 else current
    min_20 = min(closes[-20:]) if len(closes) >= 20 else current
    price_range_position = ((current - min_20) / (max_20 - min_20)) if (max_20 - min_20) > 0 else 0.5
    
    # Retour dans un ordre fixe
    features = [
        ret_1,
        ret_3,
        ret_5,
        ret_10,
        ret_20,
        vol_5,
        vol_10,
        vol_20,
        atr_14,
        vol_ratio,
        ma_ratio_5_20,
        ma_ratio_10_30,
        ema_ratio_12_26,
        price_to_ma_20,
        price_to_ma_50,
        rsi_14,
        macd_norm,
        macd_signal_norm,
        macd_hist_norm,
        bb_width,
        bb_position,
        momentum_5,
        momentum_10,
        roc_14,
        trend_slope_norm,
        trend_strength,
        price_range_position,
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
        return [0.0] * 27  # 27 features au total maintenant
    
    return compute_features_from_close_series(closes)

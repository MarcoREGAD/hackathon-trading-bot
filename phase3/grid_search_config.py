"""
Configuration personnalisée pour le grid search du bot V46.

Pour chaque paramètre, définissez une liste de valeurs à tester.
"""

# Configuration pour mode QUICK (rapide) - ~200 combinaisons
GRID_CONFIG_QUICK = {
    'EMA_SHORT_PERIOD': [2, 3, 4],
    'EMA_LONG_PERIOD': [4, 5, 6],
    'RSI_PERIOD': [3, 4, 5],
    'MOMENTUM_BOOST_MULTIPLIER': [6.0, 7.5, 9.0],
    'KELLY_MAX_CAP': [0.95, 0.99],
}

# Configuration pour mode NORMAL (défaut) - ~1000 combinaisons
GRID_CONFIG_NORMAL = {
    'EMA_SHORT_PERIOD': [2, 3, 4, 5],
    'EMA_LONG_PERIOD': [4, 5, 6, 7],
    'EMA_EXIT_PERIOD': [2, 3, 4],
    'RSI_PERIOD': [3, 4, 5, 6],
    'RSI_OVERSOLD': [35, 40, 45],
    'RSI_OVERBOUGHT': [65, 70, 75],
    'MOMENTUM_BOOST_MULTIPLIER': [5.0, 6.5, 7.5, 9.0],
    'PYRAMIDING_MAX_MULT': [10.0, 12.0, 14.0],
    'KELLY_MAX_CAP': [0.95, 0.98, 0.99],
}

# Configuration pour mode FULL (exhaustif) - ~5000+ combinaisons
GRID_CONFIG_FULL = {
    'EMA_SHORT_PERIOD': [2, 3, 4, 5, 6],
    'EMA_LONG_PERIOD': [4, 5, 6, 7, 8, 10],
    'EMA_EXIT_PERIOD': [2, 3, 4, 5],
    'EMA_ULTRA_LONG_PERIOD': [5, 6, 7, 8],
    'VOLATILITY_PERIOD': [2, 3, 4],
    'RSI_PERIOD': [3, 4, 5, 6, 7],
    'RSI_OVERSOLD': [30, 35, 40, 45],
    'RSI_OVERBOUGHT': [60, 65, 70, 75],
    'MOMENTUM_BOOST_MULTIPLIER': [4.0, 5.5, 7.0, 8.5, 10.0],
    'PYRAMIDING_PROFIT_THRESHOLD': [0.002, 0.003, 0.004],
    'PYRAMIDING_MAX_MULT': [8.0, 10.0, 12.0, 14.0],
    'BREAKOUT_LOOKBACK': [6, 8, 10, 12],
    'BREAKOUT_BOOST': [6.0, 8.0, 10.0],
    'TREND_STRENGTH_BOOST': [7.0, 8.5, 10.0],
    'BASE_MAX_RISK_ALLOCATION': [0.95, 0.98, 0.99],
    'KELLY_MAX_CAP': [0.95, 0.97, 0.99],
}

# Configuration utilisée par défaut
GRID_CONFIG = GRID_CONFIG_QUICK


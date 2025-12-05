import numpy as np

# ==================== ÉTAT GLOBAL ====================
history = []                     # historique des prix
position = 0.5                   # position actuelle (0=0%, 1=100% en actif)
consecutive_losses = 0           # pertes consécutives (pour gestion de risque)
win_streak = 0                   # gains consécutifs (pour profit‑taking)
entry_price = None               # prix d'entrée de la position (pour stop‑loss/take‑profit)
atr_value = 0                    # Average True Range (volatilité)
adx_value = 0                    # Average Directional Index (force de la tendance)

# ==================== PARAMÈTRES AJUSTABLES ====================
# Pour rendre l'algo plus agressif : augmenter les positions max, réduire les seuils de volatilité, etc.
PARAMS = {
    # périodes des indicateurs
    'sma_fast': 10, 'sma_medium': 30, 'sma_slow': 50,
    'ema_fast': 12, 'ema_slow': 26, 'ema_signal': 9,
    'rsi_period': 14,
    'bb_period': 20, 'bb_std': 2,
    'adx_period': 14,
    'atr_period': 14,
    'stoch_k': 14, 'stoch_d': 3,
    'momentum_period': 10,
    
    # seuils de signal
    'rsi_oversold': 30, 'rsi_overbought': 70,
    'stoch_oversold': 20, 'stoch_overbought': 80,
    'adx_strong_trend': 25,
    'volatility_max': 20,          # volatilité max (en %) pour autoriser une entrée
    
    # gestion de position
    'max_position': 0.95,         # position maximale (95%)
    'min_position': 0.05,         # position minimale (5%)
    'neutral_position': 0.5,
    'position_step': 0.15,        # pas d'ajout de position
    
    # gestion des risques
    'stop_loss_atr': 2.0,         
    'take_profit_atr': 3.0,     
    'max_consecutive_losses': 3,  # après 3 pertes, on réduit l'exposition
    'volatility_cut': 0.7,        # réduction de position en cas de forte volatilité
}

# ==================== FONCTIONS DES INDICATEURS ====================
def calculate_sma(prices, period):
    """Moyenne mobile simple."""
    return np.mean(prices[-period:]) if len(prices) >= period else None

def calculate_ema(prices, period, prev_ema=None):
    """Moyenne mobile exponentielle (approximation)."""
    if len(prices) < period:
        return None
    if prev_ema is None:
        return np.mean(prices[-period:])
    alpha = 2 / (period + 1)
    return price * alpha + prev_ema * (1 - alpha)

def calculate_macd(prices):
    """MACD (line, signal, histogram)."""
    if len(prices) < PARAMS['ema_slow'] + PARAMS['ema_signal']:
        return 0, 0, 0
    # Approximation des EMA
    ema_fast = calculate_ema(prices, PARAMS['ema_fast'])
    ema_slow = calculate_ema(prices, PARAMS['ema_slow'])
    macd_line = ema_fast - ema_slow if ema_fast and ema_slow else 0
    # Signal line (EMA du MACD)
    macd_history = []
    for i in range(PARAMS['ema_signal']):
        start = -PARAMS['ema_signal'] - i
        end = start + PARAMS['ema_slow']
        if end <= 0:
            ema_f = calculate_ema(prices[start:end], PARAMS['ema_fast'])
            ema_s = calculate_ema(prices[start:end], PARAMS['ema_slow'])
            if ema_f and ema_s:
                macd_history.append(ema_f - ema_s)
    signal_line = np.mean(macd_history) if macd_history else macd_line
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_rsi(prices):
    """Relative Strength Index."""
    period = PARAMS['rsi_period']
    if len(prices) < period + 1:
        return 50
    deltas = np.diff(prices[-period-1:])
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(prices):
    """Bollinger Bands (upper, middle, lower)."""
    period = PARAMS['bb_period']
    if len(prices) < period:
        return prices[-1], prices[-1], prices[-1]
    ma = np.mean(prices[-period:])
    std = np.std(prices[-period:])
    upper = ma + (PARAMS['bb_std'] * std)
    lower = ma - (PARAMS['bb_std'] * std)
    return upper, ma, lower

def calculate_atr(prices):
    """Average True Range (volatilité)."""
    period = PARAMS['atr_period']
    if len(prices) < period + 1:
        return 0
    true_ranges = []
    for i in range(1, period + 1):
        high_low = abs(prices[-i] - prices[-i-1])
        true_ranges.append(high_low)
    return np.mean(true_ranges)

def calculate_adx(prices):
    """Average Directional Index (force de la tendance)."""
    period = PARAMS['adx_period']
    if len(prices) < period * 2:
        return 0
    # Approximation simplifiée : basée sur la magnitude des mouvements
    price_changes = []
    for i in range(1, period + 1):
        change = abs(prices[-i] - prices[-i-1])
        price_changes.append(change)
    avg_change = np.mean(price_changes)
    avg_price = np.mean(prices[-period:])
    if avg_price == 0:
        return 0
    # ADX simplifié (en %)
    adx = (avg_change / avg_price) * 100
    return adx

def calculate_stochastic(prices):
    """Stochastic Oscillator (%K, %D)."""
    k_period = PARAMS['stoch_k']
    d_period = PARAMS['stoch_d']
    
    # On a besoin d'au moins k_period + d_period - 1 prix pour calculer %D
    if len(prices) < k_period + d_period - 1:
        return 50, 50
    
    # Calcul du %K actuel
    recent_prices = prices[-k_period:]
    lowest = min(recent_prices)
    highest = max(recent_prices)
    if highest - lowest == 0:
        current_k = 50
    else:
        current_k = 100 * ((prices[-1] - lowest) / (highest - lowest))
    
    # Calcul des %K pour les d_period dernières périodes
    k_values = []
    for i in range(d_period):
        # Fenêtre pour le i-ème %K (0 = actuel, 1 = précédent, ...)
        start = -k_period - i
        end = -i if i != 0 else None
        window = prices[start:end]
        if len(window) < k_period:
            # Pas assez de données pour cette fenêtre, on utilise le %K actuel
            k_val = current_k
        else:
            low = min(window)
            high = max(window)
            if high - low == 0:
                k_val = 50
            else:
                k_val = 100 * ((window[-1] - low) / (high - low))
        k_values.append(k_val)
    
    # %D est la moyenne des d_period %K
    d = np.mean(k_values)
    
    return current_k, d

def calculate_momentum(prices):
    """Momentum sur une période donnée."""
    period = PARAMS['momentum_period']
    if len(prices) < period + 1:
        return 0
    return (prices[-1] / prices[-period] - 1) * 100

# ==================== FONCTION PRINCIPALE ====================
def make_decision(epoch: int, price: float):
    global history, position, consecutive_losses, win_streak, entry_price, atr_value, adx_value
    
    # 1. Mise à jour de l'historique
    history.append({"epoch": epoch, "price": price})
    prices = [h["price"] for h in history]
    
    # 2. Calcul de tous les indicateurs
    sma_fast = calculate_sma(prices, PARAMS['sma_fast'])
    sma_medium = calculate_sma(prices, PARAMS['sma_medium'])
    sma_slow = calculate_sma(prices, PARAMS['sma_slow'])
    macd_line, macd_signal, macd_hist = calculate_macd(prices)
    rsi = calculate_rsi(prices)
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(prices)
    atr_value = calculate_atr(prices)
    adx_value = calculate_adx(prices)
    stoch_k, stoch_d = calculate_stochastic(prices)
    momentum = calculate_momentum(prices)
    
    # 3. Génération des signaux individuels (vote pondéré)
    signals = []
    
    # SMA (tendance)
    if sma_fast and sma_medium and sma_slow:
        if sma_fast > sma_medium > sma_slow:
            signals.append(1)   # fort haussier
        elif sma_fast < sma_medium < sma_slow:
            signals.append(-1)  # fort baissier
        elif sma_fast > sma_medium:
            signals.append(0.5) # léger haussier
        elif sma_fast < sma_medium:
            signals.append(-0.5)# léger baissier
    
    # MACD
    if macd_line > macd_signal:
        signals.append(1)
    elif macd_line < macd_signal:
        signals.append(-1)
    
    # RSI
    if rsi < PARAMS['rsi_oversold']:
        signals.append(1)   # survendu → achat
    elif rsi > PARAMS['rsi_overbought']:
        signals.append(-1)  # suracheté → vente
    
    # Bollinger Bands
    if price <= bb_lower:
        signals.append(1)   # touche la bande inférieure → achat
    elif price >= bb_upper:
        signals.append(-1)  # touche la bande supérieure → vente
    
    # Stochastic
    if stoch_k < PARAMS['stoch_oversold']:
        signals.append(1)
    elif stoch_k > PARAMS['stoch_overbought']:
        signals.append(-1)
    
    # Momentum
    if momentum > 2:
        signals.append(0.5)
    elif momentum < -2:
        signals.append(-0.5)
    
    # ADX (force de la tendance)
    if adx_value > PARAMS['adx_strong_trend']:
        # en tendance forte, on donne plus de poids aux signaux de tendance
        signals.append(0.7 if sma_fast and sma_fast > sma_medium else 0)
    
    # 4. Score composite (moyenne des signaux)
    score = np.mean(signals) if signals else 0
    
    # 5. Gestion des risques avancée
    # A. Volatilité excessive : on réduit l'exposition
    volatility = (atr_value / price * 100) if price > 0 else 0
    if volatility > PARAMS['volatility_max']:
        position *= PARAMS['volatility_cut']
        score = 0 
    
    # B. Stop‑loss dynamique (basé sur l'ATR)
    if entry_price is not None:
        stop_level = entry_price - PARAMS['stop_loss_atr'] * atr_value
        if price < stop_level:
            consecutive_losses += 1
            win_streak = 0
            entry_price = None
            position = max(PARAMS['min_position'], position * 0.5)
    
    # C. Take‑profit dynamique (basé sur l'ATR)
    if entry_price is not None:
        profit_level = entry_price + PARAMS['take_profit_atr'] * atr_value
        if price > profit_level:
            win_streak += 1
            consecutive_losses = 0
            entry_price = None
            position = min(PARAMS['max_position'], position * 0.8)  # on lock une partie des gains
    
    # D. Réduction après plusieurs pertes consécutives
    if consecutive_losses >= PARAMS['max_consecutive_losses']:
        position = PARAMS['min_position']
        score = 0
    
    # 6. Décision d'allocation basée sur le score
    if score > 0.5:
        # Signal haussier fort
        new_position = position + PARAMS['position_step']
        if entry_price is None:
            entry_price = price
    elif score > 0.2:
        # Signal haussier modéré
        new_position = position + (PARAMS['position_step'] * 0.5)
    elif score < -0.5:
        # Signal baissier fort
        new_position = position - PARAMS['position_step']
        entry_price = None
    elif score < -0.2:
        # Signal baissier modéré
        new_position = position - (PARAMS['position_step'] * 0.5)
    else:
        # Neutre
        new_position = PARAMS['neutral_position']
    
    # 7. Application des limites de position
    new_position = max(PARAMS['min_position'], min(PARAMS['max_position'], new_position))
    
    # 8. Mise à jour de l'état global
    position = new_position
    
    # 9. Retour de l'allocation (arrondie à 3 décimales)
    return {
        'Asset B': round(position, 3),
        'Cash': round(1 - position, 3)
    }
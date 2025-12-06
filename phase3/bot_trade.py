import pandas as pd
import numpy as np
import math
import os
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# --- ⚙️ PARAMÈTRES V46 - Chargés depuis .env ---
EMA_SHORT_PERIOD = int(os.getenv('EMA_SHORT_PERIOD', 2))
EMA_LONG_PERIOD = int(os.getenv('EMA_LONG_PERIOD', 4))
EMA_EXIT_PERIOD = int(os.getenv('EMA_EXIT_PERIOD', 2))
EMA_ULTRA_LONG_PERIOD = int(os.getenv('EMA_ULTRA_LONG_PERIOD', 5))

VOLATILITY_PERIOD = int(os.getenv('VOLATILITY_PERIOD', 2))

TREND_CONFIRMATION_THRESHOLD = float(os.getenv('TREND_CONFIRMATION_THRESHOLD', 0.000001))
MOMENTUM_SLOWDOWN_THRESHOLD = float(os.getenv('MOMENTUM_SLOWDOWN_THRESHOLD', 0.000001))

VOLATILITY_CUTOFF = float(os.getenv('VOLATILITY_CUTOFF', 1.00))

# Kelly Criterion V46
BASE_MAX_RISK_ALLOCATION = float(os.getenv('BASE_MAX_RISK_ALLOCATION', 0.99))
KELLY_MAX_CAP = float(os.getenv('KELLY_MAX_CAP', 0.99))
KELLY_MIN_TRADES = int(os.getenv('KELLY_MIN_TRADES', 1))

# Paramètres V46 ULTRA
RSI_PERIOD = int(os.getenv('RSI_PERIOD', 4))
RSI_OVERSOLD = int(os.getenv('RSI_OVERSOLD', 40))
RSI_OVERBOUGHT = int(os.getenv('RSI_OVERBOUGHT', 70))
MOMENTUM_BOOST_MULTIPLIER = float(os.getenv('MOMENTUM_BOOST_MULTIPLIER', 7.5))
PYRAMIDING_PROFIT_THRESHOLD = float(os.getenv('PYRAMIDING_PROFIT_THRESHOLD', 0.003))
PYRAMIDING_MAX_MULT = float(os.getenv('PYRAMIDING_MAX_MULT', 12.0))

# Breakout et trend V46
BREAKOUT_LOOKBACK = int(os.getenv('BREAKOUT_LOOKBACK', 8))
BREAKOUT_BOOST = float(os.getenv('BREAKOUT_BOOST', 8.0))
TREND_STRENGTH_BOOST = float(os.getenv('TREND_STRENGTH_BOOST', 8.5))

# --- Fonctions Utilitaires ---

def calculate_emas(prices, short_period, long_period, exit_period, ultra_long_period):
    """Calcule les Moyennes Mobiles Exponentielles (EMA)."""
    if len(prices) < ultra_long_period:
        return None, None, None, None
    series = pd.Series(prices)
    ema_short = series.ewm(span=short_period, adjust=False).mean().iloc[-1]
    ema_long = series.ewm(span=long_period, adjust=False).mean().iloc[-1]
    ema_ultra_long = series.ewm(span=ultra_long_period, adjust=False).mean().iloc[-1]
    
    if len(prices) >= exit_period:
        ema_exit = series.ewm(span=exit_period, adjust=False).mean().iloc[-1]
    else:
        ema_exit = None
        
    return ema_short, ema_long, ema_exit, ema_ultra_long

def calculate_volatility(prices, period):
    """Calcule la volatilité (écart-type des rendements)."""
    if len(prices) < period:
        return 1e-6 
    recent_prices = np.array(prices[-period:])
    returns = np.log(recent_prices[1:] / recent_prices[:-1])
    return pd.Series(returns).std()

def calculate_rsi(prices, period=5):
    """Calcule le RSI hyper-rapide."""
    if len(prices) < period + 1:
        return 50
    deltas = pd.Series(prices).diff()
    gains = deltas.where(deltas > 0, 0.0)
    losses = -deltas.where(deltas < 0, 0.0)
    avg_gain = gains.rolling(window=period).mean().iloc[-1]
    avg_loss = losses.rolling(window=period).mean().iloc[-1]
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_momentum_accel(prices, short, long):
    """Calcule l'accélération du momentum."""
    if len(prices) < long * 2:
        return 0
    s = pd.Series(prices)
    ema_s = s.ewm(span=short, adjust=False).mean()
    ema_l = s.ewm(span=long, adjust=False).mean()
    mom = (ema_s - ema_l) / ema_l
    if len(mom) < 5:
        return 0
    return mom.diff().iloc[-1]

def detect_breakout(prices, lookback=8):
    """Détecte si le prix actuel casse un nouveau high."""
    if len(prices) < lookback + 1:
        return False
    recent_high = max(prices[-lookback-1:-1])
    return prices[-1] > recent_high * 1.002  # +0.2% au-dessus

def calculate_trend_strength(prices, short, long):
    """Mesure la force de la tendance (0 à 1)."""
    if len(prices) < long:
        return 0
    s = pd.Series(prices)
    ema_s = s.ewm(span=short, adjust=False).mean().iloc[-1]
    ema_l = s.ewm(span=long, adjust=False).mean().iloc[-1]
    spread = abs((ema_s - ema_l) / ema_l)
    return min(spread / 0.01, 1.0)  # Normalise à 1.0 max

def calculate_kelly_f(stats):
    """Calcule le Kelly Optimal F ABSOLU V42 TOUT OU RIEN."""
    total_trades = stats['wins'] + stats['losses']
    
    if total_trades < KELLY_MIN_TRADES or stats['losses'] == 0:
        return 0.85  # START 85%
        
    p = stats['wins'] / total_trades
    
    avg_gain = stats['sum_gains'] / stats['wins'] if stats['wins'] > 0 else 0
    avg_loss = stats['sum_losses'] / stats['losses'] if stats['losses'] > 0 else 1e-10
    
    b = avg_gain / avg_loss if avg_loss > 0 else 2.0
    
    if b == 0:
        b = 0.01
    
    f = (p * (b + 1) - 1) / b
    
    # Bonus V46 ULTRA
    if p > 0.55:
        f *= 7.0
    elif p > 0.48:
        f *= 5.5
    elif p > 0.40:
        f *= 4.0
    
    if b > 1.7:
        f *= 6.0
    elif b > 1.3:
        f *= 4.0
    elif b > 0.9:
        f *= 2.8
    
    # Bonus série de gains
    if stats['wins'] > stats['losses'] * 1.2:
        f *= 3.5
        f *= 2.2
    
    # PAS de pénalité - on prend TOUS les risques
    
    return min(max(0.88, f), KELLY_MAX_CAP)

# --- Fonction de Décision Principale (V33 Corrigée) ---

def make_decision(epoch: int, price_A: float, price_B: float):
    global history
    
    allocation_A = 0.0
    allocation_B = 0.0
    
    # Initialisation de la structure de suivi Kelly et NAV
    if epoch == 0:
        history = {
            'prices_A': [price_A], 'prices_B': [price_B],
            'allocation': {'Asset A': 0.0, 'Asset B': 0.0, 'Cash': 1.0, 'NAV': 1.0},
            'kelly_stats': {'wins': 0, 'losses': 0, 'sum_gains': 0.0, 'sum_losses': 0.0, 'last_entry_price': {'A': 0.0, 'B': 0.0}}
        }
        # Ne retourne que les clés requises
        return {'Asset A': 0.0, 'Asset B': 0.0, 'Cash': 1.0} 

    # --- 1. Calcul du PnL et Mise à Jour Kelly Stats ---
    
    last_allocation = history['allocation']
    
    # Calcul de la NAV courante
    last_nav = last_allocation.get('NAV', 1.0)
    current_nav = last_nav * (1 + last_allocation['Asset A'] * (price_A/history['prices_A'][-1] - 1) + last_allocation['Asset B'] * (price_B/history['prices_B'][-1] - 1))
    
    # Mise à jour de l'historique des prix
    history['prices_A'].append(price_A)
    history['prices_B'].append(price_B)
    prices_A = history['prices_A']
    prices_B = history['prices_B']
    
    # Variables d'entrée/sortie
    entry_A = history['kelly_stats']['last_entry_price']['A']
    entry_B = history['kelly_stats']['last_entry_price']['B']
    
    # Détermination des triggers de sortie pour la mise à jour des statistiques
    min_required_points = max(EMA_LONG_PERIOD, VOLATILITY_PERIOD, EMA_EXIT_PERIOD, EMA_ULTRA_LONG_PERIOD)
    if len(prices_A) < min_required_points:
        return {'Asset A': 0.0, 'Asset B': 0.0, 'Cash': 1.0}
    
    ema_s_A, ema_l_A, ema_exit_A, ema_ul_A = calculate_emas(prices_A, EMA_SHORT_PERIOD, EMA_LONG_PERIOD, EMA_EXIT_PERIOD, EMA_ULTRA_LONG_PERIOD)
    ema_s_B, ema_l_B, ema_exit_B, ema_ul_B = calculate_emas(prices_B, EMA_SHORT_PERIOD, EMA_LONG_PERIOD, EMA_EXIT_PERIOD, EMA_ULTRA_LONG_PERIOD)
    
    exit_A_triggered = (last_allocation['Asset A'] > 0 and (ema_exit_A < ema_s_A or price_A < ema_l_A or ema_s_A < ema_l_A))
    exit_B_triggered = (last_allocation['Asset B'] > 0 and (ema_exit_B < ema_s_B or price_B < ema_l_B or ema_s_B < ema_l_B))
    
    # Mise à jour des statistiques Kelly si sortie
    if exit_A_triggered and entry_A > 0:
        pnl = (price_A / entry_A) - 1.0
        if pnl > 0:
            history['kelly_stats']['wins'] += 1
            history['kelly_stats']['sum_gains'] += pnl
        else:
            history['kelly_stats']['losses'] += 1
            history['kelly_stats']['sum_losses'] += abs(pnl)
        history['kelly_stats']['last_entry_price']['A'] = 0.0

    if exit_B_triggered and entry_B > 0:
        pnl = (price_B / entry_B) - 1.0
        if pnl > 0:
            history['kelly_stats']['wins'] += 1
            history['kelly_stats']['sum_gains'] += pnl
        else:
            history['kelly_stats']['losses'] += 1
            history['kelly_stats']['sum_losses'] += abs(pnl)
        history['kelly_stats']['last_entry_price']['B'] = 0.0
    
    # --- 2. Calcul des Indicateurs et du Kelly Factor ---
    
    volatility_A = calculate_volatility(prices_A, VOLATILITY_PERIOD)
    MoM_A_abs = (ema_s_A - ema_l_A) / ema_l_A
    
    volatility_B = calculate_volatility(prices_B, VOLATILITY_PERIOD)
    MoM_B_abs = (ema_s_B - ema_l_B) / ema_l_B
    
    # Indicateurs supplémentaires
    rsi_A = calculate_rsi(prices_A, RSI_PERIOD)
    rsi_B = calculate_rsi(prices_B, RSI_PERIOD)
    
    accel_A = calculate_momentum_accel(prices_A, EMA_SHORT_PERIOD, EMA_LONG_PERIOD)
    accel_B = calculate_momentum_accel(prices_B, EMA_SHORT_PERIOD, EMA_LONG_PERIOD)
    
    breakout_A = detect_breakout(prices_A, BREAKOUT_LOOKBACK)
    breakout_B = detect_breakout(prices_B, BREAKOUT_LOOKBACK)
    
    trend_strength_A = calculate_trend_strength(prices_A, EMA_SHORT_PERIOD, EMA_LONG_PERIOD)
    trend_strength_B = calculate_trend_strength(prices_B, EMA_SHORT_PERIOD, EMA_LONG_PERIOD)
    
    kelly_f = calculate_kelly_f(history['kelly_stats'])
    TARGET_ALLOCATION_KELLY = min(kelly_f, BASE_MAX_RISK_ALLOCATION) 
    
    # --- 3. Logique d'Entrée et Allocation ---
    
    relative_momentum_A_vs_B = ema_l_A > ema_l_B 
    relative_momentum_B_vs_A = ema_l_B > ema_l_A
    
    avg_market_volatility = (volatility_A + volatility_B) / 2.0
    market_in_stress = avg_market_volatility > VOLATILITY_CUTOFF
    
    # Critères ABSOLUMENT ÉLARGIS (quasi-toujours vrai)
    A_is_leader = (
        MoM_A_abs > TREND_CONFIRMATION_THRESHOLD and 
        relative_momentum_A_vs_B and 
        (price_A > ema_ul_A or rsi_A < 65 or breakout_A)  # RSI 65 = toujours
    )
    B_is_leader = (
        MoM_B_abs > TREND_CONFIRMATION_THRESHOLD and 
        relative_momentum_B_vs_A and
        (price_B > ema_ul_B or rsi_B < 65 or breakout_B)  # RSI 65 = toujours
    )
    
    
    if exit_A_triggered or exit_B_triggered or market_in_stress:
        allocation_A = 0.0
        allocation_B = 0.0
        
    elif A_is_leader or B_is_leader:
        
        leader = None
        
        if A_is_leader and B_is_leader:
            score_A = MoM_A_abs + abs(accel_A) * 20 + (0.01 if breakout_A else 0) + trend_strength_A * 0.005
            score_B = MoM_B_abs + abs(accel_B) * 20 + (0.01 if breakout_B else 0) + trend_strength_B * 0.005
            leader = 'A' if score_A > score_B else 'B'
        elif A_is_leader: leader = 'A'
        elif B_is_leader: leader = 'B'

        # --- ALLOCATION V45 NO CAP (A) ---
        if leader == 'A':
            MAX_ALLOCATION_ADJUSTED = min(
                BASE_MAX_RISK_ALLOCATION,
                TARGET_ALLOCATION_KELLY / max(volatility_A, 0.0002)
            )
            
            # BOOST momentum V46
            if MoM_A_abs > TREND_CONFIRMATION_THRESHOLD * 20:
                MAX_ALLOCATION_ADJUSTED = min(0.99, MAX_ALLOCATION_ADJUSTED * 6.5)
            elif MoM_A_abs > TREND_CONFIRMATION_THRESHOLD * 10:
                MAX_ALLOCATION_ADJUSTED = min(0.99, MAX_ALLOCATION_ADJUSTED * 4.5)
            elif MoM_A_abs > TREND_CONFIRMATION_THRESHOLD * 3:
                MAX_ALLOCATION_ADJUSTED = min(0.99, MAX_ALLOCATION_ADJUSTED * 3.0)
            
            # BOOST accélération V46
            if accel_A > 0.00015:
                MAX_ALLOCATION_ADJUSTED = min(0.99, MAX_ALLOCATION_ADJUSTED * MOMENTUM_BOOST_MULTIPLIER)
            elif accel_A > 0.00005:
                MAX_ALLOCATION_ADJUSTED = min(0.99, MAX_ALLOCATION_ADJUSTED * 4.5)
            
            # BOOST RSI V46
            if rsi_A < 70:
                MAX_ALLOCATION_ADJUSTED = min(0.99, MAX_ALLOCATION_ADJUSTED * 6.0)
            elif rsi_A < RSI_OVERSOLD:
                MAX_ALLOCATION_ADJUSTED = min(0.99, MAX_ALLOCATION_ADJUSTED * 4.0)
            
            # BOOST breakout V46
            if breakout_A:
                MAX_ALLOCATION_ADJUSTED = min(0.99, MAX_ALLOCATION_ADJUSTED * BREAKOUT_BOOST)
            
            # BOOST trend strength V46
            if trend_strength_A > 0.50:
                MAX_ALLOCATION_ADJUSTED = min(0.99, MAX_ALLOCATION_ADJUSTED * TREND_STRENGTH_BOOST)
            elif trend_strength_A > 0.30:
                MAX_ALLOCATION_ADJUSTED = min(0.99, MAX_ALLOCATION_ADJUSTED * 4.5)
            
            # Pyramiding x15 V46
            if entry_A > 0 and price_A > entry_A * (1 + PYRAMIDING_PROFIT_THRESHOLD):
                profit_mult = min((price_A / entry_A - 1) / PYRAMIDING_PROFIT_THRESHOLD, 15.0)
                MAX_ALLOCATION_ADJUSTED = min(0.99, MAX_ALLOCATION_ADJUSTED * (2.5 + profit_mult * 1.2))
            
            # Pas de réduction - garder momentum
            # if MoM_A_abs < MOMENTUM_SLOWDOWN_THRESHOLD * 0.1 and accel_A < -0.00015:
            #     MAX_ALLOCATION_ADJUSTED *= 0.90
            
            allocation_A = MAX_ALLOCATION_ADJUSTED
            if history['kelly_stats']['last_entry_price']['A'] == 0.0:
                history['kelly_stats']['last_entry_price']['A'] = price_A

        elif leader == 'B':
            MAX_ALLOCATION_ADJUSTED = min(
                BASE_MAX_RISK_ALLOCATION,
                TARGET_ALLOCATION_KELLY / max(volatility_B, 0.0005)
            )
            
            # BOOST momentum V46 (B)
            if MoM_B_abs > TREND_CONFIRMATION_THRESHOLD * 20:
                MAX_ALLOCATION_ADJUSTED = min(0.99, MAX_ALLOCATION_ADJUSTED * 6.5)
            elif MoM_B_abs > TREND_CONFIRMATION_THRESHOLD * 10:
                MAX_ALLOCATION_ADJUSTED = min(0.99, MAX_ALLOCATION_ADJUSTED * 4.5)
            elif MoM_B_abs > TREND_CONFIRMATION_THRESHOLD * 3:
                MAX_ALLOCATION_ADJUSTED = min(0.99, MAX_ALLOCATION_ADJUSTED * 3.0)
            
            # BOOST accélération V46 (B)
            if accel_B > 0.00015:
                MAX_ALLOCATION_ADJUSTED = min(0.99, MAX_ALLOCATION_ADJUSTED * MOMENTUM_BOOST_MULTIPLIER)
            elif accel_B > 0.00005:
                MAX_ALLOCATION_ADJUSTED = min(0.99, MAX_ALLOCATION_ADJUSTED * 4.5)
            
            # BOOST RSI V46 (B)
            if rsi_B < 70:
                MAX_ALLOCATION_ADJUSTED = min(0.99, MAX_ALLOCATION_ADJUSTED * 6.0)
            elif rsi_B < RSI_OVERSOLD:
                MAX_ALLOCATION_ADJUSTED = min(0.99, MAX_ALLOCATION_ADJUSTED * 4.0)
            
            # BOOST breakout V46 (B)
            if breakout_B:
                MAX_ALLOCATION_ADJUSTED = min(0.99, MAX_ALLOCATION_ADJUSTED * BREAKOUT_BOOST)
            
            # BOOST trend strength V46 (B)
            if trend_strength_B > 0.50:
                MAX_ALLOCATION_ADJUSTED = min(0.99, MAX_ALLOCATION_ADJUSTED * TREND_STRENGTH_BOOST)
            elif trend_strength_B > 0.30:
                MAX_ALLOCATION_ADJUSTED = min(0.99, MAX_ALLOCATION_ADJUSTED * 4.5)
            
            # Pyramiding x15 V46 (B)
            if entry_B > 0 and price_B > entry_B * (1 + PYRAMIDING_PROFIT_THRESHOLD):
                profit_mult = min((price_B / entry_B - 1) / PYRAMIDING_PROFIT_THRESHOLD, 15.0)
                MAX_ALLOCATION_ADJUSTED = min(0.99, MAX_ALLOCATION_ADJUSTED * (2.5 + profit_mult * 1.2))
            
            # Pas de réduction - garder momentum (B)
            # if MoM_B_abs < MOMENTUM_SLOWDOWN_THRESHOLD * 0.1 and accel_B < -0.00015:
            #     MAX_ALLOCATION_ADJUSTED *= 0.90
                
            allocation_B = MAX_ALLOCATION_ADJUSTED
            if history['kelly_stats']['last_entry_price']['B'] == 0.0:
                history['kelly_stats']['last_entry_price']['B'] = price_B
    
    
    # --- 4. Finalisation et Stockage de la NAV (CORRECTION) ---

    allocation_Cash = 1.0 - allocation_A - allocation_B
    allocation_Cash = max(0.0, allocation_Cash) 

    final_allocation = {
        'Asset A': allocation_A, 
        'Asset B': allocation_B, 
        'Cash': allocation_Cash,
    }

    # Stockage de la NAV et de l'allocation pour le prochain cycle DANS HISTORY
    history['allocation']['NAV'] = current_nav
    history['allocation']['Asset A'] = allocation_A
    history['allocation']['Asset B'] = allocation_B
    history['allocation']['Cash'] = allocation_Cash
    
    return final_allocation

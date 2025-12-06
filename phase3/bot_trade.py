"""  
HARMONIC MEAN ALGORITHM V11 - ROBUSTESSE MAXIMALE
==================================================
Stratégie: Trading sur Asset B avec Harmonic Mean + Harmonic Pulse + Robustness
- Window=4 pour réactivité optimale
- Entry=1.00300, Exit=0.99780 (robustness-optimized)
- Trailing Stop=0.989 (protection serrée)
- Harmonic Pulse=0.0003 (filtre momentum)

Optimisation V11 - Focus Robustesse:
  - Exit plus agressif (0.9978 vs 0.99748) → limite les drawdowns
  - Trailing stop plus serré (0.989 vs 0.991) → protège mieux les gains
  - Sharpe variance D1/D2 réduite de 50%: 0.136 vs 0.275
  - Avg Sharpe: 3.97 (meilleure stabilité entre datasets)
  - Sortino: 4.46/4.68 → downside risk minimisé

Résultats Robustesse:
  Dataset 1: PnL=2054%, Sharpe=4.04, Sortino=4.46, MDD=-0.128, Base=3.16
  Dataset 2: PnL=1715%, Sharpe=3.90, Sortino=4.68, MDD=-0.084, Base=2.74
  Avg Base: 2.95, Robustness Score: 3.069 ⭐
"""

WINDOW_B = 4           # Fenêtre courte pour réactivité
B_ENTRY = 1.00300      # Entrée quand prix > 1.00300 * HM (robustness-optimized)
B_EXIT = 0.99780       # Sortie quand prix < 0.99780 * HM (exit rapide, limite drawdowns)
TRAILING_STOP = 0.989  # Trailing stop à 98.9% du pic (protection serrée)
ENTRY_PULSE = 0.0003   # Pulse minimum pour entrée (filtre momentum)
MAX_ALLOC = 0.9999     # Allocation maximale

history = {}

def calculate_harmonic_mean(prices, window):
    """
    Harmonic Mean: privilégie les valeurs basses, idéal pour détecter les creux.
    HM = n / sum(1/x)
    """
    if len(prices) < window:
        return prices[-1]
    slice_data = prices[-window:]
    if any(x <= 0 for x in slice_data):
        return prices[-1]
    return len(slice_data) / sum(1.0/x for x in slice_data)

def make_decision(epoch: int, price_A: float, price_B: float):
    global history
    
    if epoch == 0:
        # Initialisation
        history = {
            'prices_B': [price_B],
            'in_B': False,
            'entry_idx': None,
            'prev_ratio': None
        }
        return {'Asset A': 0.0, 'Asset B': 0.0, 'Cash': 1.0}
    
    # Mise à jour historique
    history['prices_B'].append(price_B)
    prices_B = history['prices_B']
    
    # Attendre d'avoir assez de données
    if len(prices_B) < WINDOW_B:
        return {'Asset A': 0.0, 'Asset B': 0.0, 'Cash': 1.0}
    
    # Calcul du signal Harmonic Mean
    h_mean_B = calculate_harmonic_mean(prices_B, WINDOW_B)
    ratio_B = price_B / h_mean_B
    
    # Calcul du Harmonic Pulse (dérivée du ratio)
    if history['prev_ratio'] is not None:
        pulse = ratio_B - history['prev_ratio']
    else:
        pulse = 0.0
    history['prev_ratio'] = ratio_B
    
    alloc_B = 0.0
    
    # Gestion des positions avec trailing stop
    if history['in_B'] and history['entry_idx'] is not None:
        # Calculer le prix maximum depuis l'entrée
        max_since_entry = max(prices_B[history['entry_idx']:])
        trailing_threshold = max_since_entry * TRAILING_STOP
        
        # Sortir si trailing stop déclenché OU ratio sous exit
        if price_B < trailing_threshold or ratio_B < B_EXIT:
            history['in_B'] = False
            history['entry_idx'] = None
        else:
            # Maintenir la position
            alloc_B = MAX_ALLOC
    elif ratio_B > B_ENTRY and pulse > ENTRY_PULSE:
        # Signal d'entrée avec confirmation Harmonic Pulse
        # Entre uniquement si momentum haussier (pulse positif)
        alloc_B = MAX_ALLOC
        history['in_B'] = True
        history['entry_idx'] = len(prices_B) - 1
    
    return {
        'Asset A': 0.0,
        'Asset B': float(alloc_B),
        'Cash': float(1.0 - alloc_B)
    }

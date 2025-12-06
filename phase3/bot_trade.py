"""
HARMONIC MEAN ALGORITHM V8 - OPTIMIZED
=======================================
Stratégie: Trading exclusif sur Asset B avec signal Harmonic Mean
- Window court (4) pour réactivité optimale
- Entry=1.003, Exit=0.998 pour balance PnL/Sharpe
- Asset A non utilisé (corrélation faible avec B)

Résultats:
  Dataset 1: PnL=1863%, Sharpe=1.94, Base=2.90
  Dataset 2: PnL=1629%, Sharpe=1.91, Base=2.62
"""

WINDOW_B = 4         # Fenêtre courte pour réactivité
B_ENTRY = 1.003      # Entrée quand prix > 1.003 * HM
B_EXIT = 0.998       # Sortie quand prix < 0.998 * HM (hystérésis)
MAX_ALLOC = 0.9999   # Allocation maximale

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
        history = {'prices_B': [price_B], 'in_B': False}
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
    
    alloc_B = 0.0
    
    # Logique d'entrée/sortie avec hystérésis (évite les faux signaux)
    if history['in_B']:
        # Déjà en position: maintenir si ratio >= exit
        if ratio_B >= B_EXIT:
            alloc_B = MAX_ALLOC
        else:
            # Signal de sortie
            history['in_B'] = False
    elif ratio_B > B_ENTRY:
        # Signal d'entrée
        alloc_B = MAX_ALLOC
        history['in_B'] = True
    
    return {
        'Asset A': 0.0,
        'Asset B': float(alloc_B),
        'Cash': float(1.0 - alloc_B)
    }

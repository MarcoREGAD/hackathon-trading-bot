"""
Bot de trading basé sur un modèle d'IA léger.
Utilise une régression logistique avec des features simples pour prédire la direction des prix.
"""

import json
import math
from features import compute_features_from_close_series, WINDOW_MAX


# ============================================================================
# CONFIGURATION
# ============================================================================

# Nombre minimal de bougies nécessaires pour calculer les features
MIN_HISTORY_LENGTH = WINDOW_MAX

# Seuils de décision - OPTIMISÉS pour maximiser le PnL
UPPER_THRESHOLD = 0.525  # Si P(hausse) > 0.525 => position longue sur Asset A
LOWER_THRESHOLD = 0.475  # Si P(hausse) < 0.475 => position courte sur Asset A

# Taille de la position maximale sur Asset A (entre 0 et 1)
MAX_POSITION_ASSET_A = 0.90  # 90% max du capital sur Asset A pour les signaux très forts

# Seuils de haute confiance pour positions maximales
HIGH_CONFIDENCE_UPPER = 0.57  # Au-dessus = très forte confiance haussière
HIGH_CONFIDENCE_LOWER = 0.43  # En-dessous = très forte confiance baissière

# Fichier contenant les poids du modèle
MODEL_WEIGHTS_FILE = 'forex_model_weights.json'


# ============================================================================
# CHARGEMENT DU MODÈLE (au niveau module, une seule fois)
# ============================================================================

try:
    with open(MODEL_WEIGHTS_FILE, 'r') as f:
        _MODEL = json.load(f)
    _COEF = _MODEL['coef']
    _INTERCEPT = _MODEL['intercept']
    print(f"✅ Modèle chargé: {len(_COEF)} features, intercept={_INTERCEPT:.4f}")
except FileNotFoundError:
    print(f"⚠️  Fichier {MODEL_WEIGHTS_FILE} non trouvé. Utilisation de poids par défaut.")
    # Poids par défaut (5 features)
    _COEF = [0.0, 0.0, 0.0, 0.0, 0.0]
    _INTERCEPT = 0.0
except Exception as e:
    print(f"⚠️  Erreur lors du chargement du modèle: {e}")
    _COEF = [0.0, 0.0, 0.0, 0.0, 0.0]
    _INTERCEPT = 0.0


# ============================================================================
# HISTORIQUE GLOBAL
# ============================================================================

history = []


# ============================================================================
# FONCTIONS DU MODÈLE
# ============================================================================

def _sigmoid(z: float) -> float:
    """Fonction sigmoïde pour calculer la probabilité."""
    try:
        return 1.0 / (1.0 + math.exp(-z))
    except OverflowError:
        # Protection contre les valeurs extrêmes
        return 0.0 if z < 0 else 1.0


def predict_proba_up(features: list[float]) -> float:
    """
    Calcule P(hausse) = sigmoid(w·x + b) avec le modèle logistic.
    
    Args:
        features: Vecteur de features (doit correspondre à l'ordre du training)
    
    Returns:
        Probabilité que le prochain mouvement soit haussier (entre 0 et 1)
    """
    if len(features) != len(_COEF):
        print(f"⚠️  Nombre de features incorrect: {len(features)} vs {len(_COEF)} attendus")
        return 0.5  # Neutre par défaut
    
    z = _INTERCEPT
    for w, x in zip(_COEF, features):
        z += w * x
    
    return _sigmoid(z)


def proba_to_position(p_up: float) -> float:
    """
    Convertit une probabilité en position sur Asset A - VERSION OPTIMISÉE pour PnL.
    
    Args:
        p_up: Probabilité de hausse (entre 0 et 1)
    
    Returns:
        Position entre -0.85 et 0.90
        0.0 = position neutre (flat)
    """
    
    if p_up > HIGH_CONFIDENCE_UPPER:
        # TRÈS haute confiance haussière => position maximale
        strength = (p_up - HIGH_CONFIDENCE_UPPER) / (1.0 - HIGH_CONFIDENCE_UPPER)
        # Amplifier les très bons signaux
        position = 0.65 + (strength ** 0.8) * 0.25  # Entre 0.65 et 0.90
        return position
        
    elif p_up > UPPER_THRESHOLD:
        # Confiance haussière modérée => position progressive
        strength = (p_up - UPPER_THRESHOLD) / (HIGH_CONFIDENCE_UPPER - UPPER_THRESHOLD)
        # Croissance plus rapide vers les positions fortes
        position = (strength ** 0.75) * 0.65  # Entre 0 et 0.65
        return position
        
    elif p_up < HIGH_CONFIDENCE_LOWER:
        # TRÈS haute confiance baissière => minimiser Asset A
        strength = (HIGH_CONFIDENCE_LOWER - p_up) / HIGH_CONFIDENCE_LOWER
        position = -(0.55 + (strength ** 0.8) * 0.30)  # Entre -0.55 et -0.85
        return position
        
    elif p_up < LOWER_THRESHOLD:
        # Confiance baissière modérée
        strength = (LOWER_THRESHOLD - p_up) / (LOWER_THRESHOLD - HIGH_CONFIDENCE_LOWER)
        position = -((strength ** 0.75) * 0.55)  # Entre 0 et -0.55
        return position
        
    else:
        # Zone neutre => rester flat
        return 0.0


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def extract_closes_from_history(hist: list[dict]) -> list[float]:
    """Extrait la série de prix de clôture (priceA) depuis l'historique."""
    return [h['priceA'] for h in hist]


def make_flat_decision() -> dict:
    """Retourne une décision neutre (répartition égale)."""
    return {
        'Asset A': 1/3,
        'Asset B': 1/3,
        'Cash': 1/3
    }


def build_decision_from_position(target_position: float) -> dict:
    """
    Construit une décision d'allocation à partir d'une position cible sur Asset A.
    VERSION OPTIMISÉE pour maximiser le PnL.
    
    Args:
        target_position: Position sur Asset A (entre -0.85 et 0.90)
            > 0 = bull sur Asset A
            0 = neutre (33/33/33)
            < 0 = bear sur Asset A
    
    Returns:
        Dict avec les allocations pour Asset A, Asset B et Cash
    """
    if target_position > 0:
        # Signal haussier => maximiser Asset A de manière progressive
        # Formule optimisée: plus le signal est fort, plus on met en Asset A
        asset_a_weight = (1/3) + target_position * 0.70  # De 33% à ~96%
        asset_a_weight = min(asset_a_weight, MAX_POSITION_ASSET_A)
        remaining = 1 - asset_a_weight
        # Garder un peu de cash de sécurité
        cash_weight = remaining * 0.60
        asset_b_weight = remaining * 0.40
        
    elif target_position < 0:
        # Signal baissier => réduire Asset A, augmenter Asset B et Cash
        # Formule: plus le signal est baissier, moins d'Asset A
        asset_a_weight = max(0.03, (1/3) + target_position * 0.37)  # De ~3% à 33%
        remaining = 1 - asset_a_weight
        # Favoriser Asset B (potentiel de gain) et Cash (sécurité)
        asset_b_weight = remaining * 0.55
        cash_weight = remaining * 0.45
        
    else:
        # Position neutre = répartition égale
        asset_a_weight = 1/3
        asset_b_weight = 1/3
        cash_weight = 1/3
    
    # S'assurer que les poids sont valides
    total = asset_a_weight + asset_b_weight + cash_weight
    if abs(total - 1.0) > 0.001:
        # Normaliser si nécessaire
        asset_a_weight /= total
        asset_b_weight /= total
        cash_weight /= total
    
    # S'assurer que les valeurs sont dans [0, 1]
    asset_a_weight = max(0.0, min(1.0, asset_a_weight))
    asset_b_weight = max(0.0, min(1.0, asset_b_weight))
    cash_weight = max(0.0, min(1.0, cash_weight))
    
    return {
        'Asset A': asset_a_weight,
        'Asset B': asset_b_weight,
        'Cash': cash_weight
    }


# ============================================================================
# FONCTION PRINCIPALE DE DÉCISION
# ============================================================================

def make_decision(epoch: int, priceA: float, priceB: float) -> dict:
    """
    Fonction appelée à chaque tick pour prendre une décision de trading.
    
    Args:
        epoch: Numéro de l'epoch (pas de temps)
        priceA: Prix actuel de l'Asset A
        priceB: Prix actuel de l'Asset B
    
    Returns:
        Dict avec les allocations: {'Asset A': float, 'Asset B': float, 'Cash': float}
        La somme doit être égale à 1.0
    """
    # 1. Mettre à jour l'historique
    history.append({
        'epoch': epoch,
        'priceA': priceA,
        'priceB': priceB
    })
    
    # 2. Vérifier qu'on a assez d'historique
    if len(history) < MIN_HISTORY_LENGTH:
        # Pas assez de données => rester neutre
        return make_flat_decision()
    
    # 3. Extraire les prix de clôture
    closes = extract_closes_from_history(history)
    
    # 4. Calculer les features
    try:
        features = compute_features_from_close_series(closes)
    except Exception as e:
        print(f"⚠️  Erreur lors du calcul des features: {e}")
        return make_flat_decision()
    
    # 5. Obtenir la probabilité de mouvement haussier
    p_up = predict_proba_up(features)
    
    # 6. Convertir en position cible
    target_position = proba_to_position(p_up)
    
    # 7. Construire la décision
    decision = build_decision_from_position(target_position)
    
    # Debug (afficher tous les 100 epochs et forcer l'affichage des 20 premiers)
    if epoch % 100 == 0 or epoch < 20:
        print(f"Epoch {epoch}: P(hausse)={p_up:.3f}, Position={target_position:.3f}, Asset A={decision['Asset A']:.3f}")
    
    return decision

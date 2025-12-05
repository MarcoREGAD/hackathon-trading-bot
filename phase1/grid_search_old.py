#! /usr/bin/env python3

import sys
import os
import pandas as pd
from itertools import product
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

# Empêcher la création de __pycache__
sys.dont_write_bytecode = True

from scoring.scoring import get_local_score


def set_env_params(short_window: int, long_window: int, trend_gain: float, 
                    smooth_alpha: float, dd_hard: float):
    """Configure les variables d'environnement pour les paramètres de trading."""
    os.environ['SHORT_WINDOW'] = str(short_window)
    os.environ['LONG_WINDOW'] = str(long_window)
    os.environ['TREND_GAIN'] = str(trend_gain)
    os.environ['SMOOTH_ALPHA'] = str(smooth_alpha)
    os.environ['DD_HARD'] = str(dd_hard)

def find_csv_file(path_csv: str) -> pd.DataFrame:
    """Charge le fichier CSV avec les prix."""
    if not os.path.exists(path_csv):
        raise FileNotFoundError(f"Le fichier CSV {path_csv} n'existe pas")
    prices_list = [pd.read_csv(path_csv, index_col=0)]
    prices = pd.concat(prices_list, axis=1)
    prices["Cash"] = 1
    return prices

def make_decision_with_params(
    epoch: int, 
    price: float, 
    history: List[float],
    price_history: List[float],
    prev_weight_asset: float,
    portfolio_value: float,
    peak_value: float,
    short_window: int,
    long_window: int,
    trend_gain: float,
    smooth_alpha: float,
    dd_hard: float
) -> Tuple[Dict[str, float], float, float, float]:
    """
    Fonction de décision avec paramètres configurables.
    Reproduction de la logique de bot_trade.py
    """
    price_history.append(price)
    
    # Pas assez de données
    if len(price_history) < max(short_window, long_window):
        return {"Asset A": 0.5, "Cash": 0.5}, prev_weight_asset, portfolio_value, peak_value
    
    # Calcul des moyennes mobiles
    short_ma = sum(price_history[-short_window:]) / short_window
    long_ma = sum(price_history[-long_window:]) / long_window
    
    # Calcul de la pente de tendance
    trend_slope = (short_ma - long_ma) / long_ma if long_ma != 0 else 0
    
    # Poids cible basé sur la tendance
    target_weight = 0.5 + trend_gain * trend_slope
    target_weight = max(0.0, min(1.0, target_weight))
    
    # Lissage
    new_weight = smooth_alpha * target_weight + (1 - smooth_alpha) * prev_weight_asset
    
    # Mise à jour approximative du portefeuille
    if epoch > 0:
        prev_price = price_history[-2]
        asset_return = (price - prev_price) / prev_price if prev_price != 0 else 0
        portfolio_value *= (1 + prev_weight_asset * asset_return)
        peak_value = max(peak_value, portfolio_value)
    
    # Protection drawdown
    drawdown = (portfolio_value - peak_value) / peak_value if peak_value != 0 else 0
    if drawdown < -dd_hard:
        new_weight = 0.0
    
    return {
        "Asset A": new_weight,
        "Cash": 1.0 - new_weight
    }, new_weight, portfolio_value, peak_value

def evaluate_params(
    params: Tuple
) -> Dict:
    """
    Évalue une combinaison de paramètres et retourne le résultat complet.
    
    Args:
        params: Tuple (prices, short_window, long_window, trend_gain, smooth_alpha, dd_hard)
    
    Returns:
        Dictionnaire avec les résultats
    """
    prices, short_window, long_window, trend_gain, smooth_alpha, dd_hard = params
    
    try:
        price_history = []
        prev_weight_asset = 0.5
        portfolio_value = 1.0
        peak_value = 1.0
        output = []
        
        for index, row in prices.iterrows():
            decision, prev_weight_asset, portfolio_value, peak_value = make_decision_with_params(
                int(index),
                float(row['Asset A']),
                [],
                price_history,
                prev_weight_asset,
                portfolio_value,
                peak_value,
                short_window,
                long_window,
                trend_gain,
                smooth_alpha,
                dd_hard
            )
            decision['epoch'] = int(index)
            output.append(decision)
        
        positions = pd.DataFrame(output).set_index("epoch")
        local_score = get_local_score(prices=prices, positions=positions)
        
        # Le score principal est le Sharpe ratio
        stats = local_score['stats']
        
        return {
            'SHORT_WINDOW': short_window,
            'LONG_WINDOW': long_window,
            'TREND_GAIN': trend_gain,
            'SMOOTH_ALPHA': smooth_alpha,
            'DD_HARD': dd_hard,
            'sharpe_ratio': stats['sharpe_ratio'],
            'cumulative_return': stats['cumulative_return'],
            'annualized_return': stats['annualized_return'],
            'annualized_volatility': stats['annualized_volatility'],
            'max_drawdown': stats['max_drawdown'],
            'success': True
        }
    except Exception as e:
        return {
            'SHORT_WINDOW': short_window,
            'LONG_WINDOW': long_window,
            'TREND_GAIN': trend_gain,
            'SMOOTH_ALPHA': smooth_alpha,
            'DD_HARD': dd_hard,
            'error': str(e),
            'success': False
        }

def grid_search(
    prices: pd.DataFrame,
    short_window_range: range,
    long_window_range: range,
    trend_gain_values: List[float],
    smooth_alpha_values: List[float],
    dd_hard_values: List[float],
    max_workers: int = None
) -> List[Dict]:
    """
    Effectue une recherche en grille sur les paramètres avec multiprocessing.
    
    Args:
        prices: DataFrame avec les prix
        short_window_range: Range de valeurs pour SHORT_WINDOW
        long_window_range: Range de valeurs pour LONG_WINDOW
        trend_gain_values: Liste de valeurs pour TREND_GAIN
        smooth_alpha_values: Liste de valeurs pour SMOOTH_ALPHA
        dd_hard_values: Liste de valeurs pour DD_HARD
        max_workers: Nombre de workers (None = auto)
    
    Returns:
        Liste de dictionnaires avec les résultats triés par score
    """
    # Générer toutes les combinaisons valides
    combinations = [
        (prices, sw, lw, tg, sa, dd)
        for sw, lw, tg, sa, dd in product(
            short_window_range,
            long_window_range,
            trend_gain_values,
            smooth_alpha_values,
            dd_hard_values
        )
        if sw < lw  # short_window doit être < long_window
    ]
    
    total_combinations = len(combinations)
    
    if max_workers is None:
        max_workers = cpu_count()
    
    print(f"\n🔍 Grid Search en cours avec {max_workers} workers...")
    print(f"📊 Nombre total de combinaisons à tester : {total_combinations}")
    print(f"🚀 Multiprocessing activé pour une vitesse maximale !\n")
    
    results = []
    completed = 0
    
    # Utiliser ProcessPoolExecutor pour le parallélisme
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Soumettre tous les jobs
        futures = [executor.submit(evaluate_params, combo) for combo in combinations]
        
        # Collecter les résultats au fur et à mesure
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            
            if result['success']:
                results.append(result)
                
                # Affichage de progression
                if completed % 10 == 0 or completed == total_combinations:
                    best_sharpe = max([r['sharpe_ratio'] for r in results]) if results else 0
                    print(f"⏳ {completed}/{total_combinations} "
                          f"({100*completed/total_combinations:.1f}%) | "
                          f"Meilleur Sharpe : {best_sharpe:.4f}")
            else:
                print(f"❌ Erreur: {result['error']}")
    
    print(f"\n✅ Grid Search terminé ! {len(results)} combinaisons testées avec succès.\n")
    
    # Trier par Sharpe ratio décroissant
    results.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
    
    return results

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 grid_search.py <path_to_csv> [--quick|--full]")
        print("  --quick : Recherche rapide avec peu de combinaisons")
        print("  --full  : Recherche exhaustive (très long)")
        sys.exit(1)
    
    path_csv = sys.argv[1]
    mode = '--quick' if '--quick' in sys.argv else ('--full' if '--full' in sys.argv else 'normal')
    
    print("=" * 70)
    print("🤖 GRID SEARCH - Optimisation des paramètres de trading")
    print("=" * 70)
    
    # Charger les données
    prices = find_csv_file(path_csv)
    print(f"✅ Données chargées : {len(prices)} observations")
    
    # Définir les plages de recherche selon le mode
    if mode == '--quick':
        short_window_range = range(30, 45, 5)  # 30, 35, 40
        long_window_range = range(45, 60, 5)   # 45, 50, 55
        trend_gain_values = [1.5, 2.0, 2.5]
        smooth_alpha_values = [0.2, 0.3]
        dd_hard_values = [0.15, 0.20]
        print("🔬 Mode de recherche : RAPIDE")
    elif mode == '--full':
        short_window_range = range(20, 51, 5)   # 20 à 50
        long_window_range = range(40, 81, 5)    # 40 à 80
        trend_gain_values = [1.0, 1.5, 2.0, 2.5, 3.0]
        smooth_alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5]
        dd_hard_values = [0.10, 0.15, 0.20, 0.25]
        print("🔬 Mode de recherche : EXHAUSTIF")
    else:
        short_window_range = range(30, 50, 5)   # 30, 35, 40, 45
        long_window_range = range(45, 65, 5)    # 45, 50, 55, 60
        trend_gain_values = [1.5, 2.0, 2.5, 3.0]
        smooth_alpha_values = [0.2, 0.3, 0.4]
        dd_hard_values = [0.15, 0.20, 0.25]
        print("🔬 Mode de recherche : NORMAL")
    
    print(f"📈 SHORT_WINDOW : {list(short_window_range)}")
    print(f"📉 LONG_WINDOW : {list(long_window_range)}")
    print(f"📊 TREND_GAIN : {trend_gain_values}")
    print(f"🎚️  SMOOTH_ALPHA : {smooth_alpha_values}")
    print(f"🛡️  DD_HARD : {dd_hard_values}")
    
    # Effectuer la recherche
    results = grid_search(
        prices,
        short_window_range,
        long_window_range,
        trend_gain_values,
        smooth_alpha_values,
        dd_hard_values
    )
    
    # Afficher les résultats
    print("\n" + "=" * 70)
    print("🏆 TOP 10 DES MEILLEURES COMBINAISONS")
    print("=" * 70)
    
    for i, result in enumerate(results[:10], 1):
        print(f"\n#{i}")
        print(f"  SHORT_WINDOW = {result['SHORT_WINDOW']}")
        print(f"  LONG_WINDOW = {result['LONG_WINDOW']}")
        print(f"  TREND_GAIN = {result['TREND_GAIN']}")
        print(f"  SMOOTH_ALPHA = {result['SMOOTH_ALPHA']}")
        print(f"  DD_HARD = {result['DD_HARD']}")
        print(f"  Sharpe Ratio: {result['sharpe_ratio']:.4f}")
        print(f"  Rendement cumulé: {result['cumulative_return']:.2%}")
        print(f"  Rendement annualisé: {result['annualized_return']:.2%}")
        print(f"  Volatilité annualisée: {result['annualized_volatility']:.2%}")
        print(f"  Max Drawdown: {result['max_drawdown']:.2%}")
    
    # Sauvegarder tous les résultats dans un CSV
    output_file = "grid_search_results.csv"
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False)
    print(f"\n💾 Résultats complets sauvegardés dans '{output_file}'")
    
    # Afficher la meilleure combinaison et générer le .env
    best = results[0]
    print("\n" + "=" * 70)
    print("⭐ MEILLEURE CONFIGURATION TROUVÉE")
    print("=" * 70)
    print(f"SHORT_WINDOW = {best['SHORT_WINDOW']}")
    print(f"LONG_WINDOW = {best['LONG_WINDOW']}")
    print(f"TREND_GAIN = {best['TREND_GAIN']}")
    print(f"SMOOTH_ALPHA = {best['SMOOTH_ALPHA']}")
    print(f"DD_HARD = {best['DD_HARD']}")
    print("=" * 70)
    
    # Créer un fichier .env.optimized avec les meilleurs paramètres
    env_content = f"""# Configuration optimale trouvée par grid search
# Sharpe Ratio: {best['sharpe_ratio']:.4f}
# Rendement cumulé: {best['cumulative_return']:.2%}

SHORT_WINDOW={best['SHORT_WINDOW']}
LONG_WINDOW={best['LONG_WINDOW']}
TREND_GAIN={best['TREND_GAIN']}
SMOOTH_ALPHA={best['SMOOTH_ALPHA']}
DD_HARD={best['DD_HARD']}
"""
    
    with open('.env.optimized', 'w') as f:
        f.write(env_content)
    
    print(f"\n💡 Fichier '.env.optimized' créé avec les meilleurs paramètres")
    print("   Copiez-le vers '.env' pour utiliser cette configuration :")
    print("   cp .env.optimized .env")

if __name__ == "__main__":
    main()

#! /usr/bin/env python3
"""
Grid Search Générique pour l'optimisation de stratégies de trading.

Ce script lit automatiquement le fichier .env.example pour détecter tous les paramètres
configurables et génère un grid search sur ces paramètres.

Configuration du grid search via grid_search_config.py (optionnel).
"""

import sys
import os
import re
import warnings
import pandas as pd
from itertools import product
from typing import Dict, List, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

# Supprimer tous les warnings pour une sortie propre
warnings.filterwarnings('ignore')

# Empêcher la création de __pycache__
sys.dont_write_bytecode = True

from scoring.scoring import get_local_score


def parse_env_example(filepath: str = '.env.example') -> Dict[str, Any]:
    """
    Parse le fichier .env.example pour extraire les paramètres et leurs types.
    
    Returns:
        Dict avec {param_name: {'value': default_value, 'type': type}}
    """
    params = {}
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Fichier {filepath} introuvable")
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Détecter le type
                if '.' in value:
                    params[key] = {'value': float(value), 'type': float}
                else:
                    params[key] = {'value': int(value), 'type': int}
    
    return params


def load_grid_config(params: Dict[str, Any]) -> Dict[str, List]:
    """
    Charge la configuration du grid search.
    Si grid_search_config.py existe, l'utilise, sinon crée des ranges par défaut.
    
    Returns:
        Dict avec {param_name: [list_of_values]}
    """
    config_file = 'grid_search_config.py'
    
    if os.path.exists(config_file):
        # Charger la config personnalisée
        import importlib.util
        spec = importlib.util.spec_from_file_location("grid_config", config_file)
        grid_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(grid_config)
        
        # Récupérer les configs par mode
        if hasattr(grid_config, 'GRID_CONFIG'):
            return grid_config.GRID_CONFIG
    
    # Configuration par défaut basée sur les valeurs de .env.example
    config = {}
    for param_name, param_info in params.items():
        default_val = param_info['value']
        param_type = param_info['type']
        
        if param_type == int:
            # Pour les entiers, faire une plage autour de la valeur par défaut
            start = max(1, int(default_val * 0.7))
            end = int(default_val * 1.5) + 1
            step = max(1, (end - start) // 5)
            config[param_name] = list(range(start, end, step))
        else:
            # Pour les floats, créer quelques valeurs autour du défaut
            config[param_name] = [
                round(default_val * 0.7, 2),
                round(default_val * 0.85, 2),
                default_val,
                round(default_val * 1.15, 2),
                round(default_val * 1.3, 2)
            ]
    
    return config


def find_csv_file(path_csv: str) -> pd.DataFrame:
    """Charge le fichier CSV avec les prix."""
    if not os.path.exists(path_csv):
        raise FileNotFoundError(f"Le fichier CSV {path_csv} n'existe pas")
    prices_list = [pd.read_csv(path_csv, index_col=0)]
    prices = pd.concat(prices_list, axis=1)
    prices["Cash"] = 1
    return prices


def run_strategy_with_params(prices: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Exécute la stratégie avec les paramètres donnés.
    Importe bot_trade dynamiquement pour éviter les conflits de variables globales.
    """
    # Configurer les variables d'environnement
    for key, value in params.items():
        os.environ[key] = str(value)
    
    # Recharger bot_trade avec les nouvelles variables d'environnement
    import importlib
    if 'bot_trade' in sys.modules:
        importlib.reload(sys.modules['bot_trade'])
    else:
        import bot_trade
    
    # Réinitialiser les variables globales de bot_trade
    import bot_trade
    bot_trade._price_history = []
    bot_trade._prev_weight_asset = 0.5
    bot_trade._portfolio_value = 1.0
    bot_trade._peak_value = 1.0
    
    # Exécuter la stratégie
    output = []
    for index, row in prices.iterrows():
        decision = bot_trade.make_decision(int(index), float(row['Asset A']))
        decision['epoch'] = int(index)
        output.append(decision)
    
    return pd.DataFrame(output).set_index("epoch")


def evaluate_params(params_tuple: Tuple) -> Dict:
    """
    Évalue une combinaison de paramètres et retourne le résultat complet.
    
    Args:
        params_tuple: Tuple (prices, {param_name: value, ...})
    
    Returns:
        Dictionnaire avec les résultats
    """
    prices, param_dict = params_tuple
    
    try:
        positions = run_strategy_with_params(prices, param_dict)
        local_score = get_local_score(prices=prices, positions=positions)
        
        stats = local_score['stats']
        
        result = {**param_dict}
        result.update({
            'sharpe_ratio': stats['sharpe_ratio'],
            'cumulative_return': stats['cumulative_return'],
            'annualized_return': stats['annualized_return'],
            'annualized_volatility': stats['annualized_volatility'],
            'max_drawdown': stats['max_drawdown'],
            'success': True
        })
        return result
        
    except Exception as e:
        result = {**param_dict}
        result.update({
            'error': str(e),
            'success': False
        })
        return result


def grid_search(
    prices: pd.DataFrame,
    grid_config: Dict[str, List],
    max_workers: int = None
) -> List[Dict]:
    """
    Effectue une recherche en grille sur les paramètres avec multiprocessing.
    
    Args:
        prices: DataFrame avec les prix
        grid_config: Dict avec {param_name: [list_of_values]}
        max_workers: Nombre de workers (None = auto)
    
    Returns:
        Liste de dictionnaires avec les résultats triés par score
    """
    # Générer toutes les combinaisons
    param_names = list(grid_config.keys())
    param_values = list(grid_config.values())
    
    combinations = [
        (prices, dict(zip(param_names, combo)))
        for combo in product(*param_values)
    ]
    
    total_combinations = len(combinations)
    
    if max_workers is None:
        max_workers = cpu_count()
    
    print(f"\n🔍 Grid Search en cours avec {max_workers} workers...")
    print(f"📊 Nombre total de combinaisons à tester : {total_combinations}")
    print(f"🚀 Multiprocessing activé pour une vitesse maximale !\n")
    
    results = []
    completed = 0
    errors = 0
    
    # Utiliser ProcessPoolExecutor pour le parallélisme
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_params, combo) for combo in combinations]
        
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            
            if result['success']:
                results.append(result)
                
                # Affichage de progression
                if completed % 10 == 0 or completed == total_combinations:
                    best_pnl = max([r['cumulative_return'] for r in results]) if results else 0
                    print(f"⏳ {completed}/{total_combinations} "
                          f"({100*completed/total_combinations:.1f}%) | "
                          f"Meilleur PNL : {best_pnl:.2%} | "
                          f"Erreurs : {errors}")
            else:
                errors += 1
                if errors <= 3:  # Afficher seulement les 3 premières erreurs
                    print(f"❌ Erreur: {result['error']}")
    
    print(f"\n✅ Grid Search terminé ! {len(results)} combinaisons testées avec succès.\n")
    
    # Trier par Sharpe ratio décroissant
    results.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
    
    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 grid_search.py <path_to_csv>")
        print("\nLe script lit automatiquement .env.example pour détecter les paramètres.")
        print("Pour personnaliser les ranges, créez un fichier grid_search_config.py")
        sys.exit(1)
    
    path_csv = sys.argv[1]
    
    print("=" * 70)
    print("🤖 GRID SEARCH GÉNÉRIQUE - Optimisation des paramètres")
    print("=" * 70)
    
    # Parser .env.example
    print("\n📋 Lecture de .env.example...")
    env_params = parse_env_example()
    print(f"   Paramètres détectés : {list(env_params.keys())}")
    
    # Charger la config du grid search
    print("\n🔧 Configuration du grid search...")
    grid_config = load_grid_config(env_params)
    
    for param_name, values in grid_config.items():
        print(f"   {param_name}: {values}")
    
    # Calculer le nombre total de combinaisons
    total = 1
    for values in grid_config.values():
        total *= len(values)
    print(f"\n📊 Nombre total de combinaisons : {total}")
    
    # Charger les données
    print(f"\n📂 Chargement des données...")
    prices = find_csv_file(path_csv)
    print(f"   ✅ {len(prices)} observations chargées")
    
    # Effectuer la recherche
    results = grid_search(prices, grid_config)
    
    if not results:
        print("❌ Aucun résultat valide trouvé !")
        sys.exit(1)
    
    # Afficher les résultats
    print("\n" + "=" * 70)
    print("🏆 TOP 10 DES MEILLEURES COMBINAISONS")
    print("=" * 70)
    
    param_names = [k for k in results[0].keys() if k not in 
                   ['sharpe_ratio', 'cumulative_return', 'annualized_return', 
                    'annualized_volatility', 'max_drawdown', 'success', 'error']]
    
    for i, result in enumerate(results[:10], 1):
        print(f"\n#{i}")
        for param_name in param_names:
            print(f"  {param_name} = {result[param_name]}")
        print(f"  Sharpe Ratio: {result['sharpe_ratio']:.4f}")
        print(f"  Rendement cumulé: {result['cumulative_return']:.2%}")
        print(f"  Rendement annualisé: {result['annualized_return']:.2%}")
        print(f"  Volatilité annualisée: {result['annualized_volatility']:.2%}")
        print(f"  Max Drawdown: {result['max_drawdown']:.2%}")
    
    # Sauvegarder tous les résultats
    output_file = "grid_search_results.csv"
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False)
    print(f"\n💾 Résultats complets sauvegardés dans '{output_file}'")
    
    # Générer .env.optimized
    best = results[0]
    print("\n" + "=" * 70)
    print("⭐ MEILLEURE CONFIGURATION TROUVÉE")
    print("=" * 70)
    
    env_content = f"""# Configuration optimale trouvée par grid search
# Sharpe Ratio: {best['sharpe_ratio']:.4f}
# Rendement cumulé: {best['cumulative_return']:.2%}

"""
    
    for param_name in param_names:
        value = best[param_name]
        env_content += f"{param_name}={value}\n"
        print(f"{param_name} = {value}")
    
    print("=" * 70)
    
    with open('.env.optimized', 'w') as f:
        f.write(env_content)
    
    print(f"\n💡 Fichier '.env.optimized' créé avec les meilleurs paramètres")
    print("   Copiez-le vers '.env' pour utiliser cette configuration :")
    print("   cp .env.optimized .env")


if __name__ == "__main__":
    main()

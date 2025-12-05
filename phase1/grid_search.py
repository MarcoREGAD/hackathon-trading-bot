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
    mm_a_length: int, 
    mm_b_length: int
) -> Dict[str, float]:
    """
    Fonction de décision avec paramètres configurables.
    
    Args:
        epoch: L'époque actuelle
        price: Le prix actuel
        history: L'historique des prix
        mm_a_length: Longueur de la moyenne mobile A
        mm_b_length: Longueur de la moyenne mobile B
    
    Returns:
        Dictionnaire avec l'allocation {'Asset A': float, 'Cash': float}
    """
    history.append(price)
    
    # On ne peut rien faire avant d'avoir au moins 500 points
    if len(history) < 500:
        return {"Asset A": 0.5, "Cash": 0.5}
    
    # Calcul des moyennes mobiles
    mm_a = sum(history[-mm_a_length:]) / mm_a_length
    mm_b = sum(history[-mm_b_length:]) / mm_b_length
    
    # Logique de décision
    if mm_a > mm_b:
        # Tendance haussière : on investit (ex : 80%)
        return {"Asset A": 0.8, "Cash": 0.2}
    else:
        # Tendance baissière : on se protège (ex : 20%)
        return {"Asset A": 0.2, "Cash": 0.8}

def evaluate_params(
    prices: pd.DataFrame, 
    mm_a_length: int, 
    mm_b_length: int
) -> Dict:
    """
    Évalue une combinaison de paramètres et retourne le résultat complet.
    
    Args:
        prices: DataFrame avec les prix
        mm_a_length: Longueur de la moyenne mobile A
        mm_b_length: Longueur de la moyenne mobile B
    
    Returns:
        Dictionnaire avec les résultats
    """
    try:
        history = []
        output = []
        
        for index, row in prices.iterrows():
            decision = make_decision_with_params(
                int(index), 
                float(row['Asset A']), 
                history, 
                mm_a_length, 
                mm_b_length
            )
            decision['epoch'] = int(index)
            output.append(decision)
        
        positions = pd.DataFrame(output).set_index("epoch")
        local_score = get_local_score(prices=prices, positions=positions)
        
        # Le score principal est le Sharpe ratio
        stats = local_score['stats']
        
        return {
            'MM_A_LENGTH': mm_a_length,
            'MM_B_LENGTH': mm_b_length,
            'sharpe_ratio': stats['sharpe_ratio'],
            'cumulative_return': stats['cumulative_return'],
            'annualized_return': stats['annualized_return'],
            'annualized_volatility': stats['annualized_volatility'],
            'max_drawdown': stats['max_drawdown'],
            'success': True
        }
    except Exception as e:
        return {
            'MM_A_LENGTH': mm_a_length,
            'MM_B_LENGTH': mm_b_length,
            'error': str(e),
            'success': False
        }

def grid_search(
    prices: pd.DataFrame,
    mm_a_range: range,
    mm_b_range: range,
    max_workers: int = None
) -> List[Dict]:
    """
    Effectue une recherche en grille sur les paramètres avec multiprocessing.
    
    Args:
        prices: DataFrame avec les prix
        mm_a_range: Range de valeurs pour MM_A_LENGTH
        mm_b_range: Range de valeurs pour MM_B_LENGTH
        max_workers: Nombre de workers (None = auto)
    
    Returns:
        Liste de dictionnaires avec les résultats triés par score
    """
    # Générer toutes les combinaisons valides
    combinations = [
        (mm_a, mm_b) 
        for mm_a, mm_b in product(mm_a_range, mm_b_range)
        if mm_a >= mm_b  # MM_A doit être >= MM_B pour avoir du sens
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
        future_to_params = {
            executor.submit(evaluate_params, prices, mm_a, mm_b): (mm_a, mm_b)
            for mm_a, mm_b in combinations
        }
        
        # Collecter les résultats au fur et à mesure
        for future in as_completed(future_to_params):
            completed += 1
            result = future.result()
            
            if result['success']:
                results.append(result)
                
                # Affichage de progression
                if completed % 5 == 0 or completed == total_combinations:
                    best_sharpe = max([r['sharpe_ratio'] for r in results]) if results else 0
                    print(f"⏳ {completed}/{total_combinations} "
                          f"({100*completed/total_combinations:.1f}%) | "
                          f"Meilleur Sharpe : {best_sharpe:.4f}")
            else:
                print(f"❌ Erreur avec MM_A={result['MM_A_LENGTH']}, "
                      f"MM_B={result['MM_B_LENGTH']}: {result['error']}")
    
    print(f"\n✅ Grid Search terminé ! {len(results)} combinaisons testées avec succès.\n")
    
    # Trier par Sharpe ratio décroissant
    results.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
    
    return results

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 grid_search.py <path_to_csv> [--full]")
        print("  --full : Effectue une recherche exhaustive (plus long)")
        sys.exit(1)
    
    path_csv = sys.argv[1]
    full_search = "--full" in sys.argv
    
    print("=" * 60)
    print("🤖 GRID SEARCH - Optimisation des paramètres de trading")
    print("=" * 60)
    
    # Charger les données
    prices = find_csv_file(path_csv)
    print(f"✅ Données chargées : {len(prices)} observations")
    
    # Définir les plages de recherche
    if full_search:
        mm_a_range = range(3, 21)  # 3 à 20
        mm_b_range = range(2, 16)  # 2 à 15
        print("🔬 Mode de recherche : EXHAUSTIF")
    else:
        mm_a_range = range(5, 13)  # 5 à 12
        mm_b_range = range(3, 9)   # 3 à 8
        print("🔬 Mode de recherche : RAPIDE")
    
    print(f"📈 MM_A_LENGTH : {mm_a_range.start} à {mm_a_range.stop - 1}")
    print(f"📉 MM_B_LENGTH : {mm_b_range.start} à {mm_b_range.stop - 1}")
    
    # Effectuer la recherche
    results = grid_search(prices, mm_a_range, mm_b_range)
    
    # Afficher les résultats
    print("\n" + "=" * 60)
    print("🏆 TOP 10 DES MEILLEURES COMBINAISONS")
    print("=" * 60)
    
    for i, result in enumerate(results[:10], 1):
        print(f"\n#{i}")
        print(f"  MM_A_LENGTH = {result['MM_A_LENGTH']}")
        print(f"  MM_B_LENGTH = {result['MM_B_LENGTH']}")
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
    
    # Afficher la meilleure combinaison
    best = results[0]
    print("\n" + "=" * 60)
    print("⭐ MEILLEURE CONFIGURATION TROUVÉE")
    print("=" * 60)
    print(f"MM_A_LENGTH = {best['MM_A_LENGTH']}")
    print(f"MM_B_LENGTH = {best['MM_B_LENGTH']}")
    print("=" * 60)

if __name__ == "__main__":
    main()

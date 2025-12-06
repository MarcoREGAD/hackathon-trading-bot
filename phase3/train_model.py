#!/usr/bin/env python3
"""
Script d'entraînement offline pour le modèle de trading Forex.
Entraîne une régression logistique et exporte les poids dans un fichier JSON.

Usage:
    python train_model.py <path_to_csv> [path_to_csv2] [path_to_csv3] ...
    python train_model.py data/currencies/*.csv
    python train_model.py data/currencies/gbp_eur.csv data/currencies/usd_eur.csv
    
Le CSV doit contenir les colonnes: Asset A, Asset B (ou similar) OU date,exchange_rate
"""

import sys
import json
import glob
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from features import compute_features_from_close_series, WINDOW_MAX
from multiprocessing import Pool, cpu_count


def prepare_data(csv_path: str, asset_col: str = 'Asset A'):
    """
    Charge les données et prépare X (features) et y (labels).
    Supporte deux formats de CSV:
    1. Format "Asset A, Asset B" (phase1/phase3 avec index)
    2. Format Forex "date,exchange_rate" (currencies/)
    
    Args:
        csv_path: Chemin vers le fichier CSV
        asset_col: Nom de la colonne de prix à utiliser (ou 'exchange_rate' pour Forex)
    
    Returns:
        X (array), y (array), feature_names (list)
    """
    print(f"Chargement des données depuis {csv_path}...")
    
    # Essayer de détecter le format du fichier
    df_test = pd.read_csv(csv_path, nrows=5)
    
    if 'exchange_rate' in df_test.columns:
        # Format Forex: date,exchange_rate
        print("✓ Format Forex détecté (date,exchange_rate)")
        df = pd.read_csv(csv_path)
        asset_col = 'exchange_rate'
        print(f"Date range: {df['date'].iloc[0]} → {df['date'].iloc[-1]}")
    elif 'Asset A' in df_test.columns or 'Asset B' in df_test.columns:
        # Format Asset A/B avec index
        print("✓ Format Asset A/B détecté")
        df = pd.read_csv(csv_path, index_col=0)
    else:
        # Essayer de charger avec index par défaut
        df = pd.read_csv(csv_path, index_col=0)
    
    if asset_col not in df.columns:
        raise ValueError(f"Colonne '{asset_col}' non trouvée. Colonnes disponibles: {df.columns.tolist()}")
    
    print(f"Nombre de lignes: {len(df)}")
    
    # Créer le label : future_return et y
    df['future_return'] = df[asset_col].shift(-1) / df[asset_col] - 1
    df['y'] = (df['future_return'] > 0).astype(int)
    
    # Supprimer les NaN
    df = df.dropna()
    print(f"Nombre de lignes après nettoyage: {len(df)}")
    
    # Calculer les features pour chaque ligne
    closes = df[asset_col].tolist()
    X_list = []
    y_list = []
    
    print("Calcul des features...")
    for i in range(WINDOW_MAX, len(closes)):
        # Prendre la fenêtre de prix jusqu'à i (inclus)
        window = closes[:i+1]
        
        # Calculer les features
        try:
            features = compute_features_from_close_series(window)
            X_list.append(features)
            # Le label correspond à l'index i dans le DataFrame nettoyé
            # Mais attention: après dropna, les indices peuvent ne pas correspondre
            # On utilise iloc pour être sûr
            y_list.append(df.iloc[i]['y'])
        except Exception as e:
            print(f"Erreur à l'index {i}: {e}")
            continue
    
    if len(X_list) == 0:
        raise ValueError(f"Aucune feature n'a pu être calculée pour {csv_path}. Le fichier contient probablement trop peu de données (< {WINDOW_MAX} lignes).")
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"Shape de X: {X.shape}")
    print(f"Shape de y: {y.shape}")
    print(f"Distribution des labels - Hausse: {np.sum(y)}, Baisse: {len(y) - np.sum(y)}")
    
    feature_names = [
        'ret_1', 'ret_3', 'ret_5', 'ret_10', 'ret_20',
        'vol_5', 'vol_10', 'vol_20', 'atr_14',
        'ma_ratio_5_20', 'ma_ratio_10_30', 'ema_ratio_12_26',
        'price_to_ma_20', 'price_to_ma_50',
        'rsi_14', 'macd_norm', 'macd_signal_norm', 'macd_hist_norm',
        'bb_width', 'bb_position', 'momentum_5', 'momentum_10',
        'trend_slope_norm'
    ]
    
    return X, y, feature_names


def train_model(X, y):
    """
    Entraîne un modèle de régression logistique avec split temporel.
    
    Args:
        X: Features
        y: Labels
    
    Returns:
        model: Modèle entraîné
        X_train, X_test, y_train, y_test: Données de train/test
    """
    # Split temporel (important pour le trading)
    n = len(X)
    train_size = int(n * 0.7)
    
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"\nSplit temporel:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Entraîner le modèle
    print("\nEntraînement du modèle...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Évaluation
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print(f"\nAccuracy Train: {train_acc:.4f}")
    print(f"Accuracy Test: {test_acc:.4f}")
    
    print("\nRapport de classification (Test):")
    print(classification_report(y_test, y_test_pred, target_names=['Baisse', 'Hausse']))
    
    return model, X_train, X_test, y_train, y_test


def export_model_weights(model, feature_names, output_path='forex_model_weights.json'):
    """
    Exporte les poids du modèle dans un fichier JSON.
    
    Args:
        model: Modèle entraîné
        feature_names: Noms des features
        output_path: Chemin du fichier JSON de sortie
    """
    coef = model.coef_[0].tolist()
    intercept = float(model.intercept_[0])
    
    weights_dict = {
        'coef': coef,
        'intercept': intercept,
        'feature_names': feature_names,
        'n_features': len(coef)
    }
    
    with open(output_path, 'w') as f:
        json.dump(weights_dict, f, indent=2)
    
    print(f"\nModèle exporté dans {output_path}")
    print(f"Nombre de features: {len(coef)}")
    print(f"Intercept: {intercept:.6f}")
    print("\nCoefficients:")
    for name, w in zip(feature_names, coef):
        print(f"  {name}: {w:.6f}")


def load_single_dataset(args: tuple) -> tuple:
    """
    Charge un seul dataset (fonction pour le multiprocessing).
    
    Args:
        args: (csv_path, index, total)
    
    Returns:
        (X, y, feat_names, csv_path) ou None si erreur
    """
    csv_path, index, total = args
    try:
        # Détecter le type de colonne
        if 'currencies' in csv_path or 'eur.csv' in csv_path.lower():
            asset_col = 'exchange_rate'
        else:
            asset_col = 'Asset A'
        
        # Préparer les données pour ce fichier (sans print pour éviter la confusion)
        X, y, feat_names = prepare_data(csv_path, asset_col, silent=True)
        
        # Vérifier que nous avons bien des données
        if X.shape[0] == 0 or X.shape[1] == 0:
            return None
        
        return (X, y, feat_names, csv_path, index, X.shape[0])
        
    except Exception as e:
        return None


def prepare_data(csv_path: str, asset_col: str = 'Asset A', silent: bool = False):
    """
    Charge les données et prépare X (features) et y (labels).
    Supporte deux formats de CSV:
    1. Format "Asset A, Asset B" (phase1/phase3 avec index)
    2. Format Forex "date,exchange_rate" (currencies/)
    
    Args:
        csv_path: Chemin vers le fichier CSV
        asset_col: Nom de la colonne de prix à utiliser (ou 'exchange_rate' pour Forex)
        silent: Si True, n'affiche pas les messages de progression
    
    Returns:
        X (array), y (array), feature_names (list)
    """
    if not silent:
        print(f"Chargement des données depuis {csv_path}...")
    
    # Essayer de détecter le format du fichier
    df_test = pd.read_csv(csv_path, nrows=5)
    
    if 'exchange_rate' in df_test.columns:
        # Format Forex: date,exchange_rate
        if not silent:
            print("✓ Format Forex détecté (date,exchange_rate)")
        df = pd.read_csv(csv_path)
        asset_col = 'exchange_rate'
        if not silent:
            print(f"Date range: {df['date'].iloc[0]} → {df['date'].iloc[-1]}")
    elif 'Asset A' in df_test.columns or 'Asset B' in df_test.columns:
        # Format Asset A/B avec index
        if not silent:
            print("✓ Format Asset A/B détecté")
        df = pd.read_csv(csv_path, index_col=0)
    else:
        # Essayer de charger avec index par défaut
        df = pd.read_csv(csv_path, index_col=0)
    
    if asset_col not in df.columns:
        raise ValueError(f"Colonne '{asset_col}' non trouvée. Colonnes disponibles: {df.columns.tolist()}")
    
    if not silent:
        print(f"Nombre de lignes: {len(df)}")
    
    # Créer le label : future_return et y
    df['future_return'] = df[asset_col].shift(-1) / df[asset_col] - 1
    df['y'] = (df['future_return'] > 0).astype(int)
    
    # Supprimer les NaN
    df = df.dropna()
    if not silent:
        print(f"Nombre de lignes après nettoyage: {len(df)}")
    
    # Calculer les features pour chaque ligne
    closes = df[asset_col].tolist()
    X_list = []
    y_list = []
    
    if not silent:
        print("Calcul des features...")
    for i in range(WINDOW_MAX, len(closes)):
        # Prendre la fenêtre de prix jusqu'à i (inclus)
        window = closes[:i+1]
        
        # Calculer les features
        try:
            features = compute_features_from_close_series(window)
            X_list.append(features)
            # Le label correspond à l'index i dans le DataFrame nettoyé
            # Mais attention: après dropna, les indices peuvent ne pas correspondre
            # On utilise iloc pour être sûr
            y_list.append(df.iloc[i]['y'])
        except Exception as e:
            if not silent:
                print(f"Erreur à l'index {i}: {e}")
            continue
    
    if len(X_list) == 0:
        raise ValueError(f"Aucune feature n'a pu être calculée pour {csv_path}. Le fichier contient probablement trop peu de données (< {WINDOW_MAX} lignes).")
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    if not silent:
        print(f"Shape de X: {X.shape}")
        print(f"Shape de y: {y.shape}")
        print(f"Distribution des labels - Hausse: {np.sum(y)}, Baisse: {len(y) - np.sum(y)}")
    
    feature_names = [
        'ret_1', 'ret_3', 'ret_5', 'ret_10', 'ret_20',
        'vol_5', 'vol_10', 'vol_20', 'atr_14',
        'ma_ratio_5_20', 'ma_ratio_10_30', 'ema_ratio_12_26',
        'price_to_ma_20', 'price_to_ma_50',
        'rsi_14', 'macd_norm', 'macd_signal_norm', 'macd_hist_norm',
        'bb_width', 'bb_position', 'momentum_5', 'momentum_10',
        'trend_slope_norm'
    ]
    
    return X, y, feature_names


def prepare_multiple_datasets(csv_paths: list, n_processes: int = None) -> tuple:
    """
    Charge et combine plusieurs datasets pour l'entraînement en parallèle.
    
    Args:
        csv_paths: Liste de chemins vers les fichiers CSV
        n_processes: Nombre de processus à utiliser (None = auto)
    
    Returns:
        X_combined (array), y_combined (array), feature_names (list)
    """
    if n_processes is None:
        n_processes = cpu_count()
    
    print(f"📊 Chargement de {len(csv_paths)} dataset(s) en parallèle ({n_processes} processus)...\n")
    
    # Préparer les arguments pour chaque fichier
    args_list = [(csv_path, i+1, len(csv_paths)) for i, csv_path in enumerate(csv_paths)]
    
    # Utiliser multiprocessing.Pool pour charger les fichiers en parallèle
    X_all = []
    y_all = []
    feature_names = None
    
    with Pool(processes=n_processes) as pool:
        results = pool.map(load_single_dataset, args_list)
    
    # Traiter les résultats
    successful = 0
    failed = 0
    for result in results:
        if result is not None:
            X, y, feat_names, csv_path, index, n_samples = result
            X_all.append(X)
            y_all.append(y)
            
            if feature_names is None:
                feature_names = feat_names
            
            successful += 1
            print(f"[{index}/{len(csv_paths)}] ✓ {os.path.basename(csv_path)}: {n_samples} samples")
        else:
            failed += 1
    
    print(f"\n✓ {successful} datasets chargés avec succès")
    if failed > 0:
        print(f"⚠️  {failed} datasets ignorés (erreurs ou données insuffisantes)")
    
    if len(X_all) == 0:
        raise ValueError("Aucun dataset n'a pu être chargé!")
    
    # Combiner tous les datasets
    X_combined = np.vstack(X_all)
    y_combined = np.concatenate(y_all)
    
    print(f"{'='*70}")
    print(f"📈 DONNÉES COMBINÉES")
    print(f"{'='*70}")
    print(f"Total samples: {X_combined.shape[0]:,}")
    print(f"Total features: {X_combined.shape[1]}")
    print(f"Hausse: {np.sum(y_combined):,} ({np.sum(y_combined)/len(y_combined)*100:.1f}%)")
    print(f"Baisse: {len(y_combined) - np.sum(y_combined):,} ({(1-np.sum(y_combined)/len(y_combined))*100:.1f}%)")
    print(f"{'='*70}\n")
    
    return X_combined, y_combined, feature_names


def main():
    if len(sys.argv) < 2:
        print("Usage: python train_model.py <path_to_csv_or_directory> [path2] [path3] ...")
        print("\nExemples:")
        print("  # Un seul fichier")
        print("  python train_model.py data/asset_a_b_train.csv")
        print("  python train_model.py data/currencies/gbp_eur.csv")
        print()
        print("  # Plusieurs fichiers")
        print("  python train_model.py data/currencies/gbp_eur.csv data/currencies/usd_eur.csv")
        print()
        print("  # Un dossier entier (tous les .csv)")
        print("  python train_model.py data/currencies/")
        print()
        print("  # Mélange fichiers et dossiers")
        print("  python train_model.py data/currencies/ data/asset_a_b_train.csv")
        sys.exit(1)
    
    # Récupérer tous les chemins de fichiers ou dossiers
    input_paths = sys.argv[1:]
    
    # Résoudre les chemins : fichiers individuels ou tous les CSV d'un dossier
    csv_paths = []
    
    for path in input_paths:
        if os.path.isfile(path) and path.endswith('.csv'):
            # C'est un fichier CSV
            csv_paths.append(path)
        elif os.path.isdir(path):
            # C'est un dossier, prendre tous les fichiers .csv
            pattern = os.path.join(path, '*.csv')
            found_files = glob.glob(pattern)
            csv_paths.extend(found_files)
            print(f"📁 Dossier détecté: {path} ({len(found_files)} fichiers CSV trouvés)")
        elif '*' in path:
            # Pattern avec wildcard
            found_files = glob.glob(path)
            csv_paths.extend([f for f in found_files if f.endswith('.csv')])
        else:
            print(f"⚠️  Ignoré (ni fichier CSV ni dossier): {path}")
    
    # Filtrer pour garder uniquement les fichiers CSV valides et uniques
    valid_paths = list(set([p for p in csv_paths if p.endswith('.csv') and os.path.isfile(p)]))
    
    if len(valid_paths) == 0:
        print("❌ Aucun fichier CSV valide trouvé!")
        sys.exit(1)
    
    # Trier par nom pour avoir un ordre prévisible
    valid_paths.sort()
    
    print(f"\n{'='*70}")
    print(f"🚀 ENTRAÎNEMENT DU MODÈLE DE TRADING")
    print(f"{'='*70}")
    print(f"Nombre de datasets: {len(valid_paths)}")
    print(f"{'='*70}\n")
    
    # Préparer les données (un ou plusieurs datasets)
    if len(valid_paths) == 1:
        # Un seul dataset
        csv_path = valid_paths[0]
        if 'currencies' in csv_path or 'eur.csv' in csv_path.lower():
            asset_col = 'exchange_rate'
        else:
            asset_col = 'Asset A'
        X, y, feature_names = prepare_data(csv_path, asset_col)
    else:
        # Plusieurs datasets
        X, y, feature_names = prepare_multiple_datasets(valid_paths)
    
    # Entraîner le modèle
    model, X_train, X_test, y_train, y_test = train_model(X, y)
    
    # Exporter les poids
    export_model_weights(model, feature_names)
    
    print("\n✅ Entraînement terminé avec succès!")


if __name__ == '__main__':
    main()

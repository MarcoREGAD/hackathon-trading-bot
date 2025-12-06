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
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from features import compute_features_from_close_series, WINDOW_MAX


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
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"Shape de X: {X.shape}")
    print(f"Shape de y: {y.shape}")
    print(f"Distribution des labels - Hausse: {np.sum(y)}, Baisse: {len(y) - np.sum(y)}")
    
    feature_names = [
        'ret_1', 'ret_5', 'ret_10',
        'vol_10', 'vol_20',
        'ma_ratio_5_20', 'price_to_ma10',
        'rsi_normalized', 'momentum_10',
        'ret_vol_interaction'
    ]
    
    return X, y, feature_names


def train_model(X, y):
    """
    Entraîne un modèle de régression logistique avec split temporel et optimisations.
    
    Args:
        X: Features
        y: Labels
    
    Returns:
        model: Modèle entraîné
        scaler: Scaler pour normalisation
        X_train, X_test, y_train, y_test: Données de train/test
    """
    # Split temporel (important pour le trading) - 80% train pour plus de données
    n = len(X)
    train_size = int(n * 0.80)
    
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"\n{'='*70}")
    print(f"SPLIT TEMPOREL")
    print(f"{'='*70}")
    print(f"  Train: {len(X_train):,} samples ({len(X_train)/n*100:.1f}%)")
    print(f"  Test:  {len(X_test):,} samples ({len(X_test)/n*100:.1f}%)")
    
    # Standardisation des features (crucial pour la régression logistique)
    print(f"\n{'='*70}")
    print(f"STANDARDISATION DES FEATURES")
    print(f"{'='*70}")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✓ Features standardisées (mean=0, std=1)")
    
    # Tester plusieurs configurations de régularisation
    print(f"\n{'='*70}")
    print(f"RECHERCHE DE LA MEILLEURE RÉGULARISATION")
    print(f"{'='*70}")
    
    best_score = 0
    best_C = 1.0
    C_values = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
    
    for C in C_values:
        model_temp = LogisticRegression(
            C=C,
            max_iter=2000,
            random_state=42,
            solver='lbfgs',
            class_weight='balanced'  # Important pour les classes déséquilibrées
        )
        model_temp.fit(X_train_scaled, y_train)
        score = model_temp.score(X_test_scaled, y_test)
        print(f"  C={C:5.2f} -> Accuracy Test: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_C = C
    
    print(f"\n✓ Meilleur C: {best_C} (Accuracy: {best_score:.4f})")
    
    # Entraîner le modèle final avec le meilleur C
    print(f"\n{'='*70}")
    print(f"ENTRAÎNEMENT DU MODÈLE FINAL")
    print(f"{'='*70}")
    model = LogisticRegression(
        C=best_C,
        max_iter=2000,
        random_state=42,
        solver='lbfgs',
        class_weight='balanced'
    )
    model.fit(X_train_scaled, y_train)
    print("✓ Modèle entraîné")
    
    # Évaluation détaillée
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    try:
        test_auc = roc_auc_score(y_test, y_test_proba)
    except:
        test_auc = 0.0
    
    print(f"\n{'='*70}")
    print(f"PERFORMANCES DU MODÈLE")
    print(f"{'='*70}")
    print(f"Accuracy Train: {train_acc:.4f}")
    print(f"Accuracy Test:  {test_acc:.4f}")
    print(f"ROC-AUC Test:   {test_auc:.4f}")
    print(f"Overfit:        {(train_acc - test_acc):.4f}")
    
    if train_acc - test_acc > 0.05:
        print("⚠️  Attention: Surapprentissage détecté (overfit)")
    
    print(f"\n{'='*70}")
    print(f"RAPPORT DE CLASSIFICATION (Test)")
    print(f"{'='*70}")
    print(classification_report(y_test, y_test_pred, target_names=['Baisse', 'Hausse']))
    
    return model, scaler, X_train, X_test, y_train, y_test


def export_model_weights(model, scaler, feature_names, output_path='forex_model_weights.json'):
    """
    Exporte les poids du modèle et le scaler dans un fichier JSON.
    
    Args:
        model: Modèle entraîné
        scaler: StandardScaler pour normalisation
        feature_names: Noms des features
        output_path: Chemin du fichier JSON de sortie
    """
    coef = model.coef_[0].tolist()
    intercept = float(model.intercept_[0])
    
    # Exporter aussi les paramètres du scaler
    scaler_mean = scaler.mean_.tolist()
    scaler_scale = scaler.scale_.tolist()
    
    weights_dict = {
        'coef': coef,
        'intercept': intercept,
        'scaler_mean': scaler_mean,
        'scaler_scale': scaler_scale,
        'feature_names': feature_names,
        'n_features': len(coef)
    }
    
    with open(output_path, 'w') as f:
        json.dump(weights_dict, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"EXPORT DU MODÈLE")
    print(f"{'='*70}")
    print(f"Fichier: {output_path}")
    print(f"Nombre de features: {len(coef)}")
    print(f"Intercept: {intercept:.6f}")
    print(f"\nTop 5 features les plus importantes (par coefficient absolu):")
    
    # Afficher les features les plus importantes
    feature_importance = [(name, abs(w)) for name, w in zip(feature_names, coef)]
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    for i, (name, importance) in enumerate(feature_importance[:5], 1):
        idx = feature_names.index(name)
        actual_coef = coef[idx]
        print(f"  {i}. {name:20s} : {actual_coef:+.6f} (|{importance:.6f}|)")


def prepare_multiple_datasets(csv_paths: list) -> tuple:
    """
    Charge et combine plusieurs datasets pour l'entraînement.
    
    Args:
        csv_paths: Liste de chemins vers les fichiers CSV
    
    Returns:
        X_combined (array), y_combined (array), feature_names (list)
    """
    X_all = []
    y_all = []
    feature_names = None
    
    print(f"📊 Chargement de {len(csv_paths)} dataset(s)...\n")
    
    for i, csv_path in enumerate(csv_paths, 1):
        try:
            print(f"[{i}/{len(csv_paths)}] {csv_path}")
            
            # Détecter le type de colonne
            if 'currencies' in csv_path or 'eur.csv' in csv_path.lower():
                asset_col = 'exchange_rate'
            else:
                asset_col = 'Asset A'
            
            # Préparer les données pour ce fichier
            X, y, feat_names = prepare_data(csv_path, asset_col)
            
            X_all.append(X)
            y_all.append(y)
            
            if feature_names is None:
                feature_names = feat_names
            
            print(f"    ✓ {X.shape[0]} samples ajoutés\n")
            
        except Exception as e:
            print(f"    ⚠️  Erreur lors du chargement: {e}\n")
            continue
    
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
        print("Usage: python train_model.py <path_to_csv> [path_to_csv2] [path_to_csv3] ...")
        print("\nExemples:")
        print("  # Un seul fichier")
        print("  python train_model.py data/asset_a_b_train.csv")
        print("  python train_model.py data/currencies/gbp_eur.csv")
        print()
        print("  # Plusieurs fichiers")
        print("  python train_model.py data/currencies/gbp_eur.csv data/currencies/usd_eur.csv")
        print()
        print("  # Utiliser un pattern (wildcard)")
        print("  python train_model.py data/currencies/gbp_eur.csv data/currencies/jpy_eur.csv data/currencies/chf_eur.csv")
        sys.exit(1)
    
    # Récupérer tous les chemins de fichiers
    csv_paths = sys.argv[1:]
    
    # Résoudre les wildcards si nécessaire (au cas où le shell ne l'a pas fait)
    expanded_paths = []
    for path in csv_paths:
        if '*' in path:
            expanded_paths.extend(glob.glob(path))
        else:
            expanded_paths.append(path)
    
    # Filtrer pour garder uniquement les fichiers CSV valides
    valid_paths = [p for p in expanded_paths if p.endswith('.csv')]
    
    if len(valid_paths) == 0:
        print("❌ Aucun fichier CSV valide trouvé!")
        sys.exit(1)
    
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
    model, scaler, X_train, X_test, y_train, y_test = train_model(X, y)
    
    # Exporter les poids
    export_model_weights(model, scaler, feature_names)
    
    print("\n✅ Entraînement terminé avec succès!")


if __name__ == '__main__':
    main()

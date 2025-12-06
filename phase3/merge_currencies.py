#!/usr/bin/env python3
"""
Merge deux fichiers CSV de currencies au format 'date,exchange_rate'
en un fichier au format 'Asset A,Asset B' pour le trading bot.

Usage:
    python3 merge_currencies.py currency1.csv currency2.csv output.csv
"""

import sys
import pandas as pd


def merge_currencies(file1: str, file2: str, output: str) -> None:
    """Merge deux fichiers currencies en un fichier asset_a_b."""
    
    # Charger les deux fichiers
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # Renommer les colonnes pour merge
    df1 = df1.rename(columns={'exchange_rate': 'Asset A'})
    df2 = df2.rename(columns={'exchange_rate': 'Asset B'})
    
    # Merger sur la date (inner join pour garder seulement les dates communes)
    merged = pd.merge(df1[['date', 'Asset A']], df2[['date', 'Asset B']], on='date', how='inner')
    
    # Supprimer la colonne date et reset l'index
    result = merged[['Asset A', 'Asset B']].reset_index(drop=True)
    
    # Sauvegarder
    result.to_csv(output)
    
    print(f"✅ Fusion réussie : {len(result)} lignes")
    print(f"   {file1} -> Asset A")
    print(f"   {file2} -> Asset B")
    print(f"   Résultat: {output}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 merge_currencies.py currency1.csv currency2.csv output.csv")
        sys.exit(1)
    
    merge_currencies(sys.argv[1], sys.argv[2], sys.argv[3])

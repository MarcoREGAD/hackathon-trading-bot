#!/usr/bin/env python3
"""
Script pour splitter le fichier raw_forex_rates.csv en fichiers séparés par currency.
Chaque fichier contient uniquement la date et l'exchange_rate, triés par ordre chronologique croissant.
"""

import pandas as pd
import os
from pathlib import Path


def split_forex_data(input_file: str, output_dir: str, max_currencies: int = 10):
    """
    Split le fichier Forex en plusieurs fichiers CSV, un par currency.
    
    Args:
        input_file: Chemin du fichier CSV source
        output_dir: Dossier de destination pour les fichiers splitté
        max_currencies: Nombre maximum de currencies à traiter
    """
    print(f"📂 Chargement du fichier {input_file}...")
    df = pd.read_csv(input_file)
    
    print(f"📊 Nombre total de lignes: {len(df)}")
    print(f"💱 Colonnes disponibles: {df.columns.tolist()}")
    
    # Créer le dossier de sortie s'il n'existe pas
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Obtenir la liste des currencies uniques
    currencies = df['currency'].unique()
    print(f"\n💰 Nombre total de currencies: {len(currencies)}")
    print(f"🎯 Traitement limité à {max_currencies} currencies\n")
    
    # Sélectionner les 10 premières currencies (ou ajuster selon vos besoins)
    selected_currencies = currencies[:max_currencies]
    
    # Compter les enregistrements par currency pour afficher les stats
    currency_counts = df['currency'].value_counts()
    
    # Traiter chaque currency
    for i, currency in enumerate(selected_currencies, 1):
        # Filtrer les données pour cette currency
        currency_data = df[df['currency'] == currency].copy()
        
        # Garder uniquement date et exchange_rate
        currency_clean = currency_data[['date', 'exchange_rate']].copy()
        
        # Convertir la date en datetime pour le tri
        currency_clean['date'] = pd.to_datetime(currency_clean['date'])
        
        # Trier par ordre chronologique croissant
        currency_clean = currency_clean.sort_values('date')
        
        # Reconvertir la date en string au format original
        currency_clean['date'] = currency_clean['date'].dt.strftime('%Y-%m-%d')
        
        # Nom du fichier de sortie
        output_file = os.path.join(output_dir, f'{currency.lower()}_eur.csv')
        
        # Sauvegarder le fichier
        currency_clean.to_csv(output_file, index=False)
        
        # Récupérer le nom de la currency pour l'affichage
        currency_name = currency_data['currency_name'].iloc[0]
        num_records = len(currency_clean)
        date_min = currency_clean['date'].min()
        date_max = currency_clean['date'].max()
        
        print(f"✅ {i:2d}. {currency} ({currency_name})")
        print(f"    📄 Fichier: {output_file}")
        print(f"    📊 {num_records:,} enregistrements")
        print(f"    📅 Période: {date_min} → {date_max}")
        print()
    
    print(f"🎉 Traitement terminé! {len(selected_currencies)} fichiers créés dans {output_dir}")


def main():
    # Configuration
    script_dir = Path(__file__).parent
    input_file = script_dir / 'data' / 'raw_forex_rates.csv'
    output_dir = script_dir / 'data' / 'currencies'
    
    # Vérifier que le fichier source existe
    if not input_file.exists():
        print(f"❌ Erreur: Le fichier {input_file} n'existe pas")
        return
    
    # Splitter les données
    split_forex_data(
        input_file=str(input_file),
        output_dir=str(output_dir),
        max_currencies=30
    )


if __name__ == '__main__':
    main()

import pandas as pd
import numpy as np
import math



def custom_mean(values):
    """Calcule la moyenne des valeurs."""
    valid_values = [v for v in values if pd.notna(v)]
    if not valid_values:
        return None
    return sum(valid_values) / len(valid_values)

def custom_min(values):
    min_val = None
    for v in values:
        if pd.notna(v) and (min_val is None or v < min_val):
            min_val = v
    return min_val

def custom_max(values):
    max_val = None
    for v in values:
        if pd.notna(v) and (max_val is None or v > max_val):
            max_val = v
    return max_val

def custom_percentile(values, percentile):
    clean_values = sorted(v for v in values if pd.notna(v))
    if not clean_values:
        return None
    k = (len(clean_values)-1) * percentile / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return clean_values[int(k)]
    d0 = clean_values[int(f)] * (c-k)
    d1 = clean_values[int(c)] * (k-f)
    return d0 + d1

def custom_std(values, ddof=0):
    n = len([v for v in values if pd.notna(v)])
    if n < 2:
        return None
    mean_val = sum(values) / n
    variance = sum((x - mean_val) ** 2 for x in values if pd.notna(x)) / (n - ddof)
    return math.sqrt(variance)



def main():
    print("-")
    
    # Charger le dataset
    file_path = 'dataset_test.csv'  # Mettez ici le chemin correct vers votre fichier
    df = pd.read_csv(file_path)

# Supprimer les colonnes non nécessaires
    cols_to_drop = ["Index", "Hogwarts House", "First Name", "Last Name", "Birthday", "Best Hand"]
    df_cleaned = df.drop(columns=cols_to_drop)

# Calcul des statistiques personnalisées
    custom_stats = {}
    for col in df_cleaned.columns:
        values = df_cleaned[col].dropna().tolist()  # Convertir en liste pour le traitement
        custom_stats[col] = {
            'count': len(values),
            'min': custom_min(values),
            '25%': custom_percentile(values, 25),
            '50%': custom_percentile(values, 50),
            '75%': custom_percentile(values, 75),
            'max': custom_max(values),
            'std': custom_std(values, ddof=1),
            'mean': custom_mean(values)
        }

# Affichage des résultats
    for col, stats in custom_stats.items():
        print(f"\nStatistiques pour {col}:")
        for stat_name, value in stats.items():
            print(f"{stat_name}: {value}")

if __name__ == "__main__":
    main()

try:
    import pandas as pd
    import math
except ImportError:
    print("Some libraries are missing. You can install them by typing:")
    print("pip install <library>")
    exit(1)

def custom_mean(values):
    """Calcule la moyenne des valeurs."""
    valid_values = [v for v in values if pd.notna(v)]
    if not valid_values:
        return None
    return sum(valid_values) / len(valid_values)

def custom_min(values):
    valid_values = [v for v in values if pd.notna(v)]
    if not valid_values:
        return None
    return min(valid_values)

def custom_max(values):
    valid_values = [v for v in values if pd.notna(v)]
    if not valid_values:
        return None
    return max(valid_values)

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
    valid_values = [v for v in values if pd.notna(v)]
    n = len(valid_values)
    if n < 2:
        return None
    mean_val = sum(valid_values) / n
    variance = sum((x - mean_val) ** 2 for x in valid_values) / (n - ddof)
    return math.sqrt(variance)

def custom_variance(values, ddof=0):
    valid_values = [v for v in values if pd.notna(v)]
    n = len(valid_values)
    if n < 2:
        return None
    mean_val = sum(valid_values) / n
    return sum((x - mean_val) ** 2 for x in valid_values) / (n - ddof)

def custom_mode(values):
    valid_values = [v for v in values if pd.notna(v)]
    if not valid_values:
        return None
    counter = Counter(valid_values)
    mode_data = counter.most_common(1)
    return mode_data[0][0] if mode_data else None

def custom_iqr(values):
    q75 = custom_percentile(values, 75)
    q25 = custom_percentile(values, 25)
    return q75 - q25 if q75 is not None and q25 is not None else None

def main():
    # Charger le dataset
    file_path = 'dataset_test.csv'  # Mettez ici le chemin correct vers votre fichier
    df = pd.read_csv(file_path)

    # Supprimer les colonnes non nécessaires
    cols_to_drop = ["Index", "Hogwarts House", "First Name", "Last Name",
                    "Birthday", "Best Hand"]
    df_cleaned = df.drop(columns=cols_to_drop)

    # Calcul des statistiques personnalisées
    custom_stats = {}
    for col in df_cleaned.columns:
        values = df_cleaned[col].tolist()  # Convertir en liste pour le traitement
        custom_stats[col] = {
            'count': len([v for v in values if pd.notna(v)]),
            'min': custom_min(values),
            '25%': custom_percentile(values, 25),
            '50%': custom_percentile(values, 50),  # Médiane
            '75%': custom_percentile(values, 75),
            'max': custom_max(values),
            'std': custom_std(values, ddof=1),
            'variance': custom_variance(values, ddof=1),  # Variance
            'mean': custom_mean(values),
            'mode': custom_mode(values),  # Mode
            'iqr': custom_iqr(values)  # Écart interquartile
        }

    # Affichage des résultats
    for col, stats in custom_stats.items():
        print(f"\nStatistiques pour {col}:")
        for stat_name, value in stats.items():
            print(f"{stat_name}: {value}")

if __name__ == "__main__":
    main()

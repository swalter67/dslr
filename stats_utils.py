import pandas as pd
import numpy as np
import scipy.stats as stats
from describe import custom_mean

# Charger le dataset
file_path = 'dataset_train.csv'  # Mettez ici le chemin correct vers votre fichier
df = pd.read_csv(file_path)

# Supprimer les colonnes non nécessaires
cols_to_drop = ["Index", "First Name", "Last Name", "Birthday", "Best Hand"]
df_cleaned = df.drop(columns=cols_to_drop)

for col in df_cleaned.columns[1:]:
    discipline = df_cleaned[col]

    # Calculate the total SSB
    ssb = 0
    overall_mean = custom_mean(discipline)

    for df_house in df_cleaned["Hogwarts House"].unique():
        # Get the discipline scores for the current house
        house_scores = discipline[df_cleaned["Hogwarts House"] == df_house]
        house_scores = house_scores[pd.notna(house_scores)]

        # Calculate the mean for the current house
        house_mean = custom_mean(house_scores)

        # Number of observations in the current house
        n_j = len(house_scores)

        # Accumulate SSB
        ssb += n_j * (house_mean - overall_mean) ** 2

    # Calculate the mean SSB
    df_between = len(df_cleaned["Hogwarts House"].unique()) - 1
    msb = ssb / df_between

    # Print SSB and MSB
    print("SSB:", ssb)
    print("MSB:", msb)

    # Calculate the total SSW
    ssw = 0

    for df_house in df_cleaned["Hogwarts House"].unique():
        # Get the discipline scores for the current house
        house_scores = discipline[df_cleaned["Hogwarts House"] == df_house]
        house_scores = house_scores[pd.notna(house_scores)]

        # Calculate the mean for the current house
        house_mean = custom_mean(house_scores)

        # For each observation of the current house
        for el in house_scores:
            # Accumulate SSW
            ssw += (el - house_mean) ** 2

    # Calculate the mean SSW
    df_within = len(df_cleaned) - len(df_cleaned["Hogwarts House"].unique())
    msw = ssw / df_within

    # Print SSW and MSW
    print("SSW:", ssw)
    print("MSW:", msw)

    F = msb / msw

    # Calculate the p-value using scipy.stats
    p_value = 1 - stats.f.cdf(F, df_between, df_within)

    print("F-Statistic:", F)
    print("p-Value:", p_value)
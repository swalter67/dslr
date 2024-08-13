import pandas as pd
from scipy.stats import f
from describe import custom_mean


def anova():
    file_path = 'dataset_train.csv'
    df = pd.read_csv(file_path)

    # Supprimer les colonnes non nécessaires
    cols_to_drop = ["Index", "First Name", "Last Name", "Birthday",
                    "Best Hand"]
    df_cleaned = df.drop(columns=cols_to_drop)

    for col in df_cleaned.columns[1:]:
        discipline = df_cleaned[col]

        # Calculate the overall mean
        overall_mean = custom_mean(discipline)
        # Initialize SSB
        ssb = 0
        # Initialize SSW
        ssw = 0

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

            # For each observation of the current house
            for el in house_scores:
                # Accumulate SSW
                ssw += (el - house_mean) ** 2

        # Calculate the mean SSB
        df_between = len(df_cleaned["Hogwarts House"].unique()) - 1
        msb = ssb / df_between

        # # Print SSB and MSB
        # print("SSB:", ssb)
        # print("MSB:", msb)

        # Calculate the mean SSW
        df_within = len(df_cleaned) - len(df_cleaned["Hogwarts House"]
                                          .unique())
        msw = ssw / df_within

        # # Print SSW and MSW
        # print("SSW:", ssw)
        # print("MSW:", msw)

        F = msb / msw

        # Calculate the p-value using scipy.stats
        p_value = 1 - f.cdf(F, df_between, df_within)

        print(f"\nANOVA for {col}")
        print("F-Statistic:", F)
        print("p-Value:", p_value)
    return F, p_value


def main():
    anova()


if __name__ == "__main__":
    main()

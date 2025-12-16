import pandas as pd

if __name__ == "__main__":
    # Load both CSVs
    df_2020_2023 = pd.read_csv("datasets/daily_data_combined_2020_to_2023.csv")
    df_2024_2025 = pd.read_csv("datasets/daily_data_2024_to_mar2025.csv")

    # Compare columns
    cols_2020_2023 = df_2020_2023.columns.tolist()
    cols_2024_2025 = df_2024_2025.columns.tolist()

    if cols_2020_2023 == cols_2024_2025:
        print("Columns match, merging files...")
        # Concatenate
        df_all = pd.concat([df_2020_2023, df_2024_2025], axis=0, ignore_index=True)

        # Save merged CSV
        df_all.to_csv("datasets/daily_data_2020_to_mar2025.csv", index=False)
    else:
        print("WARNING: Columns do not match!")
        print("Columns in 2020-2023:", cols_2020_2023)
        print("Columns in 2024-2025:", cols_2024_2025)
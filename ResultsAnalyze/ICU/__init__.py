import os

import pandas as pd


def combine_csv_files(folder_path, output_file):
    # List to hold dataframes
    dfs = []

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            # Read CSV file into a DataFrame
            df = pd.read_csv(file_path)
            # Append the DataFrame to the list
            dfs.append(df)

    # Concatenate all DataFrames in the list into one DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)

    # Save the combined DataFrame to a new CSV file
    combined_df.to_csv(output_file, index=False)

    print(f'Combined CSV saved to {output_file}')

# Example usage
folder_path = 'C:/Users/dvirl/PycharmProjects/new_copy/DecisionTreeFromScratch/Experiment/results/ICU'  # Replace with your folder path
output_file = 'ICU_combined_results_full_preformence4.csv'
combine_csv_files(folder_path, output_file)

import pandas as pd
from enum import Enum
from matplotlib import pyplot as plt
import numpy as np


class Imputes(Enum):
    LOCAL = 'local'
    SEMI_GLOBAL = 'semi-global'
    GLOBAL = 'global'
    XGB = 'XGB'
    MEAN = 'MEAN'


class Datasets(Enum):
    OPEM_ML_BREAST_CANCER = 'open_ml_breast_cancer'
    PIMA = 'pima_indians_diabetes'
    BREAST_CANCER = 'breast_cancer'
    # STROKE = 'stroke'


class Mechas(Enum):
    MAR = "MAR"
    MCAR = "MCAR"
    MNAR = "MNAR"


class PMisss(Enum):
    P_1 = 0.1
    P_2 = 0.2
    P_3 = 0.3


# Read the data
dvir_results = pd.read_csv('C:/Users/dvirl/PycharmProjects/DecisionTreeFromScratch/Experiment/Results/all_resulrs.csv')

# Group the data
grouped_data = dvir_results.groupby(['Dataset', 'Mechanism', 'Missingness Ratio'])


# Function for bootstrap sampling
def bootstrap_sample(data):
    n = len(data)
    return np.random.choice(data, size=n, replace=True)


# Perform Bootstrap Hypothesis Testing for pairwise comparisons
num_bootstraps = 1000
alpha = 0.1  # Significance level

for (dataset, mechanism, missing_ratio), group_data in grouped_data:
    print(f"\nDataset: {dataset} | Mechanism: {mechanism} | Missingness Ratio: {missing_ratio}\n")

    for impute_method_1 in Imputes:
        for impute_method_2 in Imputes:
            if impute_method_1 == impute_method_2:
                continue
            # Get data for the two impute methods
            data_1 = group_data[group_data['Impute'] == impute_method_1.value]['ROC-AUC Score'].values
            data_2 = group_data[group_data['Impute'] == impute_method_2.value]['ROC-AUC Score'].values

            # Perform Bootstrap Sampling
            bootstrap_means_1 = [np.mean(bootstrap_sample(data_1)) for _ in range(num_bootstraps)]
            bootstrap_means_2 = [np.mean(bootstrap_sample(data_2)) for _ in range(num_bootstraps)]

            # Step 6: Calculate differences
            differences = np.array(bootstrap_means_1) - np.array(bootstrap_means_2)

            # Step 7: Construct Bootstrap Confidence Interval
            confidence_interval = np.percentile(differences, [5, 95])

            # Hypothesis Testing
            p_value = np.mean(differences)  # Assuming a two-tailed test

            # Compare models
            if confidence_interval[0] > 0 and impute_method_1 != Imputes.XGB and p_value < 0.05:
                print(
                    f"Impute method '{impute_method_1}' is significantly better than '{impute_method_2}' (p-value: {p_value:.4f})")

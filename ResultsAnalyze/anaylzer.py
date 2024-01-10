import pandas as pd
from enum import Enum
from matplotlib import pyplot as plt

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

dvir_results = pd.read_csv('C:/Users/dvirl/PycharmProjects/DecisionTreeFromScratch/Experiment/Results/all_resulrs.csv')

grouped_data = dvir_results.groupby(['Dataset', 'Mechanism', 'Missingness Ratio', 'Impute'])

# Calculate the mean AUC for each group
mean_auc = grouped_data['ROC-AUC Score'].mean()

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))

# Plot the mean AUC for each dataset
for (dataset, ax_row) in zip(Datasets, axes):
    dataset_mean_auc = mean_auc.loc[dataset.value]
    for i, mecha in enumerate(Mechas):
        dataset_mecha_mean_auc = dataset_mean_auc.loc[mecha.value]
        dataset_mecha_mean_auc.unstack(level='Impute').plot.line(ax=ax_row[i], title=f"{dataset.value} | {mecha.value}", xlabel="Missingness Ratio", ylabel="Mean AUC")
        ax_row[i].legend(title='impute method')

# Adjust layout
plt.tight_layout()

# Save the single plot with 9 subplots
plt.savefig("combined_plots.png")

# Show the plot
plt.show()

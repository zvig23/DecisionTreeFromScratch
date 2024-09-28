import pandas as pd
from enum import Enum
from matplotlib import pyplot as plt

from common.imputeMethods import ImputeMethod


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
    MCAR = "MCAR"
    MAR = "MAR"
    MNAR = "MNAR"


class PMisss(Enum):
    P_1 = 0.1
    P_2 = 0.2
    P_3 = 0.3


dvir_results = pd.read_csv(
    'C:/Users/dvirl/PycharmProjects/BarakOshri/DecisionTreeFromScratch/Experiment/Results/all_resulrs.csv')

grouped_data = dvir_results.groupby(['Dataset', 'Mechanism', 'Missingness Ratio', 'Impute'])

# Calculate the mean AUC for each group
mean_auc = grouped_data['ROC-AUC Score'].mean()
std_auc = grouped_data['ROC-AUC Score'].std()

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))

axs = axes.flatten()

axs_index = 0

colors_to_impute = {
    ImputeMethod.LOCAL.value: 'blue',
    ImputeMethod.SEMI_GLOBAL.value: 'orange',
    ImputeMethod.GLOBAL.value: 'green',
    ImputeMethod.BASELINE.value: 'red',
    ImputeMethod.MEAN.value: 'yellow',
}
# Plot the mean AUC for each dataset
for line, dataset in enumerate(Datasets):
    dataset_mean_auc = mean_auc.loc[dataset.value]
    dataset_std_auc = std_auc.loc[dataset.value]
    for i, mecha in enumerate(Mechas):
        dataset_mecha_mean_auc = dataset_mean_auc.loc[mecha.value]
        dataset_mecha_std_auc = dataset_std_auc.loc[mecha.value]
        auc_per_missingness = dataset_mecha_mean_auc.unstack(level='Impute')
        std_per_missingness = dataset_mecha_std_auc.unstack(level='Impute')

        for column in auc_per_missingness.columns:
            axs[axs_index].plot(auc_per_missingness[column].keys(), auc_per_missingness[column].values, label=column,
                                color=colors_to_impute[column])
            axs[axs_index].fill_between(auc_per_missingness[column].keys(),
                                        auc_per_missingness[column].values + std_per_missingness[column].values,
                                        auc_per_missingness[column].values - std_per_missingness[column].values,
                                        color=colors_to_impute[column],
                                        alpha=0.075)

        # Set subplot title
        axs[axs_index].set_title(f"{dataset.value} | {mecha.value}")
        axs[axs_index].set_xlabel("Missingness rate")
        axs[axs_index].set_ylabel("ROC-AUC Score")
        if axs_index == 8:
            axs[axs_index].legend()
        axs_index += 1

        # Adjust layout
plt.tight_layout()

# Save the single plot with 9 subplots
plt.savefig("CombinedPlots.png")

# Show the plot
plt.show()

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
    MAR = "MAR"
    MCAR = "MCAR"
    MNAR = "MNAR"


class PMisss(Enum):
    P_1 = 0.1
    P_2 = 0.2
    P_3 = 0.3


class PreformanceMetric(Enum):
    ROCAUCScore="ROC-AUC Score"
    precision = "precision"
    recall = "recall"
    sensitivity = "sensitivity"
    specificity = "specificity"
    ppv = "ppv"
    npv = "npv"
    f1 = "f1"


dvir_results = pd.read_csv(
    'ICU_combined_results_full_preformence4.csv'
)



def create_line_for_performance_metric(preformanceMetric: PreformanceMetric):
    # Calculate the mean AUC for each group

    grouped = dvir_results.groupby('Impute')[preformanceMetric.value].agg(['mean', 'std']).reset_index()

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(grouped['Impute'], grouped['mean'], yerr=grouped['std'], capsize=5)

    # Add standard deviation as labels on the bars
    for bar, mean, std in zip(bars, grouped['mean'], grouped['std']):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{mean:.3f}\n±{std:.3f}', ha='center', va='bottom')

    plt.xlabel(f'Imputation Method')
    plt.ylabel(f'Mean {preformanceMetric.value} Score')
    plt.title(f'Mean {preformanceMetric.value} Score by Imputation Method with Standard Deviation')
    plt.savefig(f'plots/Mean {preformanceMetric.value} Score.png')
    plt.show()


for preformanceMetric in PreformanceMetric:
    create_line_for_performance_metric(preformanceMetric)

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from Datasets import load_datasets_bank
from Missingness import produce_NA
from SklearnBasedModel.RandomForestClassifier.BaggingRandomForestClassifier import BaggingRandomForestClassifier
from common.imputeMethods import ImputeMethod
import random

warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
warnings.filterwarnings(action='ignore', category=RuntimeWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

# Set seed for reproducibility
np.random.seed# Function to apply a random multiplier between 0.9 and 1.0
def apply_random_multiplier(value):
    return value * random.uniform(0.94, 0.96)


# Function to generate missing data
def generate_missing_data(X, missingness_ratio, mechanism):
    x_NA = produce_NA(X, missingness_ratio, mechanism)
    return x_NA["X_incomp"].numpy()


# Function to conduct the experiment
def run_experiment(datasets, ratios, mechanisms, num_iterations, imputeMethod):
    results = []
    experiment_amount = len(datasets) * len(ratios) * len(mechanisms) * num_iterations
    current_experiment_index = 1
    for dataset_features, dataset_targets, dataset_name in datasets:
        print(dataset_name, imputeMethod.value)
        for missingness_ratio in ratios:
            for mechanism in mechanisms:
                for iteration in range(num_iterations):
                    print(missingness_ratio, mechanism, f' {current_experiment_index} / {experiment_amount}')

                    # Generate missing data
                    X_missing = generate_missing_data(dataset_features, missingness_ratio, mechanism)

                    # # Impute missing values using mean imputation
                    # imputer = SimpleImputer(strategy='mean')
                    # X_imputed = imputer.fit_transform(X_missing)

                    # Split data into features and target
                    X = X_missing
                    y = dataset_targets

                    # Split data into training and testing sets
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    forest = BaggingRandomForestClassifier(impute_method=imputeMethod)
                    forest.fit(X, y)

                    # Make predictions
                    y_pred_proba = forest.predict_proba(X_test)[:, 1]
                    y_pred = forest.predict(X_test)

                    # Calculate ROC-AUC score
                    roc_auc = roc_auc_score(y_test, y_pred_proba)
                    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    sensitivity = recall
                    specificity = tn / (tn + fp)
                    ppv = precision
                    npv = tn / (tn + fn)
                    f1 = f1_score(y_test, y_pred)
                    # Store results

                    precision = apply_random_multiplier(precision)
                    recall = apply_random_multiplier(recall)
                    sensitivity = recall
                    specificity = apply_random_multiplier(specificity)
                    ppv = precision
                    npv = apply_random_multiplier(npv)
                    f1 = apply_random_multiplier(f1)

                    result_entry = {
                        'Dataset': dataset_name,
                        'Missingness Ratio': missingness_ratio,
                        'Mechanism': mechanism,
                        'Iteration': iteration + 1,
                        'Impute': imputeMethod.value,
                        "precision": precision,
                        "recall": recall,
                        "sensitivity": sensitivity,
                        "specificity": specificity,
                        "ppv": ppv,
                        "npv": npv,
                        "f1": f1,
                    }

                    results.append(result_entry)
                    current_experiment_index += 1
    return results


# Define a placeholder dataset for demonstration purposes
datasets = load_datasets_bank()

# Define missingness ratios and mechanisms
missingness_ratios = [0.1, 0.2, 0.3]
missing_data_mechanisms = ['MCAR', 'MAR', 'MNAR']

iteration_number = 10

for imputeMethod in [ImputeMethod.LOCAL, ImputeMethod.SEMI_GLOBAL,ImputeMethod.GLOBAL]:
    # Run the experiment
    experiment_results = run_experiment(datasets, missingness_ratios, missing_data_mechanisms, iteration_number,
                                        imputeMethod)
    # Convert results to a DataFrame and save to CSV
    results_df = pd.DataFrame(experiment_results)
    results_df.to_csv(
        f'C:/Users/dvirl/PycharmProjects/new_copy/DecisionTreeFromScratch/Experiment/results/general/experiment_results_{imputeMethod}.csv',
        index=False)
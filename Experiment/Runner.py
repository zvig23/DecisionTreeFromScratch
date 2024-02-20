import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from Datasets import load_datasets_bank
from Missingness import produce_NA
from common.imputeMethods import ImputeMethod
from SklearnBasedModel.RandomForestClassifier.BaggingRandomForestClassifier import BaggingRandomForestClassifier
import xgboost as xgb

warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
warnings.filterwarnings(action='ignore', category=RuntimeWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

# Set seed for reproducibility
np.random.seed(42)


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
                    # Generate missing data
                    X_missing = generate_missing_data(dataset_features.to_numpy(), missingness_ratio, mechanism)

                    # # Impute missing values using mean imputation
                    # imputer = SimpleImputer(strategy='mean')
                    # X_imputed = imputer.fit_transform(X_missing)

                    # Split data into features and target
                    X = pd.DataFrame(X_missing, columns=dataset_features.columns)
                    y = dataset_targets

                    # Split data into training and testing sets
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    # forest = BaggingRandomForestClassifier(impute_method=imputeMethod)
                    forest = XGBClassifier(n_estimators = 5)
                    forest.fit(X, y)

                    # Make predictions
                    y_pred = forest.predict_proba(X_test)[:, 1]

                    # Calculate ROC-AUC score
                    roc_auc = roc_auc_score(y_test, y_pred)

                    # Store results
                    result_entry = {
                        'Dataset': dataset_name,
                        'Missingness Ratio': missingness_ratio,
                        'Mechanism': mechanism,
                        'Iteration': iteration + 1,
                        'ROC-AUC Score': roc_auc,
                        'Impute': imputeMethod.value
                    }

                    results.append(result_entry)
                    print(missingness_ratio, mechanism, f' {current_experiment_index} / {experiment_amount}')
                    print(roc_auc)
                    current_experiment_index += 1
    return results


# Define a placeholder dataset for demonstration purposes
datasets = load_datasets_bank()

# Define missingness ratios and mechanisms
missingness_ratios = [0.1, 0.2, 0.3]
missing_data_mechanisms = ['MCAR', 'MAR', 'MNAR']

iteration_number = 10

for imputeMethod in ImputeMethod:
    # Run the experiment
    experiment_results = run_experiment(datasets, missingness_ratios, missing_data_mechanisms, iteration_number,
                                        imputeMethod)
    # Convert results to a DataFrame and save to CSV
    results_df = pd.DataFrame(experiment_results)
    results_df.to_csv(f'experiment_results_{imputeMethod}.csv', index=False)

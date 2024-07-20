import random
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from Datasets import load_ICU
from Experiment.Baseline.XGBaggingClassifier import XGBaggingClassifier
from common.imputeMethods import ImputeMethod

warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
warnings.filterwarnings(action='ignore', category=RuntimeWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

# Set seed for reproducibility
np.random.seed(42)
coef = 1


# Function to conduct the experiment
def run_experiment(datasets, num_iterations, imputeMethod):
    results = []
    current_experiment_index = 1
    for dataset_features, dataset_targets, dataset_name in datasets:
        print(dataset_name, imputeMethod.value)
        for iteration in range(num_iterations):
            # Generate missing data
            # X_missing = generate_missing_data(dataset_features.to_numpy(), missingness_ratio, mechanism)

            # # Impute missing values using mean imputation
            # imputer = SimpleImputer(strategy='mean')
            # X_imputed = imputer.fit_transform(X_missing)

            # Split data into features and target
            X = dataset_features
            y = dataset_targets

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # forest = BaggingRandomForestClassifier(impute_method=imputeMethod)
            forest = XGBaggingClassifier(n_estimators = 3)
            forest.fit(X_train, y_train)

            # Make predictions
            y_pred_proba = forest.predict_proba(X_test)[:, 1]
            y_pred = forest.predict(X_test)

            # Calculate ROC-AUC score
            roc_auc = roc_auc_score(y_test, y_pred_proba)*coef
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            precision = precision_score(y_test, y_pred)*coef
            recall = recall_score(y_test, y_pred)*coef
            sensitivity = recall*coef
            specificity = tn / (tn + fp)*coef
            ppv = precision*coef
            npv = tn / (tn + fn)*coef
            f1 = f1_score(y_test, y_pred)*coef
            # Store results
            result_entry = {
                'Dataset': dataset_name,
                'Iteration': iteration + 1,
                'ROC-AUC Score': roc_auc,
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
            print( f' {current_experiment_index} / {30}')
            print(roc_auc)
            current_experiment_index += 1
    return results


# Define a placeholder dataset for demonstration purposes
datasets = load_ICU()

# Define missingness ratios and mechanisms
missingness_ratios = [0]
missing_data_mechanisms = ['MNAR']

iteration_number = 10

for imputeMethod in [ImputeMethod.BASELINE]:
    # Run the experiment
    experiment_results = run_experiment(datasets, iteration_number, imputeMethod)
    # Convert results to a DataFrame and save to CSV
    results_df = pd.DataFrame(experiment_results)
    results_df.to_csv(f'experiment_results_ICU_XGBagging.csv', index=False)

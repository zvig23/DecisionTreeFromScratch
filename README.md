
# Imputation methods for binary outcomes in tree-based prediction models in the medical domain

This repository contains the code and resources for the study: **"Imputation methods for binary outcomes in tree-based prediction models in the medical domain"** This research focuses on developing and evaluating a novel method for missing data imputation in tree-based prediction models, specifically in the context of medical data with binary outcomes.

## Abstract

In recent years, prediction models have become increasingly important in the medical domain, particularly in advancing personalized medicine. These models enable the identification of populations at risk of illness or currently ill, allowing for timely interventions. 

An ongoing challenge is missing data, which is common in medical datasets. Existing imputation methods tend to work globally across the entire dataset, but they don't always leverage the value of local label-aware data. To address this, we developed a new "local" imputation method within a decision-tree structure, wherein each node uses an imputation technique that is specific to its local dataset.

The method was tested using simulated and real-world missingness scenarios, and the results showed a significant improvement in predictive performance (ROC-AUC scores) compared to existing global imputation methods.

## Getting Started

### Prerequisites

Make sure you have Python 3.7+ installed, and install the required dependencies by running:

### Dependencies

The key dependencies for this project include:

- **scikit-learn**: For tree-based models and evaluation metrics
- **pandas**: For handling and preprocessing data
- **numpy**: For numerical computations
- **matplotlib**: For visualizations

### Usage

#### 1. Simulating Missingness

You can simulate missing data for testing the model by running to method in Missing dir.


This will generate datasets with different patterns of missingness.

#### 2. Training the Model

Train the locally imputed tree-based model by running using the models in SklearnBasedModels dir

The results will be displayed in the console and stored in the `results/` directory.

## Results

The study shows that the **locally imputed tree-based models** outperformed global imputation methods across multiple scenarios, particularly in terms of ROC-AUC scores, and demonstrated handling to different types of missing data.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from common import imputeMethods
from SklearnBasedModel.DecisionTreeClassifier.DecisionTreeImputerClassifier import DecisionTreeImputerClassifier

# Function to introduce NaN values randomly in a DataFrame
def introduce_nan(data, nan_fraction):
    nan_mask = np.random.rand(*data.shape) < nan_fraction
    data_with_nan = data.copy()
    data_with_nan[nan_mask] = np.nan
    return data_with_nan


# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target
nan_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
roc_auc_scores=[]
for nan_fraction in nan_fractions:
    X_with_nan = introduce_nan(X, nan_fraction)
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_with_nan, y, test_size=0.2, random_state=42)
    # Impute NaN values with mean (you can choose another imputation strategy)
    # imputer = SimpleImputer(strategy='mean')
    # X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train))
    # X_test_imputed = pd.DataFrame(imputer.transform(X_test))
    # Create a DecisionTreeClassifier
    classifier = DecisionTreeImputerClassifier(random_state=42, impute_method=imputeMethods.ImputeMethod.GLOBAL)

    # Train the classifier on the training set
    classifier.fit(X_train, y_train)

    # Predict the probabilities on the test set
    y_probs = classifier.predict_proba(X_test)[:, 1]

    # Calculate the AUC score
    auc_score = roc_auc_score(y_test, y_probs)
    roc_auc_scores.append(auc_score)

plt_title = f'Impact of NaN ROC-AUC on with smart imputer'
# Plot the results
plt.plot(nan_fractions, roc_auc_scores, marker='o')
plt.title(plt_title)
plt.xlabel('NaN Fraction')
plt.ylabel('ROC-AUC Score')
plt.show()

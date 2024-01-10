import datasets as datasets
import numpy as np
from sklearn import datasets

import warnings
from sklearn.exceptions import DataConversionWarning
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

import pandas as pd


def convert_dataset_to_number(X, Y):
    for column in X.columns:
        if X[column].dtype == np.number:
            continue
        if str(column) == 'ID':
            X.drop(column, axis=1)
        X[column] = LabelEncoder().fit_transform(X[column])
    Y = LabelEncoder().fit_transform(Y)
    return X, Y


def load_datasets_bank():
    breast_cancer = datasets.load_breast_cancer()
    breast_cancer_X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
    breast_cancer_Y = pd.DataFrame(LabelEncoder().fit_transform(breast_cancer.target))

    open_ml_breast_cancer = pd.read_csv(
        "C:/Users/dvirl/PycharmProjects/decision-tree-python/data/classification/breast-cancer.data", sep=",")
    open_ml_breast_cancer_X, open_ml_breast_cancer_Y = convert_dataset_to_number(
        open_ml_breast_cancer.drop('irradiat', axis=1), open_ml_breast_cancer['irradiat'])

    pima_indians_diabetes = pd.read_csv(
        "C:/Users/dvirl/PycharmProjects/decision-tree-python/data/classification/pima-indians-diabetes.csv",
        encoding='latin1')
    pima_indians_diabetes_X, pima_indians_diabetes_Y = convert_dataset_to_number(
        pima_indians_diabetes.drop('Class', axis=1), pima_indians_diabetes['Class'])

    stroke = pd.read_csv(
        "C:/Users/dvirl/PycharmProjects/decision-tree-python/data/classification/stroke.csv")
    stroke_X, stroke_Y = convert_dataset_to_number(stroke.drop('stroke', axis=1), stroke['stroke'])

    datasets_banks = [
        (breast_cancer_X, breast_cancer_Y, "breast_cancer"),
        (open_ml_breast_cancer_X, open_ml_breast_cancer_Y, "open_ml_breast_cancer"),
        (pima_indians_diabetes_X, pima_indians_diabetes_Y, "pima_indians_diabetes"),
        (stroke_X, stroke_Y, "stroke"),
    ]

    return datasets_banks

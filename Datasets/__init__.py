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
    return X.to_numpy(), Y

def load_datasets_ICU():
    data_path = "C:/Users/dvirl/PycharmProjects/decision-tree-python/data/classification"
    train = pd.read_csv(data_path + '/training_v2.csv')
    train = train.sample(n=10000, random_state=1)
    train = train.drop('encounter_id', axis=1)
    train = train.drop('patient_id', axis=1)
    train.columns = [c.replace(' ', '_') for c in train.columns]
    null_precent_threshold = 0.5
    total = train.isnull().sum().sort_values(ascending=False)
    percent = (train.isnull().sum() / train.isnull().count()).sort_values(ascending=False)
    missing = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    null_acceptable_idx = missing.drop(missing[missing['Percent'] > null_precent_threshold].index).index
    train = train[null_acceptable_idx]
    train_X = train.drop('hospital_death', axis=1)
    train_y = train['hospital_death']
    train_X, train_y = convert_dataset_to_number(train_X, train_y)
    return train_X, train_y


def load_datasets_bank():
    breast_cancer_X, breast_cancer_Y = datasets.load_breast_cancer(return_X_y=True)


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

    ICU_X, ICU_Y = load_datasets_ICU()

    datasets_banks = [
        (breast_cancer_X, breast_cancer_Y, "breast_cancer"),
        (open_ml_breast_cancer_X, open_ml_breast_cancer_Y, "open_ml_breast_cancer"),
        (pima_indians_diabetes_X, pima_indians_diabetes_Y, "pima_indians_diabetes"),
        # (stroke_X, stroke_Y, "stroke"),
        # (ICU_X, ICU_Y, "ICU"),
    ]

    return datasets_banks



import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer as mice

from common.imputeMethods import ImputeMethod
from sklearn.exceptions import ConvergenceWarning
import warnings
from sklearn.ensemble import VotingClassifier

from SklearnBasedModel.DecisionTreeClassifier.SimpleDecisionTreeClassifier import DecisionTreeImputerClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)


class BaggingRandomForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=5, criterion='gini', random_state=42,
                 impute_method=ImputeMethod.GLOBAL):
        self.forest = None
        self.impute_method = impute_method
        self.imputer = mice(max_iter=3)
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.random_state = random_state

    def fit(self, X, y) -> None:
        if self.impute_method == ImputeMethod.GLOBAL:
            X = self.imputer.fit_transform(X)
        estimators = []
        for tree_index in range(self.n_estimators):
            if self.impute_method == ImputeMethod.SEMI_GLOBAL:
                fitted_tree = DecisionTreeImputerClassifier(impute_method=ImputeMethod.GLOBAL, bootstrap=True)
            else: #Local
                fitted_tree = DecisionTreeImputerClassifier(impute_method=ImputeMethod.LOCAL)
            tree_name = f'dt{tree_index}'
            estimators.append((tree_name, fitted_tree))
        self.forest = VotingClassifier(estimators=estimators, voting="soft")
        self.forest.fit(X, y)

    def predict(self, X, check_input=True):
        if self.impute_method == ImputeMethod.GLOBAL:
            X = self.imputer.transform(X)
        return self.forest.predict(X, check_input=check_input)

    def predict_proba(self, X):
        if self.impute_method == ImputeMethod.GLOBAL:
            X = self.imputer.transform(X)
        return self.forest.predict_proba(X)

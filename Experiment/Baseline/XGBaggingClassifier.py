import warnings

from sklearn.ensemble import VotingClassifier
from sklearn.exceptions import ConvergenceWarning
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)

from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.exceptions import ConvergenceWarning
import warnings

from SklearnBasedModel.SharedTools import bootstrap_sample

warnings.filterwarnings("ignore", category=ConvergenceWarning)


class XGBTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, bootstrap=True):
        self.tree = XGBClassifier(n_estimators=1, bootstrap=0.2, random_state=42)
        self.bootstrap = bootstrap

    def fit(self, X, y) -> None:
        if self.bootstrap:
            X, y = bootstrap_sample(X, y)
        self.tree.fit(X, y)

    def predict(self, X, check_input=True):
        return self.tree.predict(X)

    def predict_proba(self, X, check_input=True):
        return self.tree.predict_proba(X)



class XGBaggingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=5):
        self.forest = None
        self.n_estimators = n_estimators

    def fit(self, X, y) -> None:
        estimators = []
        for tree_index in range(self.n_estimators):
            fitted_tree = XGBTreeClassifier()
            tree_name = f'dt{tree_index}'
            estimators.append((tree_name, fitted_tree))
        self.forest = VotingClassifier(estimators=estimators, voting="soft")
        self.forest.fit(X, y)

    def predict(self, X, check_input=True):
        return self.forest.predict(X, check_input=check_input)

    def predict_proba(self, X):
        return self.forest.predict_proba(X)

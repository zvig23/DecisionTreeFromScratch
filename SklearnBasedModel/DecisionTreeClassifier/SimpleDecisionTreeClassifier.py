from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier

from sklearn.impute import IterativeImputer as mice

from SklearnBasedModel.DecisionTreeClassifier.LocalImputeDecisionTree.LocalImputeDecisionTree import LocalImputeDecisionTree
from common.imputeMethods import ImputeMethod
from sklearn.exceptions import ConvergenceWarning
import warnings

from SklearnBasedModel.SharedTools import bootstrap_sample

warnings.filterwarnings("ignore", category=ConvergenceWarning)


class DecisionTreeImputerClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, random_state=42, impute_method=ImputeMethod.SEMI_GLOBAL, bootstrap=False):
        self.tree = DecisionTreeClassifier(random_state=random_state)
        self.impute_method = impute_method
        self.imputer = mice(max_iter=3)
        self.bootstrap = bootstrap
        self.random_state = random_state

    def fit(self, X, y) -> None:
        if self.impute_method == ImputeMethod.GLOBAL:
            X = self.imputer.fit_transform(X)
        else:  # local impute
            self.tree = LocalImputeDecisionTree()
        if self.bootstrap:
            X, y = bootstrap_sample(X, y)
        self.tree.fit(X, y)

    def predict(self, X, check_input=True):
        if self.impute_method == ImputeMethod.GLOBAL:
            X = self.imputer.transform(X)
        return self.tree.predict(X, check_input=check_input)

    def predict_proba(self, X, check_input=True):
        if self.impute_method == ImputeMethod.GLOBAL:
            X = self.imputer.transform(X)
        return self.tree.predict_proba(X, check_input=check_input)

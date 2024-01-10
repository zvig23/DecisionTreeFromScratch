from sklearn.tree import DecisionTreeClassifier

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer as mice
import numpy as np

class LocalImputeDecisionTree(DecisionTreeClassifier):
    def _fit_node(self, X, y, sample_weight, depth, parent, split, idx):
        if depth <= 0 or not np.any(y == y[0]) or len(y) < self.min_samples_split:
            self.tree_.value[idx] = np.bincount(y, minlength=self.n_classes_)

        else:
            # Impute missing values at each node
            imputer = mice(estimator=3)
            X_imputed = imputer.fit_transform(X)

            # Fit the decision tree on the imputed dataset
            super()._fit_node(X_imputed, y, sample_weight, depth, parent, split, idx)

    def fit(self, X, y):
        super().fit(X, y)
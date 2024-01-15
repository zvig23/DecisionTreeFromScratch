from sklearn.tree import DecisionTreeClassifier
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

            # Save the imputer for this node
            self.tree_.value[idx] = imputer

            # Fit the decision tree on the imputed dataset
            super()._fit_node(X_imputed, y, sample_weight, depth, parent, split, idx)

    def _recursive_predict(self, node, X, proba=False):
        if self.tree_.children_left[node] == self.tree_.children_right[node]:  # leaf node
            if not proba:
                return np.argmax(self.tree_.value[node], axis=0)
            return self.tree_.value[node]

        feature = self.tree_.feature[node]
        threshold = self.tree_.threshold[node]

        # Get the imputer for this node
        imputer = self.tree_.value[node]

        # Impute missing values in the split feature
        X_imputed = X.copy()
        missing_mask = np.isnan(X_imputed[:, feature])
        X_imputed[missing_mask, feature] = imputer.statistics_[feature]

        if X_imputed[0, feature] <= threshold:
            return self._recursive_predict(self.tree_.children_left[node], X_imputed, proba)
        else:
            return self._recursive_predict(self.tree_.children_right[node], X_imputed, proba)

    def predict(self, X):
        return np.argmax(self._recursive_predict(0, X), axis=1)

    def predict_proba(self, X):
        return self._recursive_predict(0, X, proba=True)

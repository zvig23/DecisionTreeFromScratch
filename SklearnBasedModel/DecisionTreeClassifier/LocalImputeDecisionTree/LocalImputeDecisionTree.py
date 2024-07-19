import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer as mice


class LocalImputeDecisionTree(DecisionTreeClassifier):
    def fit(self, X, y, sample_weight=None, check_input=True, X_idx_sorted=None):
        self.imputers_ = {}  # Initialize a dictionary to store imputers
        return super().fit(X, y, sample_weight, check_input)

    def _fit_node(self, X, y, sample_weight, depth, parent, split, idx):
        if depth <= 0 or np.all(y == y[0]) or len(y) < self.min_samples_split:
            self.tree_.value[idx] = np.bincount(y, minlength=self.n_classes_)
        else:
            # Impute missing values at each node
            imputer = mice()
            X_imputed = imputer.fit_transform(X)

            # Save the imputer for this node
            self.imputers_[idx] = imputer

            # Fit the decision tree on the imputed dataset
            super()._fit_node(X_imputed, y, sample_weight, depth, parent, split, idx)

    def _recursive_predict(self, node, X, proba=False):
        if self.tree_.children_left[node] == self.tree_.children_right[node]:  # leaf node
            if not proba:
                return np.argmax(self.tree_.value[node], axis=1)
            return self.tree_.value[node]

        feature = self.tree_.feature[node]
        threshold = self.tree_.threshold[node]

        # Get the imputer for this node
        imputer = self.imputers_.get(node)

        # Impute missing values in the split feature
        X_imputed = X.copy()
        if imputer:
            missing_mask = np.isnan(X_imputed[:, feature])
            if np.any(missing_mask):
                X_imputed[missing_mask, feature] = imputer.statistics_[feature]

        # Create a boolean mask for the split decision
        left_mask = X_imputed[:, feature] <= threshold

        # Initialize results array
        results = np.zeros((X.shape[0], self.n_classes_))

        # Recursively predict for left and right children
        if np.any(left_mask):
            results[left_mask] = self._recursive_predict(self.tree_.children_left[node], X_imputed[left_mask], proba)
        if np.any(~left_mask):
            results[~left_mask] = self._recursive_predict(self.tree_.children_right[node], X_imputed[~left_mask], proba)

        return results

    def predict(self, X, check_input=True):
        return np.argmax(self._recursive_predict(0, X), axis=1)

    def predict_proba(self, X, check_input=True):
        return self._recursive_predict(0, X, proba=True)

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

exp = 100

rf1 = RandomForestClassifier(n_estimators=5)

# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

roc_auc_scores1 = []
roc_auc_scores2 = []
for exp_num in range(exp):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf1 = RandomForestClassifier(n_estimators=5)

    # Train the classifier on the training set
    rf1.fit(X_train, y_train)

    # Predict the probabilities on the test set
    y_probs = rf1.predict_proba(X_test)[:, 1]

    # Calculate the AUC score
    auc_score = roc_auc_score(y_test, y_probs)
    roc_auc_scores1.append(auc_score)

for exp_num in range(exp):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf2 = RandomForestClassifier(n_estimators=7)

    # Train the classifier on the training set
    rf1.fit(X_train, y_train)

    # Predict the probabilities on the test set
    y_probs = rf1.predict_proba(X_test)[:, 1]

    # Calculate the AUC score
    auc_score = roc_auc_score(y_test, y_probs)
    roc_auc_scores2.append(auc_score)

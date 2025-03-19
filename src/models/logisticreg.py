'''
We are using logistic regression model to check whether the pairwise data used in
RankNet is suitable or not. Logistic regression will provide us a sanity check.
'''

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.model_selection import train_test_split


def logReg(pairwise_data):
    # Define feature columns and target
    feature_cols = [col for col in pairwise_data.columns if col.startswith('delta_')]
    target_col = 'label'

    X = pairwise_data[feature_cols]
    y = pairwise_data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")

    # Define and fit the logistic regression model
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)

    # Make predictions
    y_pred_proba = lr_model.predict_proba(X_test)[:,1] # Get probability for class 1
    y_pred = lr_model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    loss = log_loss(y_test, y_pred_proba)

    print(f"Accuracy: {accuracy: .4f}")
    print(f"ROC_AUC: {roc_auc: .4f}")
    print(f"Log Loss: {loss: .4f}")    

    return "Done"
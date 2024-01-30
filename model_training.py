# model_training.py
# Training a machine learning model for football match predictions.

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_model(data, target_column):
    """
    Train a machine learning model on the given dataset.

    :param data: DataFrame containing the preprocessed football match data
    :param target_column: Name of the column that we want to predict (e.g., match outcome)
    """

    # Splitting data into features (X) and target (y)
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Choosing the Model - Random Forest Classifier in this case
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Training the Model
    model.fit(X_train, y_train)

    # Making Predictions and Evaluating the Model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    return model, accuracy, report

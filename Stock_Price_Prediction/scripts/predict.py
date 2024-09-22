# Predict the model using preprocessed test data

# Import required libraries
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# Import custom libraries
from preprocess import load_and_preprocess_data, prepare_headlines

# Get roor directory path
base_path = os.path.abspath(__file__)

def predict_stock_movement(test_data, model_path, vectorizer_path):
    # Load the model and vectorizer
    with open(model_path, 'rb') as clf:
        clf = pickle.load(clf)
    
    with open(vectorizer_path, 'rb') as cv:
        cv = pickle.load(cv)

    # Load and preprocess test data
    _, test_data = load_and_preprocess_data(test_data)
    X_test = test_data.iloc[:,2:]
    X_test = prepare_headlines(test_data)

    # Transform and predict
    X_test = cv.transform(X_test)
    y_test = test_data['Label']
    y_pred = clf.predict(X_test)

    # Evaluate the predictions
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    return accuracy, report, matrix

# Load model and vectorizer
model_path = os.path.join(base_path,'..','../models/stock_model.pkl')
vectorizer_path = os.path.join(base_path,'..','../models/count_vectorizer.pkl')

# Predict on test data
accuracy, report, matrix = predict_stock_movement(os.path.join(base_path,'..','../data/StockData.csv'),
                                                   model_path, vectorizer_path)
print(f'Accuracy of the predictions is {accuracy}')
print(f'Classification Report of the predictions is \n{report}')
print(f'Confusion Matrix of the predictions is \n{matrix}')
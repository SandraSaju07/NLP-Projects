# Main function where the app runs

# Import required libraries
import streamlit as st # type: ignore
import pickle
import pandas as pd

# Import custom libraries
from scripts.preprocess import prepare_headlines

# Load model and vectorizer
model_path = './models/stock_model.pkl'
vectorizer_path = './models/count_vectorizer.pkl'

with open(model_path, 'rb') as clf:
    clf = pickle.load(clf)

with open(vectorizer_path, 'rb') as cv:
    cv = pickle.load(cv)

# Streamlit App
st.title('Stock Price Prediction based on News Headlines')
st.write('Enter News Headlines below to Predict Stock Movement separated by dot(.)')

headline_input = st.text_area('')

if st.button('Predict'):
    if headline_input:
        headlines = prepare_headlines(pd.DataFrame([headline_input.split('.')]))
        headlines = cv.transform(headlines)
        prediction = clf.predict(headlines)
        result = 'Increase' if prediction[0] == 1 else 'Decrease'
        st.success(f'Stock Price is likely to {result}.')
    else:
        st.warning(f'Please enter headlines to predict.')

# Stock Price Prediction Based on News Headlines

## Overview
This project leverages Natural Language Processing (NLP) to predict stock price movements based on news headlines. By analyzing historical news and stock price data, the model aims to classify whether a stock's price will increase or decrease.

The project is developed as a web application using Streamlit, providing an interactive interface for users to input news headlines and receive stock price movement predictions.

## Project Structure
```bash 
Stock_Price_Prediction/
│
├── app.py                        # Main script for running the Streamlit app
├── data/
│   └── StockData.csv             # Dataset used for training and testing
│
├── models/
│   ├── count_vectorizer.pkl      # Pre-trained vectorizer for text transformation
│   └── stock_model.pkl           # Trained model for predicting stock movements
│
├── requirements.txt              # Python dependencies for the project
│
├── scripts/
│   ├── preprocess.py             # Script for data preprocessing and headline preparation
│   └── train_model.py            # Script to train and save the model
│   └── predict.py                # Script to predict from saved model
└── README.md                     # Project documentation
```
## How to Run
1. To install all the required libraries, run:
   pip install -r requirements.txt

2. To train the stock classifier model, run:
   python train_model.py

3. To start the Streamlit web application, run:
   streamlit run app.py
   
4. Predict Stock Movement
   Enter some headlines in the input field of the web interface.
   Click the Predict button to determine whether the stock is up or down.

## Future Enhancements
- Integrate additional machine learning algorithms and further optimize performance.
- Add a feature for real-time data analysis and prediction.
- Deploy the application to a cloud platform (e.g., Heroku or AWS) for wider accessibility.
   

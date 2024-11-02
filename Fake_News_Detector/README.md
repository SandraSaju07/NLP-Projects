# Fake News Detector App

## Overview
This is a web application that detects fake news using a deep learning LSTM model trained on a dataset of news articles.

## Project Structure
```bash 
Fake_News_Dectector/
│
├── data                      # Contains data to train the model
│   └── FakeNewsData.csv      # Fake News Data to train the model
├── model/                    # Directory where the trained models are stored
├── notebooks/                # Contains notebook file to perform training
│   └── model_training.ipynb  # Script to train the model
├── static/                   # Contains stylesheets for the web app
│   └── styles.css            # Styles for the HTML pages
├── templates/                # HTML templates for the web app
│   ├── index.html            # Web page to input fake news to be detected
│   └── result.html           # Web page to display result
├── app.py                    # The Flask application
├── preprocessing.py          # Script for text preprocessing (cleaning, tokenization)
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies required for the project
```
## How to Run
1. To install all the required libraries, run:
   pip install -r requirements.txt

2. To train and save the fake news detector model, run the notebook file:
   model_training.ipynb

3. To start the Flask web application, run:
   python app.py
   Then open your web browser and navigate to http://localhost:5000 to use the fakew news detector app.

4. Identify fake news
   Enter a fake news in the input field of the web interface.
   Click the Identify button to determine whether the news is Fake or Real.

   

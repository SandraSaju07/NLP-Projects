# Spam Classifier Web App

## Overview
This project is a web application that classifies messages as **Spam** or **Ham** (Not Spam) using Natural Language Processing (NLP) techniques. It leverages the **sklearn** library for text preprocessing and model training, and the application is built with **Flask** to provide a user-friendly web interface.

The project also addresses imbalanced data through techniques like **SMOTE** to improve classification performance. The model can be trained using algorithms such as **Naive Bayes** and **Random Forest**, and it automatically selects the best-performing algorithm.

## Project Structure
SpamClassifierApp/
│
├── app.py                   # The Flask application
├── preprocessing.py          # Script for text preprocessing (cleaning, tokenization)
├── training.py               # Script to train the model
├── inference.py              # Script to load the model and make predictions
├── models/                   # Directory where the trained models are stored
├── static/                   # Contains stylesheets for the web app
│   └── styles.css            # Styles for the HTML pages
├── templates/                # HTML templates for the web app
│   ├── index.html            # Web page to input messages for classification
│   └── result.html           # Web page to display classification result
├── Data/SpamData.csv         # The dataset used for training the model
├── requirements.txt          # Python dependencies required for the project
└── README.md                 # Project documentation

## How to Run
1. To install all the required libraries, run:
   pip install -r requirements.txt

2. To train the spam classifier model, run:
   python training.py

3. To start the Flask web application, run:
   python app.py
   Then open your web browser and navigate to http://localhost:5000 to use the spam classifier.
4. Classify Messages
   Enter a message in the input field of the web interface.
   Click the Classify button to determine whether the message is Spam or Ham.

## Future Enhancements
- Integrate additional machine learning algorithms and further optimize performance.
- Add a feature for real-time data analysis and message classification.
- Deploy the application to a cloud platform (e.g., Heroku or AWS) for wider accessibility.
   

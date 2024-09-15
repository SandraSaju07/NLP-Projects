## Inference Module - Loading the model and performing inference using Flask

# Import required libraries
import joblib
# Import custom Data Preprocessing Module
from preprocessing import clean_text

# Load trained model and vectorizer
def load_model():
    model = joblib.load('./models/spam_classifier.pkl')
    vectorizer = joblib.load('./models/tfidf_vectorizer.pkl')
    return model, vectorizer

# Predict the input message for Spam
def predict_message(message,model,vectorizer):
    message_clean = clean_text(message)
    message_vector = vectorizer.transform([message_clean]).toarray()
    return model.predict(message_vector)
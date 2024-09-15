## Data Preprocessing Module - Handle cleaning, lemmatization and vectorization

# Import required libraries
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('stopwords')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Clean and lemmatize the input text
def clean_text(text):
    review = re.sub('[^a-zA-Z]',' ',text)
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if word not in stopwords.words('english')]
    return ' '.join(review)

# Preprocess and vectorize the messages
def preprocess_text(messages):
    corpus = [clean_text(msg) for msg in messages]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus).toarray()
    return X, vectorizer
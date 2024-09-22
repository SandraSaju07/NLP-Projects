# Train the model using preprocessed data

# Import required libraries
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import os

# Import custom libraries
from preprocess import load_and_preprocess_data, prepare_headlines

# Get root directory path
base_path = os.path.abspath(__file__)

# Load and preprocess data
train_data,_ = load_and_preprocess_data(os.path.join(base_path,'..','../data/StockData.csv'))
X_train = train_data.iloc[:,2:]
X_train = prepare_headlines(X_train)

# Create Bag of Words - Vectorization
cv = CountVectorizer(ngram_range=(2,2))
X_train = cv.fit_transform(X_train)
y_train = train_data['Label']

# Train model using RandomForestClassifier
clf = RandomForestClassifier(n_estimators=200, criterion = 'entropy')
clf.fit(X_train,y_train)

# Save model and vectorizer
with open(os.path.join(base_path,'..','../models/stock_model.pkl'),'wb') as model_file:
    pickle.dump(clf, model_file)

with open(os.path.join(base_path,'..','../models/count_vectorizer.pkl'),'wb') as vectorizer_file:
    pickle.dump(cv, vectorizer_file)

# Import required libraries
import nltk # type: ignore
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer # type: ignore
from nltk.corpus import stopwords # type: ignore
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.preprocessing.text import one_hot # type: ignore

# Initialize lemmatizer object
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Define vocabulary size and max input size length
vocab_size = 5000
max_length = 47   # Set this to max length during training

# Define text preprocessing function
def preprocess_text(text):
    corpus = []
    for i in range(len(text)):
        review = re.sub('[^A-Za-z]',' ',text[i])
        review = review.lower().split()
        review = [lemmatizer.lemmatize(word) for word in review if word not in stop_words]
        review = ' '.join(review)
        corpus.append(review)

    return corpus

# Convert preprocessed text to onehot representation
def text_embedding(text):
    corpus = preprocess_text(text)
    onehot = [one_hot(words,vocab_size) for words in corpus]
    embedded_text = pad_sequences(onehot,padding='pre',maxlen=max_length)
    return embedded_text
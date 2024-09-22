# Preprocess data - perform basic data cleaning and transformations

# Import required libraries
import pandas as pd

def load_and_preprocess_data(file_path):
    # Load data
    df = pd.read_csv(file_path, encoding = 'ISO-8859-1')

    # Remove punctuation and convert to lowercase
    df.iloc[:,2:] = df.iloc[:,2:].replace('[^a-zA-Z]',' ', regex = True)
    df.iloc[:,2:] = df.iloc[:,2:].apply(lambda x:x.str.lower())

    # Split data into train and test based on Date
    train_data = df[df['Date'] < '20150101']
    test_data = df[df['Date'] > '20141231']

    return train_data, test_data

def prepare_headlines(data):
    headlines = [' '.join(str(x) for x in row) for row in data.values]
    return headlines
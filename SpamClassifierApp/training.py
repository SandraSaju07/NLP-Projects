## Model Training Module - Train preprocessed data with different algorithms to build a model
# Here, SMOTE is implemented to deal with imbalanced data and test different algorithms using GridSearchCV

# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
# Import custom Data Preprocessing Module
from preprocessing import preprocess_text

# Train the model using preprocessed data
def train_model():
    # Load and preprocess trainining data
    messages = pd.read_csv('./data/SpamData',sep='\t',names = ['label','message'])
    X, vectorizer = preprocess_text(messages['message'])
    y = pd.get_dummies(messages['label'],drop_first=True).values.ravel()

    # Handle imbalanced data using SMOTE
    smote = SMOTE(random_state=0)
    X_res, y_res = smote.fit_resample(X,y)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_res,y_res,test_size = 0.2,random_state=0)

    # Hyperparameter tuning with GridSearchCV
    models = {
        "RandomForest": RandomForestClassifier(),
        "NaiveBayes": MultinomialNB()
    }
    param_grid = {
        "RandomForest":{'n_estimators':[100,200]},
        "NaiveBayes":{}
    }

    best_model = None
    best_score = 0

    for name,model in models.items():
        clf = GridSearchCV(model,param_grid[name], scoring = 'accuracy', cv = 5)
        clf.fit(X_train,y_train)
        if clf.best_score_ > best_score:
            best_score = clf.best_score_
            best_model = clf.best_estimator_

    # Save best model and vectorizer
    joblib.dump(best_model,'./models/spam_classifier.pkl')
    joblib.dump(vectorizer,'./models/tfidf_vectorizer.pkl')

    # Evaluate on test data
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    conf_mat = confusion_matrix(y_test,y_pred)

    print(f"Model Accuracy: {accuracy}")
    print(f"Confusion Matrix: {conf_mat}")

# Start training the model...
print("Training the model....")
train_model()
print("Models are saved")
## Flask Web Application

# Import required libraries
from flask import Flask,render_template,request
from inference import load_model,predict_message

# Initialize Flask app. Load trained model and vectorizer
app = Flask(__name__)
model, vectorizer = load_model()

# Define and render Home page
@app.route('/')
def home():
    return render_template('index.html')

# Define and render Result page
@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        print(message)
        prediction = predict_message(message, model, vectorizer)
        print(prediction)
        result = 'Spam' if prediction[0] == 1 else 'Ham'
        return render_template('result.html', prediction = result)

# Run the web app
if __name__ == "__main__":
    app.run()
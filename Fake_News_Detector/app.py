# Import necessary packages
from flask import Flask,render_template,request # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from preprocessing import text_embedding

# Load the pretrained model
model = load_model('./model/fake_news_model.h5')

# Define prediction function
def predict_fake_news(preprocessed_text):
    prediction = model.predict(preprocessed_text)
    print(prediction)
    return 1 if prediction[0][0] < 0.5 else 0

# Initialize flask object
app = Flask(__name__)

# Index page function
@app.route('/')
def home():
    return render_template('index.html')

# Result page function
@app.route('/predict',methods=['POST'])
def predict():
    text = request.form['text']
    preprocessed_text = text_embedding([text])
    prediction = predict_fake_news(preprocessed_text)
    return render_template('result.html',prediction=prediction)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

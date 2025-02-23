# app.py

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model
model_path = 'spam_model.pkl'
with open(model_path, 'rb') as file:
    vectorizer,model = pickle.load(file)
print(type(model)) 

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    message = request.form.get('text')
    message = vectorizer.transform([message])
    
    prediction = model.predict(message)
    output = 'Spam' if prediction[0] == 1 else 'not sapm'

    return render_template('index.html', prediction_text='Prediction: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
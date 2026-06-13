from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__)

# Model load karo
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Text clean karne ka function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    cleaned = clean_text(news)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    confidence = model.predict_proba(vectorized)[0]
    
    if prediction == 1:
        result = "REAL NEWS ✅"
        score = round(confidence[1] * 100, 2)
        color = "green"
    else:
        result = "FAKE NEWS ❌"
        score = round(confidence[0] * 100, 2)
        color = "red"
    
    return render_template('index.html', 
                         result=result, 
                         score=score,
                         color=color,
                         news=news)

if __name__ == '__main__':
    app.run(debug=True)
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template, jsonify
import pickle

app = Flask(__name__)

# Load your pre-trained model and tokenizer
model = load_model('sentiment.keras')

# Load and configure tokenizer (ensure this matches your training setup)
with open('tkn.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

max_words = 100

def preprocess_text(text):
    # Remove numbers
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'\d+', '', text)
    return text

def predict_sentiment(feedback):
    # Preprocess the feedback
    feedback = preprocess_text(feedback)
    sequence = tokenizer.texts_to_sequences([feedback])
    padded_sequence = pad_sequences(sequence, maxlen=max_words)

    # Predict sentiment
    prediction = model.predict(padded_sequence)
    sentiment_labels = ['negative 😢', 'neutral 😐', 'positive 😁']
    sentiment = sentiment_labels[prediction.argmax()]
    return sentiment

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    feedback = request.json['feedback']
    sentiment = predict_sentiment(feedback)
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)

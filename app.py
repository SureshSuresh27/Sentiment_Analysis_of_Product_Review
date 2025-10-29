
from flask import Flask, render_template, request
import joblib
model = joblib.load("sentiment_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_review = request.form['review']
    transformed_text = vectorizer.transform([user_review])
    prediction = model.predict(transformed_text)[0]
    sentiment = "Positive-->" if prediction == 1 else "Negative !!"

    return render_template('index.html', prediction_text=f"Review Sentiment: {sentiment}")

if __name__ == "__main__":
    app.run(debug=True)

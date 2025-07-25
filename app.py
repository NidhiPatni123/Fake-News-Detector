from flask import Flask, render_template, request
import joblib
from datetime import datetime
from scipy.sparse import hstack, csr_matrix

app = Flask(__name__)

# Load model and encoders
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
source_encoder = joblib.load("source_encoder.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None

    if request.method == "POST":
        try:
            title = request.form.get("title", "")
            text = request.form.get("text", "")
            source = request.form.get("source", "")
            date_str = request.form.get("date", "")

            combined = f"{title} {text}"

            # Vectorize
            text_vect = vectorizer.transform([combined])

            # Encode source
            try:
                source_encoded = source_encoder.transform([source])[0]
            except Exception:
                source_encoded = 0

            # Extract year
            try:
                year = datetime.strptime(date_str, "%Y-%m-%d").year
            except Exception:
                year = 0

            # Combine meta as sparse — NO DataFrame!
            meta_sparse = csr_matrix([[source_encoded, year]])

            # Combine all
            X_input = hstack([text_vect, meta_sparse])

            # Predict
            label = model.predict(X_input)[0]
            prob = model.predict_proba(X_input)[0]
            confidence = prob[label] * 100

            prediction = "✅ TRUE Information" if label == 1 else "❌ FALSE Information"

        except Exception as e:
            print("🔥 ERROR:", e)
            prediction = "Something went wrong!"
            confidence = 0

    return render_template("index.html",
                           prediction=prediction,
                           confidence=round(confidence, 2) if confidence else None)

if __name__ == "__main__":
    app.run(debug=True)

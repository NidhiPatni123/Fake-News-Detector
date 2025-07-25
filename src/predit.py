import joblib
from datetime import datetime
from scipy.sparse import hstack, csr_matrix  # ✅ Add csr_matrix

# Load model and encoders
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
source_encoder = joblib.load('source_encoder.pkl')

# Get user input
title = input("Enter the title: ")
text = input("Enter the text: ")
source = input("Enter the source: ")
date_str = input("Enter the date (YYYY-MM-DD): ")

# Combine text fields
combined = f"{title} {text}"

# Parse year
try:
    year = datetime.strptime(date_str, "%Y-%m-%d").year
except ValueError:
    print("⚠️ Invalid date format. Defaulting year to 0.")
    year = 0

# Vectorize text
text_vect = vectorizer.transform([combined])

# Encode source
try:
    source_encoded = source_encoder.transform([source])[0]
except ValueError:
    print("⚠️ Unknown source. Defaulting to 0.")
    source_encoded = 0

# Combine features as sparse
meta_sparse = csr_matrix([[source_encoded, year]])  # ✅ Simpler than DataFrame
X_input = hstack([text_vect, meta_sparse])

# Predict label
label = model.predict(X_input)[0]

# Predict confidence
prob = model.predict_proba(X_input)[0]
confidence = prob[label] * 100

# Show result
print("\n===============================")
if label == 1:
    print("✅ Prediction: This is TRUE information.")
else:
    print("❌ Prediction: This is FALSE information.")
print(f"🔒 Confidence: {confidence:.2f}%")
print("===============================\n")

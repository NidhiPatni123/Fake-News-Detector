import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib  # For saving the vectorizer

df = pd.read_csv("C:/Users/patni/OneDrive/Desktop/fake/data/fake_news_india.csv")

# Load your dataset (example path, adjust accordingly)


# Import your clean_text function
from preprocessing import clean_text

# Apply text cleaning to your 'text' or 'content' column
df['cleaned_text'] = df['text'].apply(clean_text)

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust this

# Fit and transform the cleaned text
X = vectorizer.fit_transform(df['cleaned_text'])

# Save the vectorizer for later use (e.g., during prediction)
joblib.dump(vectorizer, 'C:/Users/patni/OneDrive/Desktop/fake/models/tfidf_vectorizer.pkl')

# Optional: save the transformed feature matrix
joblib.dump(X, 'C:/Users/patni/OneDrive/Desktop/fake/models/features_tfidf.pkl')

# Also save labels if you're training later
y = df['label']  # Assuming your CSV has a 'label' column like FAKE/REAL
joblib.dump(y, 'C:/Users/patni/OneDrive/Desktop/fake/models/labels.pkl')
print("✅ Feature extraction complete!")


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import resample
import numpy as np
import random
import joblib

# Load dataset
df = pd.read_csv("C:/Users/patni/OneDrive/Desktop/fake/data/fake_news_india.csv")

# Drop nulls and exact duplicates
df = df.dropna().drop_duplicates()

# Combine 'title' and 'text' fields
df['content'] = df['title'] + " " + df['text']

# Remove near-duplicates (same content with same label)
df = df.drop_duplicates(subset=['content', 'label'])

# Shuffle data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Balance the dataset
df_majority = df[df.label == 1]
df_minority = df[df.label == 0]

df_minority_upsampled = resample(df_minority, 
                                  replace=True, 
                                  n_samples=len(df_majority), 
                                  random_state=42)

df_balanced = pd.concat([df_majority, df_minority_upsampled])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Optional: add small label noise to reduce model overfitting
flip_ratio = 0.05  # flip 5% of labels randomly
indices_to_flip = df_balanced.sample(frac=flip_ratio, random_state=42).index
df_balanced.loc[indices_to_flip, 'label'] = 1 - df_balanced.loc[indices_to_flip, 'label']

# Split features and target
X = df_balanced['content']
y = df_balanced['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Vectorize after split
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vect, y_train)

# Predict
y_pred = model.predict(X_test_vect)

# Evaluate
accuracy = accuracy_score(y_test, y_pred) * 100
print("Accuracy:", round(accuracy, 2), "%")

print("\n Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred))
#  6. Save model & vectorizer at the end of train_model.py
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("Model and vectorizer saved successfully.")
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 📍 Point to your custom NLTK data location
nltk.data.path.append("C:/Users/patni/OneDrive/Desktop/fake/nltk_data")

# ✅ Download check is optional if data already exists
# nltk.download('stopwords')
# nltk.download('wordnet')

# ✅ Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """Clean input text by removing punctuation/digits and lemmatizing."""
    if not isinstance(text, str):
        return ""

    # 🔤 Remove non-alphabetic characters and lowercase
    text = re.sub(r'[^a-zA-Z]', ' ', text.lower())

    # 🧹 Tokenize, remove stopwords, and lemmatize
    words = [
        lemmatizer.lemmatize(w)
        for w in text.split()
        if w not in stop_words
    ]
    
    return ' '.join(words)
if __name__ == "__main__":
    sample = "The quick brown foxes were jumping over the lazy dogs in 2023!"
    cleaned = clean_text(sample)
    print("🔍 Original:", sample)
    print("🧼 Cleaned :", cleaned)


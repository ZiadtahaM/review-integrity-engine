import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

print("Loading dataset...")
df = pd.read_csv('fake reviews dataset.csv')

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

print("Preprocessing text...")
df['clean_text'] = df['text_'].apply(preprocess_text)

print("Vectorizing...")
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(df['clean_text'])

print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_vec, df['label'])

print("Saving model and vectorizer...")
joblib.dump(model, 'logistic_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Done! Saved as logistic_model.pkl and vectorizer.pkl")

import pandas as pd
import numpy as np
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load dataset
df = pd.read_csv("UpdatedResumeDataSet.csv")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Text cleaning function
def clean_text(text):
    text = re.sub(r'\n|\r', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Preprocessing with spaCy
def preprocess(text):
    text = clean_text(text)
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

# Apply preprocessing
df['Cleaned_Resume'] = df['Resume'].apply(preprocess)

# Encode labels
le = LabelEncoder()
df['Encoded_Category'] = le.fit_transform(df['Category'])

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['Cleaned_Resume'])
y = df['Encoded_Category']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Evaluation
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Save the model and vectorizer
joblib.dump(clf, "resume_classifier_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(le, "label_encoder.pkl")

# Function to predict category from new resume
def predict_resume_category(resume_text):
    cleaned = preprocess(resume_text)
    vector = vectorizer.transform([cleaned])
    pred = clf.predict(vector)
    return le.inverse_transform(pred)[0]

# 🔍 Example Prediction
sample_resume = df['Resume'].iloc[0]
print("\nSample Prediction:", predict_resume_category(sample_resume))

import streamlit as st
import joblib
import spacy
import re
import pdfplumber
import pytesseract
from PIL import Image
import io

# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Load trained ML components
model = joblib.load("resume_classifier_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ----------------------------
# Text Preprocessing Functions
# ----------------------------
def clean_text(text):
    text = re.sub(r'\n|\r', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def preprocess(text):
    text = clean_text(text)
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

# --------------------------------
# PDF Text Extraction with OCR Fallback
# --------------------------------
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"

    # Fallback to OCR if text is insufficient
    if len(text.strip()) < 100:
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                img = page.to_image(resolution=300).original
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                image = Image.open(img_byte_arr)
                ocr_text = pytesseract.image_to_string(image)
                text += ocr_text + "\n"
    return text

# ----------------------------
# Resume Category Prediction
# ----------------------------
def predict_category(resume_text):
    cleaned = preprocess(resume_text)
    vector = vectorizer.transform([cleaned])
    probabilities = model.predict_proba(vector)[0]
    predicted_index = probabilities.argmax()
    category = label_encoder.inverse_transform([predicted_index])[0]
    confidence = probabilities[predicted_index] * 100
    return category, confidence

# ----------------------------
# Resume Improvement Tips
# ----------------------------
def generate_resume_tips(text):
    tips = []

    # Word count
    word_count = len(text.split())
    if word_count < 150:
        tips.append("🔍 Your resume seems too short. Consider adding more experience, projects, or skills.")
    elif word_count > 800:
        tips.append("✂️ Your resume is quite long. Try to keep it concise and focused (1-2 pages).")

    # Keywords check
    keywords = ['python', 'machine learning', 'project', 'internship', 'leadership', 'communication', 'data analysis']
    found_keywords = [kw for kw in keywords if kw.lower() in text.lower()]
    if len(found_keywords) < 3:
        tips.append("💡 Add relevant keywords like 'project', 'internship', or 'data analysis' to strengthen your resume.")

    # Education check
    if not re.search(r"(b\.tech|bachelor|master|university|degree)", text, re.IGNORECASE):
        tips.append("🎓 Include your educational background (e.g., B.Tech, Bachelor, Master, etc.).")

    # Contact info check
    if not re.search(r"\b\d{10}\b", text) and not re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.\w+\b", text):
        tips.append("📬 Make sure to include your email and phone number.")

    # Soft skills
    soft_skills = ['team', 'leadership', 'communication']
    if not any(skill in text.lower() for skill in soft_skills):
        tips.append("🤝 Include soft skills like teamwork, leadership, or communication.")

    return tips if tips else ["✅ Your resume looks well-rounded! Great job!"]

# ----------------------------
# Streamlit App UI
# ----------------------------
st.set_page_config(page_title="Resume Classifier", page_icon="📄", layout="centered")
st.title("📄 Resume Classifier")
st.markdown("Upload a resume (`.txt` or `.pdf`) to classify it into a job category and get improvement suggestions.")

uploaded_file = st.file_uploader("Upload a resume file", type=["txt", "pdf"])
resume_text = ""

# File Handling
if uploaded_file:
    if uploaded_file.name.endswith(".txt"):
        resume_text = uploaded_file.read().decode("utf-8")
    elif uploaded_file.name.endswith(".pdf"):
        st.info("Extracting text from PDF...")
        resume_text = extract_text_from_pdf(uploaded_file)

# Display and Analyze Resume
if resume_text:
    st.subheader("📑 Resume Preview")
    st.text_area("Extracted Resume Text", resume_text, height=300)

    if st.button("🧠 Predict Job Category"):
        with st.spinner("Analyzing..."):
            category, confidence = predict_category(resume_text)
        st.success(f"🎯 Predicted Category: **{category}**")
        st.info(f"📊 Confidence: **{confidence:.2f}%** match")

        st.subheader("📌 Tips to Improve Your Resume")
        for tip in generate_resume_tips(resume_text):
            st.markdown(f"- {tip}")
else:
    st.warning("Please upload a valid resume file (.txt or .pdf).")

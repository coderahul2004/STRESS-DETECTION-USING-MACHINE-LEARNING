import streamlit as st
import pickle
import nltk
import re
import string
from nltk.corpus import stopwords

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
cv = pickle.load(open("vectorizer.pkl", "rb"))

# Download stopwords (only first time)
nltk.download('stopwords')

# Setup NLP tools
stemmer = nltk.SnowballStemmer("english")
stop_words = set(stopwords.words("english"))

# Text cleaning function
def clean(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)

    text = [word for word in text.split() if word not in stop_words]
    text = " ".join(text)

    text = [stemmer.stem(word) for word in text.split()]
    text = " ".join(text)

    return text

# UI
st.title("Stress Detection App 🧠")

# Input box
user_input = st.text_area("Enter your text")

# Button
if st.button("Predict"):
    
    # Check empty input
    if user_input.strip() == "":
        st.warning("Please enter some text")
    
    else:
        # Clean text
        cleaned_text = clean(user_input)

        # Transform input
        data = cv.transform([cleaned_text]).toarray()

        # Predict
        prediction = model.predict(data)

        # Show result
        if prediction[0] == 0:
            st.success("Not Stressed 😌")
        else:
            st.error("Stressed 😰")

        # Confidence score (optional but useful)
        try:
            prob = model.predict_proba(data)
            confidence = max(prob[0]) * 100
            st.info(f"Confidence: {confidence:.2f}%")
        except:
            pass
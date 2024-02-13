import streamlit as st
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

label = ['neutral', 'joy', 'sadness', 'fear', 'surprise', 'anger', 'shame', 'disgust']

# Use the correct file path
model = joblib.load('D:/Projects/sentiment_analysis/sentiment_analysis/pipeline_model.joblib')

st.title('Sentiment Analysis')

text = st.text_input("Enter your text here:")

if text:
    # Perform sentiment analysis on the input text
    prediction = model.predict_proba([text])[0]
    index = prediction.argmax()
    confidence = prediction[index] * 100

    # Use the index to get the sentiment label
    result = label[index]

    # Display the sentiment analysis result with bigger font size and confidence rate
    
    st.markdown("<h1 style='text-align: center;'>Sentiment: " + result + "</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Confidence: " + f"{confidence:.2f}%" + "</h2>", unsafe_allow_html=True)

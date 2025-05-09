import re
import pandas as pd
from streamlit_option_menu import option_menu
import streamlit as st
import joblib
import matplotlib.pyplot as plt

# Load models once at the top
sentiment_model = joblib.load("E:/MlModels-main/app/sentiment_model.pkl")
vectorizer = joblib.load("E:/MlModels-main/app/tfidf_vectorizer.pkl")

# Preprocessing function for cleaning the text input
def clean_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove non-alphabetic characters (e.g., punctuation)
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra whitespaces
    text = ' '.join(text.split())
    return text

# Sidebar Menu
with st.sidebar:
    st.title("Machine Learning Models")
    selected = option_menu(
        "Select a ML model To Predict", 
        ['Sentiment Analysis', 'SleepTime Prediction'],
        icons=["emoji-smile-fill", "alarm-fill"], 
        default_index=0
    )

# Sentiment Analysis Section
if selected == "Sentiment Analysis":
    st.title("Sentiment Analysis")
    text = st.text_area("Enter the Sentence to predict the sentiment")
    
    if st.button("Predict"):
        # Clean and preprocess the input text
        cleaned_text = clean_text(text)
        
        # Transform the cleaned input text using the vectorizer
        transformed_text = vectorizer.transform([cleaned_text])
        
        # Predict sentiment
        prediction = sentiment_model.predict(transformed_text)

        st.success(f"Prediction: {prediction[0]}")

        # Clear the previous plot
        plt.clf()

        # Define sentiment counts
        if prediction[0] == 'positive':
            sent = {'positive': 100, 'negative': 0, 'neutral': 0}
        elif prediction[0] == 'negative':
            sent = {'positive': 0, 'negative': 100, 'neutral': 0}
        else:
            sent = {'positive': 0, 'negative': 0, 'neutral': 100}

        # Display the sentiment bar plot
        plt.bar(sent.keys(), sent.values(), color=['green', 'red', 'gray'])
        plt.xlabel('Sentiments')    
        plt.ylabel('Count') 
        plt.title('Sentiment Analysis')
        st.pyplot(plt)

# Sleep Time Prediction Section
else:
    st.title("SleepTime Prediction")

    col1, col2 = st.columns(2)
    with col1:
        work = st.number_input("Enter your Workout Time", min_value=0.0, step=0.1, max_value=10.0)
    with col2:
        study = st.number_input("Enter your Study Time", min_value=0.0, step=0.1, max_value=10.0)

    col3, col4 = st.columns(2)
    with col3:
        phone_usage = st.number_input("Enter your Phone usage Time", min_value=0.0, step=0.1, max_value=10.0)
    with col4:
        workhours = st.number_input("Enter your Work Hours Time", min_value=0.0, step=0.1, max_value=10.0)

    col5, col6 = st.columns(2)
    with col5:
        caffeine = st.number_input("Enter your Caffeine Intake", min_value=0.0, step=0.1, max_value=300.0)
    with col6:
        relax = st.number_input("Enter your Relaxation Time", min_value=0.0, step=0.1, max_value=10.0)

    if st.button("Predict Sleep Time"):
        # Load the sleep time prediction model
        sleep_model = joblib.load("E:/MlModels-main/app/sleep.pkl")

        # Create a DataFrame with column names matching the training data
        input_df = pd.DataFrame([{
            'work': work,
            'study': study,
            'phone_usage': phone_usage,
            'workhours': workhours,
            'caffeine': caffeine,
            'relax': relax
        }])

        # Predict using the model
        prediction = sleep_model.predict(input_df)

        # Display the result
        st.success(f"Predicted Sleep Time is {prediction[0]:.2f} hours")

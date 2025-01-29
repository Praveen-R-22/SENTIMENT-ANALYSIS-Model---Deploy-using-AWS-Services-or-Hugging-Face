import streamlit as st
import mysql.connector
from datetime import datetime
from transformers import pipeline

# Set up a connection to the RDS instance
def get_db_connection():
    connection = mysql.connector.connect(
        host='app-log.clewgymk8cwr.ap-south-1.rds.amazonaws.com',
        user='admin',
        password='12345678',
        database='user_info'
    )
    return connection


# Log user information to RDS
def log_user_info(username):
    conn = get_db_connection()
    cursor = conn.cursor()
    login_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute("INSERT INTO users (username, login_time) VALUES (%s, %s)", (username, login_time))
    conn.commit()
    cursor.close()
    conn.close()

# Load the pre-trained model for sentiment analysis
st.title("Sentiment Analysis Web App")
st.write("Enter a text to predict its sentiment.")

def load_model():
    model_path = "/home/ubuntu/BERT_Finetuned_model"  # Update the path accordingly
    return pipeline("sentiment-analysis", model=model_path, return_all_scores=True)

# Load the model
model = load_model()

# Create a text box to accept user input
user_input = st.text_area("Enter text here:")

# Define a mapping for labels to sentiment names
label_to_sentiment = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

# Capture the username and log it to the database
username = st.text_input("Enter your username:")
if username.strip():
    log_user_info(username)
    st.write(f"Welcome {username}! Your login time has been recorded.")


if st.button("Analyse"):
    if user_input.strip():
        # Perform the prediction
        response = model(user_input)

        if response:
            # Display all confidence scores for each sentiment
            st.write("**Sentiment Probabilities:**")
            for sentiment in response[0]:
                sentiment_label = label_to_sentiment.get(sentiment['label'], 'Unknown')
                confidence_score = round(sentiment['score'], 4)
                st.write(f"{sentiment_label}: {confidence_score}")

            # Extract the sentiment with the highest confidence
            max_sentiment = max(response[0], key=lambda x: x['score'])
            predicted_sentiment = label_to_sentiment.get(max_sentiment['label'], "Unknown Sentiment")
            confidence = round(max_sentiment['score'], 4)

            # Display the final prediction with the highest confidence
            st.write(f"\n**Prediction:** {predicted_sentiment}")
            st.write(f"**Confidence (Highest):** {confidence}")
    else:
        st.warning("Please enter some text to analyze.")
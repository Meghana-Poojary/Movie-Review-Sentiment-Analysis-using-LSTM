import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
import re
import streamlit as st

word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

model = load_model('MovieReviews.h5')


def preprocess_text(text, maxlen=500, vocab_size=10000):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)  # remove punctuation

    words = text.split()
    encoded = []

    for word in words:
        idx = word_index.get(word, 2)
        if idx >= vocab_size:
            idx = 2
        encoded.append(idx)

    return pad_sequences([encoded], maxlen=maxlen, padding='post')

def prediction(review):
    preprocessed = preprocess_text(review)
    pred = model.predict(preprocessed)
    sentiment = 'Positive' if pred[0][0] > 0.5 else 'Negative'
    return pred[0][0], sentiment


def main():
    st.set_page_config(page_title="Movie Review Sentiment Analyzer", page_icon="🎬", layout="wide")
    
    st.title("🎬 Movie Review Sentiment Analyzer")
    st.markdown("Analyze the sentiment of movie reviews using our trained deep learning model!")
    
    st.sidebar.header("About")
    st.sidebar.markdown("""
    This app uses a neural network trained on the IMDB movie review dataset to classify reviews as positive or negative.
    
    **How it works:**
    1. Enter your movie review text
    2. Click 'Analyze Sentiment'
    3. Get instant sentiment analysis!
    """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter Your Movie Review")
        user_input = st.text_area(
            "Type or paste your movie review here:",
            height=200,
            placeholder="This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout..."
        )
        
        analyze_button = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("Results")
        if analyze_button and user_input.strip():
            with st.spinner("Analyzing sentiment..."):
                confidence, sentiment = prediction(user_input)
            
            # Display results
            if sentiment == 'Positive':
                st.success(f"**{sentiment}** Review")
                st.markdown("😊 This review appears to be positive!")
            else:
                st.error(f"**{sentiment}** Review")
                st.markdown("😞 This review appears to be negative.")
            
            
            # Raw prediction value
            st.markdown("**Prediction Details:**")
            st.write(f"Raw prediction score: {confidence:.4f}")
            st.write(f"Threshold: 0.5 (Positive > 0.5, Negative ≤ 0.5)")
            
        elif analyze_button and not user_input.strip():
            st.warning("Please enter a movie review to analyze.")
        else:
            st.info("Enter a review and click 'Analyze Sentiment' to see results!")
    
    # Footer
    st.markdown("---")
    st.markdown("*Built with Streamlit and TensorFlow*")


if __name__ == "__main__":
    main()
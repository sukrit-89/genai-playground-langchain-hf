import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

#load the word index
word_index=imdb.get_word_index()
reverse_word_index={value:key for key,value in word_index.items()}

#load the trained model
model= load_model('simple_rnn_imdb.keras')

# helper Functions
#Function to decode review

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?')for i in encoded_review])

# Function to preprocess users input

def preprocess_text(text):
    words=text.lower().split()
    max_features = 10000  # Must match the vocabulary size used during training
    encoded_review = []
    for word in words:
        index = word_index.get(word, 2) + 3
        # Clamp index to valid range [0, max_features)
        if index >= max_features:
            index = 2  # Use unknown token for out-of-vocabulary words
        encoded_review.append(index)
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review 


# creating our prediction function

def pred_sentiment(review):
    preprocessed_review=preprocess_text(review) #encoded format
    prediction=model.predict(preprocessed_review)
    sentiment='Positive' if prediction [0][0]>0.5 else 'Negative'
    return sentiment,prediction[0][0]


# designing streamlit
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a moview review to classify it as positive or negative.')

#user input 
user_input=st.text_area('Movie Review')

if st.button('Classify'):
    pre_input=preprocess_text(user_input)

    #prediction
    prediction=model.predict(pre_input)
    sentiments='Positive' if prediction[0][0]>0.5 else 'Negative'
    
    #Display the result

    st.write(f'Sentiment:{sentiments}')
    st.write(f'Prediction Score:{prediction[0][0]}')
else:
    st.write('Please enter a movie review.')

#sample review 
st.write('sample reviews\n: This is a great movie and the plot was thrilling\n or \n This moview was a average moview and the acting was also not so good ')
import streamlit as st
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import tensorflow as tf
import pickle
import pandas as pd

## Loading the trained model
model = tf.keras.models.load_model('regression_model.h5')

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('onehot_encoder_geo', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Streamlit app
st.title('💰 Customer Salary Prediction')
st.write('Predict estimated salary based on customer demographics and banking information')

# User Input
st.subheader('Enter Customer Information')

col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0].tolist())
    gender = st.selectbox('Gender', label_encoder_gender.classes_)
    age = st.slider('Age', 18, 92, 35)
    credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=650)
    balance = st.number_input('Account Balance', min_value=0.0, value=50000.0, step=1000.0)

with col2:
    tenure = st.slider('Tenure (Years with Bank)', 0, 10, 5)
    num_of_products = st.slider('Number of Products', 1, 4, 1)
    has_cr_card = st.selectbox('Has Credit Card', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    is_active_member = st.selectbox('Is Active Member', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

# Predict button
if st.button('Predict Salary', type='primary'):
    # Prepare the input data (without EstimatedSalary - that's what we're predicting)
    input_data = {
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member]
    }
    
    # One-hot encode geography
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

    # Combine input data with encoded geography
    input_data_df = pd.concat([pd.DataFrame(input_data).reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale the input data
    input_data_scaled = scaler.transform(input_data_df)

    # Make prediction
    prediction = model.predict(input_data_scaled)
    predicted_salary = prediction[0][0]

    # Display results
    st.success(' Prediction Complete!')
    
    st.metric(
        label="Estimated Annual Salary",
        value=f"${predicted_salary:,.2f}",
        delta=None
    )
    
    # Additional insights
    st.subheader('Customer Profile Summary')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**Location:** {geography}")
        st.info(f"**Age:** {age} years")
        st.info(f"**Gender:** {gender}")
    
    with col2:
        st.info(f"**Credit Score:** {credit_score}")
        st.info(f"**Balance:** ${balance:,.2f}")
        st.info(f"**Tenure:** {tenure} years")
    
    with col3:
        st.info(f"**Products:** {num_of_products}")
        st.info(f"**Credit Card:** {'Yes' if has_cr_card == 1 else 'No'}")
        st.info(f"**Active:** {'Yes' if is_active_member == 1 else 'No'}")

# Footer
st.markdown('---')
st.caption(' This prediction is based on an Artificial Neural Network trained on customer banking data.')
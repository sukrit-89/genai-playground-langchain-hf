import streamlit as st
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
import tensorflow as tf
import pickle
import pandas as pd

## loading the trained model

model=tf.keras.models.load_model('model.h5')
with open('label_encoder_gender.pkl','rb') as f:
    label_encoder_gender=pickle.load(f)
with open('OHE_encoder_geo.pkl','rb') as f:
    OHE_encoder_geo=pickle.load(f)
with open('Scaler.pkl','rb') as f:
    Scaler=pickle.load(f)
    
#streamlit app
st.title('Customer Churn Prediction')

##User Input
geography =st.selectbox('Geography',OHE_encoder_geo.categories_[0].tolist())
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credict Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of Products',1,4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])

# Predict button
if st.button('Predict Churn'):
    # Prepare the input data
    input_data={
        'CreditScore':[credit_score],
        'Gender':[label_encoder_gender.transform([gender])[0]],
        'Age':[age],
        'Tenure':[tenure],
        'Balance':[balance],
        'NumOfProducts':[num_of_products],
        'HasCrCard':[has_cr_card],
        'IsActiveMember':[is_active_member],
        'EstimatedSalary':[estimated_salary]
    }
    geo_encoded=OHE_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df=pd.DataFrame(geo_encoded,columns=OHE_encoder_geo.get_feature_names_out(['Geography']))

    input_data=pd.concat([pd.DataFrame(input_data).reset_index(drop=True),geo_encoded_df],axis=1)

    input_data_scaled=Scaler.transform(input_data)

    # Make prediction
    prediction=model.predict(input_data_scaled)
    prediction_proba=prediction[0][0]

    # Display results
    st.subheader('Prediction Result')
    if prediction_proba>0.5:
        st.error(f"⚠️ The customer is likely to churn (Probability: {prediction_proba:.2%})")
    else:
        st.success(f"✅ The customer is not likely to churn (Probability: {prediction_proba:.2%})")
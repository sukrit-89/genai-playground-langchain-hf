import streamlit as st
import pandas as pd
st.title("Streamlit Text Input")
name=st.text_input("Enter your name:")
age=st.slider("Select your age:",0,100,25)
st.write(f"Your Age is {age}")
opts=["Python","Java","C++","JavaScript"]
choice=st.selectbox("Choose your favourite language:",opts)
st.write(f"You selected {choice}.")
if name:
    st.write(f"Hello,{name}")
    
    
data={
    "Name":["X","Y","Z","W"],
    "Age":[25,30,48,45],
    "City":["A","B","C","D"]
    }
df=pd.DataFrame(data)
st.write(df)

Upload_files=st.file_uploader("Choose a pdf file",type="csv")

if Upload_files is not None:
    df=pd.read_csv(Upload_files)
    st.write(df)
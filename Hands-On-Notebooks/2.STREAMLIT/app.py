import streamlit as st
import pandas as pd
import numpy as np

#title
st.title("Hello Streamlit")

#displaying a text
st.write("Hey this is dev Sukrit")

#create DF

df=pd.DataFrame({
    'First column':[1,2,5,6,8],
    'Second column':[10,20,50,60,80]
})

#display dataframe

st.write("Here is the DataFrame")
st.write(df)

#create a line chart
chart_data=pd.DataFrame(
    np.random.rand(20,3),columns=['a','b','c']
)

st.line_chart(chart_data)
import streamlit as st
import numpy as np
import pandas as pd

st.title("Space Traffic Management , Find Collision Probability and Risk Factor")
st.header("Space Traffic Management")

st.markdown("Markdown ")
st.write("The value of :red[***x***] is : ", 12)
sample_df = pd.read_csv('data/sample_data.csv')
    
with st.expander('Sample Data'):
    sample_df.shape
    
    sample_df

col1, col2 = st.columns(2)
with col1:
    x = st.slider("Choose and x value" , 1, 10)
with col2:
    x = st.slider("Choose and y value" , 1, 10)


with st.sidebar:
    st.header("About app")
    st.write("Created By Nexus IIT-J")
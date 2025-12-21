import streamlit as st
import numpy as np
import pandas as pd

page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #f0f8ff; /* Light blue background */
}

[data-testid="stSidebar"] {
    background-color: #e6e6fa; /* Light lavender sidebar */
}
</style>
"""
st.set_page_config(
    layout="wide" , 
    initial_sidebar_state="expanded",
    )

main_page = st.Page("FrontPg.py",title="Main Page",default=True )
About = st.Page("About.py",title="About" )
Future_Development = st.Page("FD.py",title="Incoming development" )
pg = st.navigation(
    [main_page,About,Future_Development],
    position="top",
    expanded=True,
    
    )
pg.run()

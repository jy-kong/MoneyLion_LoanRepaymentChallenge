import streamlit as st
import pandas as pd
import pkg_resources
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

st.set_page_config(layout="wide", page_title='Loan Risk Predictor', page_icon='💰')

st.title("Exploratory Data Analysis")

st.markdown("""
    You can upload loan applicant data here and analyze it using our loan risk prediction tools!
""")

data = None

def explore(data):    
    if data is None:
        st.error("Please upload a CSV file before exploration.")
    else:
        empty_columns = [column for column in data.columns if data[column].isnull().values.any()]
        
        if empty_columns:
            st.error(f"Your loan applicant data contains missing values in the following column(s): {empty_columns}. Please ensure all fields are complete before proceeding.")
        else:
            pr = ProfileReport(data, title="Loan Applicant Data Profiling Report", explorative=True)
            st_profile_report(pr)

uploaded_file = st.file_uploader("Choose a CSV file", type='csv')
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

if st.button("Explore"):
    explore(data)

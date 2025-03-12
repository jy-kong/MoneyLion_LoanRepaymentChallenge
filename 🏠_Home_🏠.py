import streamlit as st
import pandas as pd

st.set_page_config(
    page_title='Loan Risk Predictor',
    layout='wide',
    page_icon="💰"
)

st.title("Loan Risk Predictor 💰")

st.markdown(
    """
    #### Introduction
    Welcome to the main page of **Loan Risk Predictor**. This web application is a prototype designed to 
    assist **financial institutions**, **loan officers**, and **data analysts** in assessing the risk of loan repayment. 
    The **LightGBM** model serves as the predictive engine in this application.  
    Loan Risk Predictor offers three key functionalities: **Exploratory Data Analysis**, **Single Prediction**, 
    and **Multiple/Dataset Prediction**. 

    Users can upload datasets containing applicant details and repayment history to explore insights on the 
    **Exploratory Data Analysis** page. Additionally, users can manually input an applicant's financial and profile 
    data on the **Single Prediction** page or upload a CSV file on the **Multiple/Dataset Prediction** page 
    to evaluate the repayment risk of multiple applicants efficiently.

    #### Loan Repayment Challenge
    Loan repayment prediction is a critical task in financial risk management. Financial institutions must assess 
    the likelihood of an applicant repaying their loan on time to optimize interest rates, reduce default risks, 
    and maintain a healthy loan portfolio. Factors influencing loan repayment include credit history, 
    income level, debt-to-income ratio, and previous repayment behavior. By leveraging machine learning, 
    this application aims to improve risk assessment accuracy and enhance decision-making in loan approvals.

    #### Video About Loan Facts
    """
)
col1, col2 = st.columns(2)

video_file = open("Loans 101 (Loan Basics 1_3).mp4", 'rb')
video_bytes = video_file.read()

with col1: 
    st.video(video_bytes)


st.markdown("""
    #### What are the loan risk prediction outcomes assigned to applicants?
    The prediction model categorizes loan applicants into **two primary risk groups** based on their repayment behavior:

    - **High Risk**: Indicates a significant likelihood of default, meaning the borrower may struggle to repay the loan.
    - **Low/Medium Risk**: Indicates a lower likelihood of default, where the borrower has a history of successful or ongoing repayments.

    However, in the original dataset, loan statuses are provided with more granularity, as shown below:

    **Loan Risk Categorization**

    1) **High Risk Category**  
    These statuses indicate **significant risk of default**, where the borrower has missed payments, required collections, or had their loan written off due to non-repayment.

    | Loan Status               | Description |
    |---------------------------|-------------|
    | External Collection       | Loan sent to an external debt collection agency. |
    | Internal Collection       | Loan handled by the lender's internal collection team. |
    | Charged Off              | Loan written off as a loss due to non-repayment. |
    | Charged Off Paid Off      | Initially defaulted but repaid later. |
    | Credit Return Void        | Payment failure due to insufficient funds. |
    | Settled Bankruptcy        | Borrower declared bankruptcy and settled the loan under those conditions. |

    2) **Low/Medium Risk Category**  
    These statuses indicate **successful repayment, ongoing loans, or administrative cancellations**, meaning they do not pose a significant risk.

    | Loan Status                   | Description |
    |--------------------------------|-------------|
    | Paid Off Loan                  | Loan fully repaid on time. |
    | New Loan                       | Newly approved loan, with no payment issues yet. |
    | Settlement Paid Off             | Loan settled successfully by borrower. |
    | Pending Paid Off                | Loan is close to being fully repaid. |
    | Returned Item                   | A single missed payment, but not severe enough to indicate default. |
    | Pending Rescind                 | Loan application may be canceled. |
    | Withdrawn Application           | Borrower or lender withdrew the application. |
    | Voided New Loan                 | Loan approved but later canceled before use. |
    | CSR Voided New Loan             | Customer service team voided the loan. |
    | Customer Voided New Loan        | Borrower voluntarily canceled the loan. |
    | Settlement Pending Paid Off     | Borrower agreed to a settlement but has not completed full payment yet. |

    The **Loan Risk Predictor** simplifies this classification into **High Risk** or **Low/Medium Risk** for decision-making purposes while maintaining transparency by presenting the original loan statuses.

    #### User Manual
    To understand how to use the **Loan Risk Predictor** in detail, please read the [Loan Risk Predictor User Manual](https://drive.google.com/file/d/19jXwrDRwvzRH8blj8xpRkgY_afelRTxU/view?usp=share_link).  
""")


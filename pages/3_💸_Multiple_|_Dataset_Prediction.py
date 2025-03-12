import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

st.set_page_config(
    page_title='Loan Risk Predictor',
    layout='centered',
    page_icon='💰'
)

data = None;

st.title('Multiple/Dataset Prediction')

st.markdown(
    """
    #### Important Note
    Before uploading your CSV file, ensure it contains the following column headers in the correct order with values in the accepted format as described below. You can also download the CSV template below to input your loan application data.

    | **Column Name**         | **Accepted Values**                                       | **Values Description**                                                                                   |
    |--------------------------|----------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
    | state                   | **KY, IL, SC, CT, OH, LA, IN, NV, WI, NC, NJ, VA, PA, TX, MO, TN, FL, CO, MI, UT, CA, KS, MN, AZ, WA, DE, NM, MS, AL, OK, SD, NE, IA, ID, ND, WY, GA, AK, RI, HI, MD** | Client's US state of residence. |
    | loan_count              | **1 - 30**                                               | Total previous loan counts applied by the client (including the current one). |
    | nPaidOff                | **0 - 25**                                               | Number of MoneyLion loans the client has successfully paid off in the past. |
    | leadType                | **lead, bvMandatory, rc_returning, organic, express, prescreen, california, lionpay, instant-offer, repeat** | Lead type determining underwriting rules for a lead. |
    | payFrequency            | **B, W, S, M, I**                                        | Repayment frequency of the loan: **B** (Bi-weekly), **W** (Weekly), **S** (Semi-monthly), **M** (Monthly), **I** (Irregular). |
    | installmentIndex        | **1 - 100**                                              | Counts the nth payment for the loan. First payment is **1**, second is **2**, and so on. In other words, this represents the highest number of installments successfully paid in a single previous loan by the client. |
    | fpStatus                | **Checked, Rejected, Cancelled, Skipped, Pending, Returned** | Status of the first payment of the loan. |
    | paymentStatus           | **Cancelled, Checked, Rejected**                         | Current payment status of the loan. |
    | paymentAmount           | **Positive real numbers up to 2 decimal places (e.g., 150.75, 200.50)** | Total amount of payments made from all previous loans (fees + principal). |
    | CF.inquiry.1_min_ago    | **1 - 50**                                               | Number of unique inquiries for the consumer seen by Clarity in the last 1 minute. |
    | CF.inquiry.15_days_ago  | **1 - 100**                                              | Number of unique inquiries for the consumer seen by Clarity in the last 15 days. |
    | CF.inquiry.30_days_ago  | **1 - 150**                                              | Number of unique inquiries for the consumer seen by Clarity in the last 30 days. |
    | CF.inquiry.90_days_ago  | **1 - 250**                                              | Number of unique inquiries for the consumer seen by Clarity in the last 90 days. |
    | CF.inquiry.365_days_ago | **1 - 600**                                              | Number of unique inquiries for the consumer seen by Clarity in the last 365 days. |
    | clearfraudscore         | **100 - 1000**                                           | Fraud score provided by Clarity. A higher score suggests a lower default probability. |
    """
)

def convert_df(df):
    return df.to_csv(index=False)


template = pd.DataFrame(columns=[
    'loan_count', 
    'installmentIndex', 
    'paymentAmount', 
    'nPaidOff', 
    'CF.inquiry.30_days_ago', 
    'CF.inquiry.1_min_ago', 
    'CF.inquiry.90_days_ago', 
    'CF.inquiry.15_days_ago', 
    'CF.inquiry.365_days_ago', 
    'clearfraudscore', 
    'paymentStatus', 
    'leadType', 
    'payFrequency', 
    'state', 
    'fpStatus'
])
template = convert_df(template)

st.download_button(
            label='Download CSV template',
            data=template,
            file_name='Template.csv',
            mime='text/csv'
        )

def predict(data):
    if data is None:
        st.error("Please upload a CSV file before prediction.")
        return

    # Check for missing columns or empty values
    required_columns = [
        'loan_count', 'installmentIndex', 'paymentAmount', 'nPaidOff', 
        'CF.inquiry.30_days_ago', 'CF.inquiry.1_min_ago', 'CF.inquiry.90_days_ago', 
        'CF.inquiry.15_days_ago', 'CF.inquiry.365_days_ago', 'clearfraudscore', 
        'paymentStatus', 'leadType', 'payFrequency', 'state', 'fpStatus'
    ]

    missing_columns = [col for col in required_columns if col not in data.columns]
    empty_columns = [col for col in data.columns if data[col].isnull().any()]

    if missing_columns:
        st.error(f"The uploaded data is missing required column(s): {', '.join(missing_columns)}.")
        return
    if empty_columns:
        st.error(f"The uploaded data contains empty values in column(s): {', '.join(empty_columns)}.")
        return

    st.markdown("## Prediction Results Preview")

    # Standardize column names
    data.columns = [
        'loan_count', 'installmentIndex', 'paymentAmount', 'nPaidOff', 
        'CF.inquiry.30_days_ago', 'CF.inquiry.1_min_ago', 'CF.inquiry.90_days_ago', 
        'CF.inquiry.15_days_ago', 'CF.inquiry.365_days_ago', 'clearfraudscore', 
        'paymentStatus', 'leadType', 'payFrequency', 'state', 'fpStatus'
    ]

    # Load the trained model
    with open("classifier_lgbm_model.pkl", "rb") as f:
        loaded_model = pickle.load(f)

    # Load preprocessing objects (LabelEncoder, TargetEncoder, ColumnTransformer, StandardScaler)
    with open("preprocessor.pkl", "rb") as f:
        le, te, ct, sc = pickle.load(f)
    
    
    # Preprocess the data for batch prediction
    input_data = data.copy(deep=True)

    # Apply Target Encoding on 'state' column
    input_data["state"] = te.transform(input_data["state"].values.reshape(-1, 1))
    #input_data["state"] = te.transform(input_data["state"])  # Use pre-fitted encoder

    # Apply One-Hot Encoding & Feature Transformation using ColumnTransformer
    sample_data_transformed = ct.transform(input_data)

    # Standardize the transformed data
    sample_data_scaled = sc.transform(sample_data_transformed)

    # Perform batch predictions
    predictions = loaded_model.predict(sample_data_scaled)
    probabilities = loaded_model.predict_proba(sample_data_scaled)

    # Assign predictions to the original data
    output_data = input_data.assign(Prediction=predictions)
    output_data["Prediction"] = predictions
    output_data["Probability"] = probabilities.max(axis=1)
    output_data["Prediction"] = output_data["Prediction"].map({0: "Low/Medium Risk", 1: "High Risk"})
    
    
    # Define a function to apply custom styling based on 'Prediction' column
    def color_prediction_row(row):
        if row['Prediction'] == 'Default':
            return ['background-color: tomato'] * len(row)  # Highlight risky loans
        else:
            return ['background-color: lightgreen'] * len(row)  # Highlight safe loans

    # Apply the color styling
    styled_output_data = output_data.style.apply(color_prediction_row, axis=1)

    # Display the styled table in Streamlit
    st.dataframe(styled_output_data)

    # Downloadable CSV
    csv = convert_df(output_data)

    st.download_button(
        label='Download Prediction Results',
        data=csv,
        file_name='Loan_Prediction_Results.csv',
        mime='text/csv'
    )

    # Visualize results
    prediction_counts = output_data['Prediction'].value_counts()
    colors = ['lightgreen', 'tomato']
    fig = go.Figure(data=[go.Pie(labels=prediction_counts.index,
                                 values=prediction_counts.values, 
                                 hole=.3, textinfo='label+percent')])
    fig.update_traces(marker=dict(colors=colors))

    st.plotly_chart(fig, theme='streamlit')

csv_file = st.file_uploader("Choose a CSV file", type='csv')

if csv_file is not None:
    data = pd.read_csv(csv_file)

if st.button('Batch predict'):
    predict(data)

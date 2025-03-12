# +
import streamlit as st
import pandas as pd
import numpy as np
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import lines

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from category_encoders import TargetEncoder
from lightgbm import LGBMClassifier
# -
st.set_page_config(layout="centered", page_title='Loan Risk Predictor', page_icon='💰')

if 'result' not in st.session_state:
    st.session_state.result = 100

if 'prediction_outcome' not in st.session_state:
    st.session_state.prediction_outcome = ''

if 'prediction_probability' not in st.session_state:
    st.session_state.prediction_probability = 0

st.title("Single Prediction")

col1, col2 = st.columns(2, gap='small')

with st.form("Single Prediction", clear_on_submit=True):
    col1, col2 = st.columns(2)
    
    with col1:
        state = st.selectbox("Select State:", 
                             options=["Kentucky (KY)", "Illinois (IL)", "South Carolina (SC)", "Connecticut (CT)",
                                      "Ohio (OH)", "Louisiana (LA)", "Indiana (IN)", "Nevada (NV)", "Wisconsin (WI)",
                                      "North Carolina (NC)", "New Jersey (NJ)", "Virginia (VA)", "Pennsylvania (PA)",
                                      "Texas (TX)", "Missouri (MO)", "Tennessee (TN)", "Florida (FL)", "Colorado (CO)",
                                      "Michigan (MI)", "Utah (UT)", "California (CA)", "Kansas (KS)", "Minnesota (MN)",
                                      "Arizona (AZ)", "Washington (WA)", "Delaware (DE)", "New Mexico (NM)", "Mississippi (MS)",
                                      "Alabama (AL)", "Oklahoma (OK)", "South Dakota (SD)", "Nebraska (NE)", "Iowa (IA)", "Idaho (ID)",
                                      "North Dakota (ND)", "Wyoming (WY)", "Georgia (GA)", "Alaska (AK)", "Rhode Island (RI)",
                                      "Hawaii (HI)", "Maryland (MD)"])
        
        loan_count = st.number_input("Total Previous Loan Applications:", min_value=1, max_value=30, step=1)
        nPaidOff = st.number_input("Number of Loans Paid Off:", min_value=0, max_value=25, step=1)
        
        lead_type = st.selectbox("Lead Type:", 
                                 options=["Lead", "BV Mandatory", "Returning Customer", "Organic", "Express", 
                                          "Prescreen", "California", "LionPay", "Instant Offer", "Repeat"])
        
        pay_frequency = st.radio("Payment Frequency:", 
                                 options=["Bi-weekly", "Weekly", "Semi-monthly", "Monthly", "Irregular"])
        
        installment_index = st.number_input("Max Installments (Index) Paid in Any Loan:", min_value=1, max_value=100, step=1)
    
    with col2:
        fp_status = st.radio("First Payment Status:", 
                             options=["Checked", "Rejected", "Cancelled", "Skipped", "Pending", "Returned"])
        
        payment_status = st.radio("Current Payment Status:", 
                                  options=["Cancelled", "Checked", "Rejected"])
        
        payment_amount = st.number_input("Total Previous Payments Made (Fees + Principal):", min_value=0.0, step=0.01, format="%.2f")
        
        cf_inquiry_1_min = st.number_input("Clarity Inquiries (Last 1 Min):", min_value=1, max_value=50, step=1)
        cf_inquiry_15_days = st.number_input("Clarity Inquiries (Last 15 Days):", min_value=1, max_value=100, step=1)
        cf_inquiry_30_days = st.number_input("Clarity Inquiries (Last 30 Days):", min_value=1, max_value=150, step=1)
        cf_inquiry_90_days = st.number_input("Clarity Inquiries (Last 90 Days):", min_value=1, max_value=250, step=1)
        cf_inquiry_365_days = st.number_input("Clarity Inquiries (Last 365 Days):", min_value=1, max_value=600, step=1)
        
        clearfraudscore = st.number_input("Clear Fraud Score:", min_value=100, max_value=1000, step=1)
    
    submitted = st.form_submit_button("Predict")
    
    if submitted:
        # Map state abbreviations
        state_mapping = {s.split(" (")[0]: s.split(" (")[1][:-1] for s in st.session_state.keys() if "(" in s}
        state = state_mapping.get(state, state)
        
        # Lead type mapping
        lead_type_mapping = {
            "Lead": "lead", "BV Mandatory": "bvMandatory", "Returning Customer": "rc_returning", 
            "Organic": "organic", "Express": "express", "Prescreen": "prescreen", "California": "california", 
            "LionPay": "lionpay", "Instant Offer": "instant-offer", "Repeat": "repeat"
        }
        lead_type = lead_type_mapping.get(lead_type, lead_type)
        
        # Payment frequency mapping
        pay_frequency_mapping = {"Bi-weekly": "B", "Weekly": "W", "Semi-monthly": "S", "Monthly": "M", "Irregular": "I"}
        pay_frequency = pay_frequency_mapping.get(pay_frequency, pay_frequency)
        
        # Create input data dictionary in the specified order
        input_data = {
            "loan_count": [loan_count],
            "installmentIndex": [installment_index],
            "paymentAmount": [payment_amount],
            "nPaidOff": [nPaidOff],
            "CF.inquiry.30_days_ago": [cf_inquiry_30_days],
            "CF.inquiry.1_min_ago": [cf_inquiry_1_min],
            "CF.inquiry.90_days_ago": [cf_inquiry_90_days],
            "CF.inquiry.15_days_ago": [cf_inquiry_15_days],
            "CF.inquiry.365_days_ago": [cf_inquiry_365_days],
            "clearfraudscore": [clearfraudscore],
            "paymentStatus": [payment_status],
            "leadType": [lead_type],
            "payFrequency": [pay_frequency],
            "state": [state],
            "fpStatus": [fp_status]
        }
        
        # Convert the dictionary into a DataFrame
        df = pd.DataFrame(input_data)

        # Load the trained model
        with open("classifier_lgbm_model.pkl", "rb") as f:
            loaded_model = pickle.load(f)

        # Load preprocessing objects (LabelEncoder, TargetEncoder, ColumnTransformer, StandardScaler)
        with open("preprocessor.pkl", "rb") as f:
            le, te, ct, sc = pickle.load(f)

        # Apply Target Encoding on 'state' column
        #df["state"] = te.transform(df[["state"]])  # Directly apply transformation using pre-fitted encoder
        df["state"] = te.transform(df["state"].values.reshape(-1, 1))

        # Apply Target Encoding on 'state' column (Index 13)
        #df.iloc[:, 13] = te.transform(df.iloc[:, 13])

        # Apply One-Hot Encoding & Feature Transformation
        sample_data_transformed = ct.transform(df)

        # Standardization (scaling)
        sample_data_scaled = sc.transform(sample_data_transformed)

        # Make predictions
        prediction = loaded_model.predict(sample_data_scaled)
        st.session_state.prediction_probability = loaded_model.predict_proba(sample_data_scaled)

        # Display results
        if prediction == 0:
            st.session_state.result = 0
            st.session_state.prediction_outcome = "Low/Medium Risk"
        else:
            st.session_state.result = 1
            st.session_state.prediction_outcome = "High Risk"

image_path = ''
col3, col4 = st.columns(2)

with col3:
    if st.session_state.result == 0:
        st.subheader("Prediction Result")
        image_path = 'loan_repayment.png'
        st.image(image_path, width=200)
    elif st.session_state.result == 1:
        st.subheader("Prediction Result")
        image_path = 'loan_default.png'
        st.image(image_path, width=200)

with col4:
    if st.session_state.result == 0:
        st.header(f':green[{st.session_state.prediction_outcome}]')  
        st.subheader(f"You have {st.session_state.prediction_probability[0, 0] * 100:.0f}% probability of loan **being approved**.")
    elif st.session_state.result == 1:
        st.header(f':red[{st.session_state.prediction_outcome}]')
        st.subheader(f"You have {st.session_state.prediction_probability[0, 1] * 100:.0f}% probability of loan **not being approved**.")

if st.session_state.result == 0 or st.session_state.result == 1:
    # Trigger for the pop-up
    if st.button("Prediction Insights!"):
        with st.expander("Machine Learning \"Black Box\"", expanded=True):
            st.write("Let's explore how the LightGBM model made predictions by analyzing the feature importance of each variable!")

            # Load the pre-trained LightGBM model
            with open('classifier_lgbm_model.pkl', 'rb') as f:
                loaded_model = pickle.load(f)

            # Define feature names corresponding to the columns in the training dataset
            feature_names = [
                'paymentStatus_Cancelled', 'paymentStatus_Checked', 'paymentStatus_Rejected',
                'leadType_bvMandatory', 'leadType_california', 'leadType_express', 'leadType_instant-offer',
                'leadType_lead', 'leadType_lionpay', 'leadType_organic', 'leadType_prescreen',
                'leadType_rc_returning', 'leadType_repeat', 'payFrequency_B', 'payFrequency_I',
                'payFrequency_M', 'payFrequency_S', 'payFrequency_W', 'fpStatus_Cancelled',
                'fpStatus_Checked', 'fpStatus_Pending', 'fpStatus_Rejected', 'fpStatus_Returned',
                'fpStatus_Skipped', 'loan_count', 'installmentIndex', 'paymentAmount', 'nPaidOff',
                'CF.inquiry.30_days_ago', 'CF.inquiry.1_min_ago', 'CF.inquiry.90_days_ago',
                'CF.inquiry.15_days_ago', 'CF.inquiry.365_days_ago', 'clearfraudscore', 'state'
            ]

            # Create DataFrame with feature names and importances
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': loaded_model.feature_importances_
            })

            # Sort features by importance and keep only the top 15
            feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(15).reset_index(drop=True)

            # Visualization setup
            background_color = "#fbfbfb"
            fig, ax = plt.subplots(1, 1, figsize=(10, 8), facecolor=background_color)

            # Color mapping: Highlight top 3 features
            color_map = ['lightgray' for _ in range(len(feature_importance_df))]
            color_map[0] = color_map[1] = color_map[2] = '#0f4c81'  # Highlight top 3 features

            # Bar plot of feature importance
            sns.barplot(
                data=feature_importance_df,
                x='Importance',
                y='Feature',
                ax=ax,
                palette=color_map
            )

            # Adjust plot aesthetics
            ax.set_facecolor(background_color)
            for spine in ['top', 'left', 'right']:
                ax.spines[spine].set_visible(False)

            fig.text(
                0.12, 0.92,
                "Feature Importance: LightGBM Loan Prediction",
                fontsize=18, fontweight='bold', fontfamily='serif'
            )

            fig.text(
                1.03, 0.92,
                "Insight",
                fontsize=18, fontweight='bold', fontfamily='serif'
            )

            fig.text(
                1.0, 0.315,
                '''
                Understanding feature importance helps explain
                model decisions and improve interpretability.

                The most influential factors are payment amount
                and installment index, highlighting the impact
                of repayment behavior on loan approval.

                Fraud detection is also critical, as indicated
                by the high importance of the fraud score.

                Credit inquiries over different timeframes
                play a key role, with long-term history
                (365 days) being the most significant.

                Additionally, payment status and frequency
                contribute to the model's decision-making,
                influencing risk assessment.
                ''',
                fontsize=14, fontweight='light', fontfamily='serif'
            )

            ax.tick_params(axis='both', which='both', length=0)

            # Add vertical line
            l1 = lines.Line2D(
                [0.98, 0.98],
                [0, 1],
                transform=fig.transFigure,
                figure=fig,
                color='black',
                lw=0.2
            )
            fig.lines.extend([l1])

            plt.xlabel("", fontsize=12, fontweight='light', fontfamily='serif', loc='left')
            plt.ylabel("", fontsize=12, fontweight='light', fontfamily='serif')

            # Integrate with Streamlit
            st.title("Feature Importance Visualization")
            st.pyplot(fig)

            st.button("Close")  # Add a button for users to "close" it

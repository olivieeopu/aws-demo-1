import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px


# LOAD FILES
model = joblib.load("model_artifact/model.joblib")
scaler = joblib.load("model_artifact/scaler.joblib")
encoder = joblib.load("model_artifact/encoder.joblib")
feature_names = joblib.load("model_artifact/feature_names.joblib")

# PAGE CONFIG
st.set_page_config(
    page_title="Credit Score Prediction",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Score Prediction")

# GET CATEGORIES FROM ENCODER
occupation_categories = list(encoder.categories_[0])
credit_mix_categories = list(encoder.categories_[1])
payment_min_categories = list(encoder.categories_[2])
payment_behaviour_categories = list(encoder.categories_[3])

# PERSONAL INFORMATION
st.header("👤 Personal Information")

col1, col2 = st.columns(2)

with col1:

    occupation = st.selectbox(
        "Occupation",
        occupation_categories
    )

with col2:

    annual_income = st.number_input(
        "Annual Income",
        min_value=1.0,
        value=50000.0
    )

# CREDIT INFORMATION
st.header("🏦 Credit Information")

col1, col2 = st.columns(2)

with col1:

    num_bank_accounts = st.number_input(
        "Number of Bank Accounts",
        min_value=0.0,
        value=4.0
    )

    num_credit_card = st.number_input(
        "Number of Credit Cards",
        min_value=0.0,
        value=3.0
    )

    interest_rate = st.number_input(
        "Interest Rate",
        min_value=0.0,
        value=8.0
    )

with col2:

    num_of_loan = st.number_input(
        "Number of Loans",
        min_value=0.0,
        value=1.0
    )

    credit_history_age = st.number_input(
        "Credit History Age (Months)",
        min_value=0.0,
        value=120.0
    )

    changed_credit_limit = st.number_input(
        "Changed Credit Limit",
        value=10.0
    )

# PAYMENT INFORMATION
st.header("💰 Payment Information")

col1, col2 = st.columns(2)

with col1:

    delay_from_due_date = st.number_input(
        "Delay From Due Date",
        min_value=0.0,
        value=5.0
    )

    num_delayed_payment = st.number_input(
        "Number of Delayed Payments",
        min_value=0.0,
        value=2.0
    )

    num_credit_inquiries = st.number_input(
        "Number of Credit Inquiries",
        min_value=0.0,
        value=2.0
    )

with col2:

    payment_of_min_amount = st.selectbox(
        "Payment of Minimum Amount",
        payment_min_categories
    )

    payment_behaviour = st.selectbox(
        "Payment Behaviour",
        payment_behaviour_categories
    )

# DEBT INFORMATION
st.header("📊 Debt Information")

col1, col2 = st.columns(2)

with col1:

    outstanding_debt = st.number_input(
        "Outstanding Debt",
        min_value=0.0,
        value=1000.0
    )

with col2:

    credit_mix = st.selectbox(
        "Credit Mix",
        credit_mix_categories
    )

# AUTO FEATURE ENGINEERING
debt_to_income_ratio = (
    outstanding_debt / annual_income
    if annual_income > 0
    else 0
)

risk_score = (
    0.5 * delay_from_due_date
    +
    0.3 * num_delayed_payment
    +
    0.2 * num_credit_inquiries
)

financial_exposure = (
    num_bank_accounts
    +
    num_credit_card
    +
    num_of_loan
)

# PREDICT BUTTON
st.markdown("---")

if st.button("🚀 Predict Credit Score"):

    input_df = pd.DataFrame({

        "Occupation":
            [occupation],

        "Num_Bank_Accounts":
            [num_bank_accounts],

        "Num_Credit_Card":
            [num_credit_card],

        "Interest_Rate":
            [interest_rate],

        "Num_of_Loan":
            [num_of_loan],

        "Delay_from_due_date":
            [delay_from_due_date],

        "Num_of_Delayed_Payment":
            [num_delayed_payment],

        "Changed_Credit_Limit":
            [changed_credit_limit],

        "Num_Credit_Inquiries":
            [num_credit_inquiries],

        "Credit_Mix":
            [credit_mix],

        "Outstanding_Debt":
            [outstanding_debt],

        "Credit_History_Age":
            [credit_history_age],

        "Payment_of_Min_Amount":
            [payment_of_min_amount],

        "Payment_Behaviour":
            [payment_behaviour],

        "Debt_to_Income_Ratio":
            [debt_to_income_ratio],

        "Risk_Score":
            [risk_score],

        "Financial_Exposure":
            [financial_exposure]
    })

    categorical_cols = [
        "Occupation",
        "Credit_Mix",
        "Payment_of_Min_Amount",
        "Payment_Behaviour"
    ]

    encoded = encoder.transform(
        input_df[categorical_cols]
    )

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(
            categorical_cols
        )
    )

    final_df = pd.concat(
        [
            input_df.drop(
                columns=categorical_cols
            ),
            encoded_df
        ],
        axis=1
    )

    final_df = final_df.reindex(
        columns=feature_names,
        fill_value=0
    )

    scaled_input = scaler.transform(
        final_df
    )

    prediction = model.predict(
        scaled_input
    )[0]

    probabilities = model.predict_proba(
        scaled_input
    )[0]

    label_map = {
        0: "Poor",
        1: "Standard",
        2: "Good"
    }

    prediction_label = label_map[
        prediction
    ]

    confidence = np.max(
        probabilities
    )

    st.markdown("---")

    st.header("📈 Prediction Result")

    if prediction == 0:

        st.error(
            f"🔴 {prediction_label}"
        )

    elif prediction == 1:

        st.warning(
            f"🟡 {prediction_label}"
        )

    else:

        st.success(
            f"🟢 {prediction_label}"
        )

    st.metric(
        "Prediction Confidence",
        f"{confidence:.2%}"
    )

    # PROBABILITY CHART

    prob_df = pd.DataFrame({

        "Class": [
            "Poor",
            "Standard",
            "Good"
        ],

        "Probability":
            probabilities
    })

    fig = px.bar(
        prob_df,
        x="Probability",
        y="Class",
        orientation="h",
        text="Probability",
        height=280
    )

    fig.update_traces(
        texttemplate="%{text:.2%}",
        textposition="outside"
    )

    fig.update_layout(
        showlegend=False,
        xaxis_range=[0, 1],
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Models and Scaler
# -----------------------------
log_model = joblib.load("logistic_model.pkl")
nb_model = joblib.load("naive_bayes_model.pkl")
knn_model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Loan Approval Prediction System")
st.write("Enter applicant details to predict loan approval.")


# -----------------------------
# Model Selection
# -----------------------------
model_name = st.selectbox(
    "Select Machine Learning Model",
    [
        "Logistic Regression",
        "Naive Bayes",
        "KNN"
    ]
)


# -----------------------------
# Input Fields
# -----------------------------

col1, col2 = st.columns(2)

with col1:

    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0.0,
        value=50000.0
    )

    coapplicant_income = st.number_input(
        "Coapplicant Income",
        min_value=0.0,
        value=0.0
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    dependents = st.number_input(
        "Dependents",
        min_value=0,
        max_value=10,
        value=0
    )

    existing_loans = st.number_input(
        "Existing Loans",
        min_value=0,
        max_value=20,
        value=0
    )

    savings = st.number_input(
        "Savings",
        min_value=0.0,
        value=100000.0
    )

    collateral_value = st.number_input(
        "Collateral Value",
        min_value=0.0,
        value=200000.0
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0.0,
        value=100000.0
    )

    loan_term = st.number_input(
        "Loan Term",
        min_value=1,
        max_value=480,
        value=120
    )

    education_level = st.selectbox(
        "Education Level",
        ["Graduate", "Not Graduate"]
    )

    employment_status = st.selectbox(
        "Employment Status",
        ["Salaried", "Self-employed", "Unemployed"]
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["Single", "Married"]
    )


with col2:

    loan_purpose = st.selectbox(
        "Loan Purpose",
        ["Car", "Education", "Home", "Personal"]
    )

    property_area = st.selectbox(
        "Property Area",
        ["Semiurban", "Urban"]
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    employer_category = st.selectbox(
        "Employer Category",
        [
            "Government",
            "MNC",
            "Private",
            "Unemployed"
        ]
    )

    dti_ratio = st.number_input(
        "DTI Ratio",
        min_value=0.0,
        value=0.30
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=700
    )


# -----------------------------
# Prediction
# -----------------------------

if st.button("🔮 Predict Loan Approval"):

    # Create input dictionary
    input_data = {
        "Applicant_Income": applicant_income,
        "Coapplicant_Income": coapplicant_income,
        "Age": age,
        "Dependents": dependents,
        "Existing_Loans": existing_loans,
        "Savings": savings,
        "Collateral_Value": collateral_value,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,

        "Education_Level": 1 if education_level == "Graduate" else 0,

        "Employment_Status_Salaried":
            1 if employment_status == "Salaried" else 0,

        "Employment_Status_Self-employed":
            1 if employment_status == "Self-employed" else 0,

        "Employment_Status_Unemployed":
            1 if employment_status == "Unemployed" else 0,

        "Marital_Status_Single":
            1 if marital_status == "Single" else 0,

        "Loan_Purpose_Car":
            1 if loan_purpose == "Car" else 0,

        "Loan_Purpose_Education":
            1 if loan_purpose == "Education" else 0,

        "Loan_Purpose_Home":
            1 if loan_purpose == "Home" else 0,

        "Loan_Purpose_Personal":
            1 if loan_purpose == "Personal" else 0,

        "Property_Area_Semiurban":
            1 if property_area == "Semiurban" else 0,

        "Property_Area_Urban":
            1 if property_area == "Urban" else 0,

        "Gender_Male":
            1 if gender == "Male" else 0,

        "Employer_Category_Government":
            1 if employer_category == "Government" else 0,

        "Employer_Category_MNC":
            1 if employer_category == "MNC" else 0,

        "Employer_Category_Private":
            1 if employer_category == "Private" else 0,

        "Employer_Category_Unemployed":
            1 if employer_category == "Unemployed" else 0,

        "DTI_Ratio_sq": dti_ratio ** 2,

        "Credit_Score_sq": credit_score ** 2
    }


    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])


    # Make sure column order is exactly same as training
    feature_columns = [
        'Applicant_Income',
        'Coapplicant_Income',
        'Age',
        'Dependents',
        'Existing_Loans',
        'Savings',
        'Collateral_Value',
        'Loan_Amount',
        'Loan_Term',
        'Education_Level',
        'Employment_Status_Salaried',
        'Employment_Status_Self-employed',
        'Employment_Status_Unemployed',
        'Marital_Status_Single',
        'Loan_Purpose_Car',
        'Loan_Purpose_Education',
        'Loan_Purpose_Home',
        'Loan_Purpose_Personal',
        'Property_Area_Semiurban',
        'Property_Area_Urban',
        'Gender_Male',
        'Employer_Category_Government',
        'Employer_Category_MNC',
        'Employer_Category_Private',
        'Employer_Category_Unemployed',
        'DTI_Ratio_sq',
        'Credit_Score_sq'
    ]

    input_df = input_df[feature_columns]


    # Scale input
    input_scaled = scaler.transform(input_df)


    # Select model
    if model_name == "Logistic Regression":
        model = log_model

    elif model_name == "Naive Bayes":
        model = nb_model

    else:
        model = knn_model


    # Prediction
    prediction = model.predict(input_scaled)[0]


    # Result
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")


    # Probability
    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(input_scaled)[0]

        st.write(
            f"Approval Probability: **{probability[1] * 100:.2f}%**"
        )
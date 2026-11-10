
import streamlit as st
import pandas as pd
import joblib

# -----------------------------------
# 1. Load saved machine learning files
# -----------------------------------

model = joblib.load("models/decision_tree_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/feature_names.pkl")

numerical_features = ["tenure", "MonthlyCharges", "TotalCharges"]

# -----------------------------------
# 2. Configure Streamlit page
# -----------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction")
st.write(
    "Enter customer information below to predict "
    "whether the customer is likely to churn."
)

st.divider()

# -----------------------------------
# 3. Customer information form
# -----------------------------------

with st.form("customer_form"):

    st.subheader("👤 Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner?", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents?", ["No", "Yes"])

    with col2:
        tenure = st.number_input(
            "Tenure (months)", min_value=0, max_value=72, value=12
        )
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )
        paperless = st.selectbox(
            "Paperless Billing", ["No", "Yes"]
        )
        payment = st.selectbox(
            "Payment Method",
            [
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Electronic check",
                "Mailed check"
            ]
        )

    with col3:
        monthly_charges = st.number_input(
            "Monthly Charges ($)", min_value=0.0,
            max_value=200.0, value=70.0
        )
        total_charges = st.number_input(
            "Total Charges ($)", min_value=0.0,
            value=840.0
        )
        phone = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox(
            "Multiple Lines", ["No", "Yes"]
        )

    st.divider()
    st.subheader("🌐 Internet and Additional Services")

    col4, col5, col6 = st.columns(3)

    with col4:
        internet = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox(
            "Online Security", ["No", "Yes", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online Backup", ["No", "Yes", "No internet service"]
        )

    with col5:
        device_protection = st.selectbox(
            "Device Protection", ["No", "Yes", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech Support", ["No", "Yes", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV", ["No", "Yes", "No internet service"]
        )

    with col6:
        streaming_movies = st.selectbox(
            "Streaming Movies", ["No", "Yes", "No internet service"]
        )

    submitted = st.form_submit_button(
        "🔍 Predict Customer Churn",
        use_container_width=True
    )

# -----------------------------------
# 4. Prepare customer data
# -----------------------------------

if submitted:

    customer_data = {feature: 0 for feature in feature_names}

    # Basic information
    customer_data["SeniorCitizen"] = int(senior == "Yes")
    customer_data["tenure"] = tenure
    customer_data["MonthlyCharges"] = monthly_charges
    customer_data["TotalCharges"] = total_charges

    # Categorical features
    mappings = {
        "gender_Male": gender == "Male",
        "Partner_Yes": partner == "Yes",
        "Dependents_Yes": dependents == "Yes",
        "PhoneService_Yes": phone == "Yes",
        "MultipleLines_No phone service": phone == "No",
        "MultipleLines_Yes": multiple_lines == "Yes" and phone == "Yes",
        "InternetService_Fiber optic": internet == "Fiber optic",
        "InternetService_No": internet == "No",
        "OnlineSecurity_Yes": online_security == "Yes",
        "OnlineSecurity_No internet service": online_security == "No internet service",
        "OnlineBackup_Yes": online_backup == "Yes",
        "OnlineBackup_No internet service": online_backup == "No internet service",
        "DeviceProtection_Yes": device_protection == "Yes",
        "DeviceProtection_No internet service": device_protection == "No internet service",
        "TechSupport_Yes": tech_support == "Yes",
        "TechSupport_No internet service": tech_support == "No internet service",
        "StreamingTV_Yes": streaming_tv == "Yes",
        "StreamingTV_No internet service": streaming_tv == "No internet service",
        "StreamingMovies_Yes": streaming_movies == "Yes",
        "StreamingMovies_No internet service": streaming_movies == "No internet service",
        "Contract_One year": contract == "One year",
        "Contract_Two year": contract == "Two year",
        "PaperlessBilling_Yes": paperless == "Yes",
        "PaymentMethod_Credit card (automatic)": payment == "Credit card (automatic)",
        "PaymentMethod_Electronic check": payment == "Electronic check",
        "PaymentMethod_Mailed check": payment == "Mailed check"
    }

    for feature, value in mappings.items():
        if feature in customer_data:
            customer_data[feature] = int(value)

    # Convert to DataFrame in the exact training feature order
    input_df = pd.DataFrame([customer_data], columns=feature_names)

    # Apply the same scaler used during model training
    input_df[numerical_features] = scaler.transform(
        input_df[numerical_features]
    )

    # -----------------------------------
    # 5. Make prediction
    # -----------------------------------

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    churn_probability = probabilities[1] * 100

    st.divider()
    st.subheader("📈 Prediction Results")

    if prediction == 1:
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is not likely to churn.")

    st.metric(
        "Predicted Churn Probability",
        f"{churn_probability:.2f}%"
    )

    st.progress(float(probabilities[1]))

    st.caption(
        "Prediction generated using the trained Decision Tree model."
    )

%%writefile app.py

import streamlit as st
import pandas as pd
import joblib

# Load full pipeline (preprocessing + model)
model = joblib.load("HR_Attrition_ML.pkl")

st.title("HR Attrition Prediction")
st.write("Enter employee details to predict attrition")

# -------- INPUTS -------- #
age = st.number_input("Age", 18, 65, 30)
monthly_income = st.number_input("Monthly Income", 1000, 200000, 30000)
distance_from_home = st.number_input("Distance From Home (KM)", 0, 100, 10)
years_at_company = st.number_input("Years At Company", 0, 40, 5)

job_satisfaction = st.selectbox("Job Satisfaction (1=Low, 4=High)", [1,2,3,4])
work_life_balance = st.selectbox("Work Life Balance (1=Bad, 4=Excellent)", [1,2,3,4])

overtime = st.selectbox("OverTime", ["Yes", "No"])
gender = st.selectbox("Gender", ["Male", "Female"])
department = st.selectbox("Department", ["Sales", "HR", "R&D"])
job_role = st.selectbox("Job Role", ["Manager", "Developer", "Analyst"])

# -------- CREATE DATAFRAME (RAW FEATURES) -------- #
input_data = pd.DataFrame([{
    "Age": age,
    "MonthlyIncome": monthly_income,
    "DistanceFromHome": distance_from_home,
    "YearsAtCompany": years_at_company,
    "JobSatisfaction": job_satisfaction,
    "WorkLifeBalance": work_life_balance,
    "OverTime": overtime,
    "Gender": gender,
    "Department": department,
    "JobRole": job_role
}])

# -------- PREDICTION -------- #
if st.button("Predict Attrition"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠ Employee likely to leave (Risk: {probability*100:.2f}%)")
    else:
        st.success(f"✅ Employee likely to stay (Risk: {probability*100:.2f}%)")

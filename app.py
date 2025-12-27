
import streamlit as st
import numpy as np
import joblib

model = joblib.load("HR_Attrition_ML.pkl")

st.title("HR Attrition Prediction")
st.write("Enter employee details to predict attrition")

age = st.number_input("Age", min_value=18, max_value=65, value=30)
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=200000, value=30000)
distance_from_home = st.number_input("Distance From Home (KM)", min_value=0, max_value=100, value=10)
years_at_company = st.number_input("Years At Company", min_value=0, max_value=40, value=5)

job_satisfaction = st.selectbox("Job Satisfaction (1=Low, 4=High)", [1, 2, 3, 4])
work_life_balance = st.selectbox("Work Life Balance (1=Bad, 4=Excellent)", [1, 2, 3, 4])

overtime = st.selectbox("OverTime", ["Yes", "No"])
gender = st.selectbox("Gender", ["Male", "Female"])
department = st.selectbox("Department", ["Sales", "HR", "R&D"])
job_role = st.selectbox("Job Role", ["Manager", "Developer", "Analyst"])

overtime = 1 if overtime == "Yes" else 0
gender = 1 if gender == "Male" else 0

department_map = {"Sales": 0, "HR": 1, "R&D": 2}
jobrole_map = {"Manager": 0, "Developer": 1, "Analyst": 2}

department = department_map[department]
job_role = jobrole_map[job_role]

if st.button("Predict Attrition"):

    X = np.array([[age,
                   monthly_income,
                   distance_from_home,
                   years_at_company,
                   job_satisfaction,
                   work_life_balance,
                   overtime,
                   gender,
                   department,
                   job_role]])

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    if prediction == 1:
        st.error(f"⚠ Employee likely to leave (Risk: {probability*100:.2f}%)")
    else:
        st.success(f"✅ Employee likely to stay (Risk: {probability*100:.2f}%)")

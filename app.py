%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("HR_Attrition_ML.pkl")

st.title("HR Attrition Prediction (Excel Upload)")
st.write("Upload an Excel file to predict employee attrition")

# Upload Excel file
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    # Encoding mappings
    department_map = {"Sales": 0, "HR": 1, "R&D": 2}
    jobrole_map = {"Manager": 0, "Developer": 1, "Analyst": 2}

    # Preprocessing
    df["OverTime"] = df["OverTime"].map({"Yes": 1, "No": 0})
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
    df["Department"] = df["Department"].map(department_map)
    df["JobRole"] = df["JobRole"].map(jobrole_map)

    # Feature selection (order matters!)
    features = [
        "Age",
        "MonthlyIncome",
        "DistanceFromHome",
        "YearsAtCompany",
        "JobSatisfaction",
        "WorkLifeBalance",
        "OverTime",
        "Gender",
        "Department",
        "JobRole"
    ]

    X = df[features].values

    if st.button("Predict Attrition"):
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]

        df["Attrition_Prediction"] = predictions
        df["Attrition_Risk_%"] = (probabilities * 100).round(2)

        df["Attrition_Prediction"] = df["Attrition_Prediction"].map(
            {1: "Likely to Leave", 0: "Likely to Stay"}
        )

        st.subheader("Prediction Results")
        st.dataframe(df)


import streamlit as st
import pandas as pd
import joblib

# Load model package
data = joblib.load("HR_Attrition_ML.pkl")
model = data["model"]
columns = data["columns"]
encoders = data["encoders"]

st.title("HR Attrition Prediction")
st.write("Fill employee details to predict attrition")

user_input = {}

# Dynamically create inputs based on training columns
for col in columns:
    if col in encoders:
        options = list(encoders[col].classes_)
        user_input[col] = st.selectbox(col, options)
    else:
        user_input[col] = st.number_input(col, value=0)

if st.button("Predict Attrition"):
    input_df = pd.DataFrame([user_input])

    # Encode categorical values
    for col, encoder in encoders.items():
        input_df[col] = encoder.transform(input_df[col])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠ Employee likely to leave (Risk: {probability*100:.2f}%)")
    else:
        st.success(f"✅ Employee likely to stay (Risk: {probability*100:.2f}%)")

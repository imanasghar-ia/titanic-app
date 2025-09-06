import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("titanic_model.pkl")

st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger details to predict survival:")

# Inputs from user
pclass = st.selectbox("Passenger Class", [1, 2, 3])
age = st.number_input("Age", 0, 100, 25)
fare = st.number_input("Fare", 0, 600, 30)

# Make prediction
if st.button("Predict"):
    input_df = pd.DataFrame([[pclass, age, fare]], columns=['Pclass','Age','Fare'])
    prediction = model.predict(input_df)[0]
    result = "✅ Survived" if prediction == 1 else "❌ Did not survive"
    st.subheader(result)

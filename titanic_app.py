import streamlit as st
import pandas as pd
import joblib
import requests
from io import BytesIO

st.title("🚢 Titanic Survival Prediction App")

# -----------------------------
# Function to download files from GitHub
# -----------------------------
def load_file_from_github(url, file_type="csv"):
    if file_type == "csv":
        return pd.read_csv(url)
    elif file_type == "pkl":
        response = requests.get(url)
        return joblib.load(BytesIO(response.content))

# -----------------------------
# URLs for your files on GitHub
# -----------------------------
# Replace these URLs with the "raw" GitHub links for your files
TRAIN_CSV_URL = "https://raw.githubusercontent.com/imanasghar-ia/titanic-app/main/train.csv"
MODEL_PKL_URL = "https://raw.githubusercontent.com/imanasghar-ia/titanic-app/main/titanic_model.pkl"

# -----------------------------
# Load dataset
# -----------------------------
st.subheader("Dataset Preview")
df = load_file_from_github(TRAIN_CSV_URL)
st.dataframe(df.head())

# -----------------------------
# Load model
# -----------------------------
model = load_file_from_github(MODEL_PKL_URL, file_type="pkl")

# -----------------------------
# User input for prediction
# -----------------------------
st.subheader("Predict Survival")
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=30)
sibsp = st.number_input("Siblings/Spouses aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.0)

# Convert user input into DataFrame
input_df = pd.DataFrame({
    "Pclass": [pclass],
    "Sex": [sex],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare]
})

# -----------------------------
# Encode categorical columns (if your model needs it)
# -----------------------------
input_df["Sex"] = input_df["Sex"].map({"male": 0, "female": 1})

# -----------------------------
# Make prediction
# -----------------------------
prediction = model.predict(input_df)[0]

st.write("✅ Prediction:", "Survived" if prediction == 1 else "Did not survive")

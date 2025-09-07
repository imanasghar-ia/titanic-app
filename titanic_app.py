import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# 1. Load Dataset & Model Locally
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("train.csv")   # Local dataset in repo

@st.cache_resource
def load_model():
    return joblib.load("titanic_model.pkl")   # Local model in repo

df = load_data()
model = load_model()
# -------------------------------
# 2. Streamlit App UI
# -------------------------------
st.set_page_config(page_title="🚢 Titanic Survival Prediction", layout="wide")
st.title("🚢 Titanic Survival Prediction App")

st.write("This app predicts whether a passenger survived or not based on input features.")

# -------------------------------
# 3. User Input
# -------------------------------
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", 0, 8, 0)
parch = st.number_input("Number of Parents/Children Aboard", 0, 6, 0)
fare = st.slider("Passenger Fare", 0, 500, 50)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# -------------------------------
# 4. Feature Engineering
# -------------------------------
# Match training preprocessing (you must adapt if your model used encodings/scalers)
sex_val = 1 if sex == "male" else 0
embarked_dict = {"C": 0, "Q": 1, "S": 2}
embarked_val = embarked_dict[embarked]

features = [[pclass, sex_val, age, sibsp, parch, fare, embarked_val]]

# -------------------------------
# 5. Prediction
# -------------------------------
if st.button("Predict Survival"):
    prediction = model.predict(features)
    result = "✅ Survived" if prediction[0] == 1 else "❌ Did Not Survive"
    st.subheader(f"Prediction: {result}")

# -------------------------------
# 6. Show Data
# -------------------------------
with st.expander("See Titanic Training Data"):
    st.dataframe(df.head())
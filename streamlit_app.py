import streamlit as st
import pandas as pd
import joblib

# ================== Load Model ==================
model = joblib.load("logreg_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("column.pkl")

# ================== Page Config ==================
st.set_page_config(page_title="❤️ Heart Stroke Predictor", page_icon="❤️", layout="centered")

# ================== Custom CSS ==================
st.markdown("""
    <style>
        /* Title Gradient */
        .title {
            font-size: 40px;
            font-weight: bold;
            background: -webkit-linear-gradient(45deg, #FF4B4B, #FF7B7B);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 20px;
        }
        /* Input card */
        .stSlider, .stSelectbox, .stNumberInput {
            background-color: #1E1E1E;
            padding: 15px;
            border-radius: 12px;
            margin-bottom: 10px;
        }
        /* Prediction result box */
        .result-box {
            padding: 20px;
            border-radius: 12px;
            font-size: 20px;
            font-weight: bold;
            text-align: center;
            margin-top: 20px;
        }
        .high-risk {
            background-color: #FF4B4B;
            color: white;
        }
        .low-risk {
            background-color: #2ECC71;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# ================== Title ==================
st.markdown("<div class='title'>❤️ Heart Stroke Prediction</div>", unsafe_allow_html=True)
st.markdown("Provide the following details to check your heart stroke risk:")

# ================== Sidebar Branding ==================
st.sidebar.title("🔴 App Info")
st.sidebar.markdown("This app predicts **Heart Disease Risk** using a trained Logistic Regression model.")
st.sidebar.markdown("---")
st.sidebar.markdown("👨‍💻 Made with ❤️ by **Sohil**")

# ================== Collect User Input ==================
age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["Male", "Female"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# ================== Prediction ==================
if st.button("Predict"):
    # Raw input dict
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    # Input dataframe
    input_df = pd.DataFrame([raw_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]

    # Scale input
    scaled_input = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled_input)[0]

    # Show styled result
    if prediction == 1:
        st.markdown("<div class='result-box high-risk'>⚠️ High Risk of Heart Disease</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='result-box low-risk'>✅ Low Risk of Heart Disease</div>", unsafe_allow_html=True)

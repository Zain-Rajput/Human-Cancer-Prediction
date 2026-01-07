import streamlit as st
import pandas as pd
import joblib

# Load saved model, scaler, and expected columns
model = joblib.load("LOGISTIC_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

st.title("Heart Stroke Prediction by Zain")
st.markdown("Provide the following details to check your heart stroke risk:")

# ----------------------- Page Config -----------------------
st.set_page_config(page_title="❤️ Heart Stroke Prediction", page_icon="💓", layout="centered")

# ----------------------- Custom CSS Styling -----------------------
st.markdown("""
    <style>
    body {
        background-color: #f7f8fa;
        font-family: 'Poppins', sans-serif;
    }
    .main-title {
        text-align: center;
        color: #e63946;
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .sub-title {
        text-align: center;
        color: #555;
        font-size: 18px;
        margin-bottom: 30px;
    }
    .stButton button {
        background-color: #e63946;
        color: white;
        border-radius: 10px;
        padding: 0.6em 2em;
        font-size: 18px;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton button:hover {
        background-color: #d62828;
        transform: scale(1.05);
    }
    .css-1d391kg {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 30px;
        box-shadow: 0px 5px 15px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------- App Title -----------------------
st.markdown("<h1 class='main-title'>❤️ Heart Stroke Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Check your heart stroke risk instantly with our trained AI model</p>", unsafe_allow_html=True)

# ----------------------- Input Section -----------------------
st.write("### 🧠 Enter Your Health Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)

with col2:
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# ----------------------- Predict Button -----------------------
if st.button("💖 Predict My Heart Risk"):
    # Prepare input data
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

    input_df = pd.DataFrame([raw_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    # ----------------------- Display Result -----------------------
    st.markdown("<hr>", unsafe_allow_html=True)
    if prediction == 0:
        st.error("🚨 **High Risk of Heart Disease Detected!**\n\nPlease consult your doctor for a detailed checkup.")
        st.markdown("<div style='text-align:center;font-size:120px;'>💔</div>", unsafe_allow_html=True)
    else:
        st.success("✅ **Low Risk of Heart Disease**\n\nKeep up your healthy lifestyle! ❤️")
        st.markdown("<div style='text-align:center;font-size:120px;'>💪</div>", unsafe_allow_html=True)

# ----------------------- Footer -----------------------
st.markdown("<br><p style='text-align:center;color:#888;'>Developed by <b>Zain</b> 💻 | Powered by Machine Learning</p>", unsafe_allow_html=True)

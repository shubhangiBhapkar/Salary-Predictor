import streamlit as st
import joblib as jb
import numpy as np

# Load Model
salary = jb.load(r"C:\Users\Admin\OneDrive\Desktop\Salary-Predictor\Sal_predictor.pkl")

# Custom Styling
st.markdown(
    """
    <style>
        .title {
            font-size: 40px;
            color: #3498db;
            text-align: center;
            font-weight: bold;
        }
        .stButton>button {
            background-color: #2ecc71;
            color: white;
            font-size: 18px;
            padding: 10px 24px;
            border-radius: 10px;
        }
        .stButton>button:hover {
            background-color: #27ae60;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown('<div class="title">Welcome to Salary Predictor Model</div>', unsafe_allow_html=True)
st.write("\n")

# Input Section with Icon
exp = st.number_input("\U0001F4BC Experience (0-45 years)", min_value=0.0, max_value=45.0)
st.write("\n")

# Predict Button and Result
if st.button("✨ Predict Salary"):
    new_input = np.array([[exp]])
    prediction = salary.predict(new_input)
    predicted_salary = float(prediction[0])
    st.success(f"\U0001F4B0 The expected salary with {exp} years of experience is ₹{predicted_salary:,.2f}")
    st.balloons()

# Footer
st.write("\n")
st.info("Thank you for using the Salary Predictor Model. Good luck with your career!")

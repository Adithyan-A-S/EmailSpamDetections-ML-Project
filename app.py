import streamlit as st
import joblib
import numpy as np
import base64

# Load trained model (Pipeline: TF-IDF + Logistic Regression)
model = joblib.load("spam_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="📧",
    layout="centered"
)

# ---------- BACKGROUND IMAGE ----------
def add_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        /* Full screen background */
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* STRONG dark overlay */
        .stApp::before {{
            content: "";
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, 0.75);
            z-index: -1;
        }}

        /* Center content container */
        .block-container {{
            background: rgba(0, 0, 0, 0.65);
            padding: 40px 45px;
            border-radius: 18px;
            max-width: 620px;
            margin: 10px auto;
            box-shadow: 0 0 40px rgba(0,0,0,0.6);
        }}

        /* Title */
        h1 {{
            color: #ffffff !important;
            font-size: 42px !important;
            text-align: center;
            margin-bottom: 10px;
        }}

        /* Text & labels */
        p, label, span {{
            color: #f9fafb !important;
            font-size: 18px !important;
        }}

        /* Inputs */
        textarea, input {{
            background-color: rgba(15, 23, 42, 0.9) !important;
            color: #ffffff !important;
            font-size: 18px !important;
            border-radius: 10px !important;
        }}

        /* Radio buttons */
        div[data-baseweb="radio"] {{
            color: #ffffff !important;
            font-size: 18px !important;
        }}

        /* Button */
        button {{
            font-size: 18px !important;
            padding: 10px 22px !important;
            border-radius: 10px !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# Add background image
add_bg(r"D:\EmailSpamDetection\newbg.jpg")

# Title
st.title("📧 Email Spam Detection System")
st.markdown(
    "<p style='text-align:center; font-size:20px;'>This system classifies an email text or link as <b>Spam</b> or <b>Not Spam</b>.</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# Choose input type
input_type = st.radio(
    "Select input type:",
    ("Email Text", "Email URL / Link")
)

# Input field
if input_type == "Email Text":
    user_input = st.text_area(
        "✉️ Enter Email Content",
        height=180,
        placeholder="Paste the email message here..."
    )
else:
    user_input = st.text_input(
        "🔗 Paste Email URL / Link",
        placeholder="https://example.com/verify-account"
    )

# Prediction button
if st.button("🔍 Check Spam"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter email text or a URL.")
    else:
        probabilities = model.predict_proba([user_input])[0]
        prediction = np.argmax(probabilities)

        if prediction == 1:
            st.error(
                f"🚫 **SPAM DETECTED**\n\n"
                f"Confidence: **{probabilities[1] * 100:.2f}%**"
            )
        else:
            st.success(
                f"✅ **NOT SPAM**\n\n"
                f"Confidence: **{probabilities[0] * 100:.2f}%**"
            )
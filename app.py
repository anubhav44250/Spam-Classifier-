import streamlit as st
import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


st.set_page_config(page_title="Spam Classifier", layout="centered")


st.markdown("""
<style>
body {
    background-color: #0E1117;
    color: white;
}

h1 {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #9CA3AF;
    margin-bottom: 30px;
}

.stTextArea textarea {
    background-color: #1C1F26;
    color: white;
    border-radius: 12px;
    padding: 12px;
    font-size: 16px;
}

.stButton>button {
    background: linear-gradient(90deg, #2563EB, #3B82F6);
    color: white;
    border-radius: 10px;
    height: 45px;
    width: 100%;
    font-weight: bold;
    border: none;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #1D4ED8, #2563EB);
}

.result-box {
    background-color: #1C1F26;
    padding: 20px;
    border-radius: 14px;
    margin-top: 25px;
    box-shadow: 0px 0px 15px rgba(0,0,0,0.4);
}
</style>
""", unsafe_allow_html=True)


st.markdown("<h1>🛡️ Spam Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI Message Security Analyzer</p>", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    df = pd.read_csv("sms.tsv", sep="\t", header=None)
    df.columns = ["label", "message"]
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    def clean_text(text):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z]", " ", text)
        return text

    df["message"] = df["message"].apply(clean_text)

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["message"], df["label"], test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    return model, vectorizer

model, vectorizer = load_model()


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    return text


suspicious_words = ["win", "free", "prize", "money", "offer", "gift", "reward"]
urgency_words = ["now", "urgent", "immediately", "hurry", "instantly"]
sensitive_words = ["otp", "password", "bank", "account", "withdraw", "login"]
financial_words = ["pay", "invoice", "due", "billing", "transaction", "penalty"]

def detect_risk(message):
    msg = message.lower()
    reasons = []

    if any(word in msg for word in suspicious_words):
        reasons.append("Contains suspicious words")

    if any(word in msg for word in urgency_words):
        reasons.append("Uses urgency words")

    if any(word in msg for word in sensitive_words):
        reasons.append("Requests sensitive info")

    if any(word in msg for word in financial_words):
        reasons.append("Financial request detected")

    if "http" in msg or "www" in msg:
        reasons.append("Contains link")

    return reasons

def override_spam(pred, reasons):
    if pred == 1:
        return pred

    strong_signals = [
        "Requests sensitive info",
        "Contains link",
        "Financial request detected"
    ]

    if any(r in reasons for r in strong_signals) and len(reasons) >= 2:
        return 1

    if len(reasons) >= 3:
        return 1

    return pred

def risk_level(reasons, pred):
    if len(reasons) == 0:
        return "Safe"
    elif len(reasons) <= 2:
        return "Suspicious"
    else:
        return "Dangerous"


message = st.text_area("Enter your message", height=150)
analyze = st.button("Analyze")


if analyze and message:
    msg_clean = [clean_text(message)]
    msg_vec = vectorizer.transform(msg_clean)

    pred = model.predict(msg_vec)[0]
    prob = model.predict_proba(msg_vec)[0][1]

    reasons = detect_risk(message)
    pred = override_spam(pred, reasons)
    risk = risk_level(reasons, pred)

    st.markdown("<div class='result-box'>", unsafe_allow_html=True)

    if pred == 1:
        st.markdown(f"<h3 style='color:#EF4444;'>🚨 Spam: YES ({prob*100:.2f}%)</h3>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h3 style='color:#22C55E;'>✅ Spam: NO ({(1-prob)*100:.2f}%)</h3>", unsafe_allow_html=True)

    st.markdown(f"<p><b>Risk Level:</b> {risk}</p>", unsafe_allow_html=True)

    if reasons:
        st.markdown("<p><b>Reasons:</b></p>", unsafe_allow_html=True)
        for r in reasons:
            st.markdown(f"<p>• {r}</p>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
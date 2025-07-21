# app.py

import streamlit as st
from classifier import classify_ticket

st.set_page_config(page_title="Support Ticket Classifier", page_icon="📬")
st.title("📬 Zero-Shot Support Ticket Classifier")
st.markdown("Enter a support ticket message and get a predicted category.")

ticket = st.text_area("📝 Enter support message:", height=150)

if st.button("Classify"):
    if ticket.strip():
        category, confidence = classify_ticket(ticket)
        st.success(f"✅ **Predicted Category**: {category}")
        st.info(f"🔍 **Confidence Score**: {confidence:.2f}")
    else:
        st.warning("Please enter a message first.")

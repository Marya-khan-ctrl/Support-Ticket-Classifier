# app.py

import streamlit as st
from classifier_few_shot import classify_ticket_few_shot

st.set_page_config(page_title="Support Ticket Few-Shot Classifier", page_icon="🎯")

st.title("🎯 Few-Shot Support Ticket Classifier")
st.markdown("Enter a support message. The model uses examples to predict the category.")

ticket = st.text_area("📝 Enter support message:", height=150)

if st.button("Classify"):
    if ticket.strip():
        category = classify_ticket_few_shot(ticket)
        st.success(f"✅ **Predicted Category**: {category}")
    else:
        st.warning("Please enter a message first.")

# classifier.py

from transformers import pipeline

# Load zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define support categories
candidate_labels = ["Billing", "Technical Support", "Account Management", "Product Inquiry"]

# Function used by Streamlit app
def classify_ticket(ticket):
    result = classifier(ticket, candidate_labels)
    category = result["labels"][0]
    confidence = result["scores"][0]
    return category, confidence

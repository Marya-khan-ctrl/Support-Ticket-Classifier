# classifier_few_shot.py

from transformers import pipeline

# Load a language model for few-shot-style prompting
generator = pipeline("text-generation", model="gpt2")

# Few-shot examples to include in prompt
examples = """
Classify the support ticket into one of these categories: Billing, Technical Support, Account Management, Product Inquiry.

Ticket: I can't update my payment method.
Category: Billing

Ticket: The app crashes when I try to open it.
Category: Technical Support

Ticket: How can I change my subscription plan?
Category: Account Management

Ticket: I want to know more about the features before buying.
Category: Product Inquiry
"""

def classify_ticket_few_shot(ticket):
    # Combine prompt with new input
    prompt = examples + f"\nTicket: {ticket}\nCategory:"
    
    # Generate output, allowing 20 new tokens after prompt
    result = generator(prompt, num_return_sequences=1, max_new_tokens=20)[0]['generated_text']
    
    # Extract predicted category (after the last 'Category:')
    predicted = result.split("Category:")[-1].split("\n")[0].strip()
    return predicted

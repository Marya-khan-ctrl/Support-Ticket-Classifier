# Support-Ticket-Classifier
This project classifies support tickets into predefined categories using transformer models, including zero-shot (BART), few-shot (GPT-2), and a fine-tuned BERT trained on Google Colab GPU.


Objective
The aim of this project is to automatically classify customer support tickets into specific categories: Billing, Technical Support, Account Management, and Product Inquiry. Automating this process helps reduce response time and improve customer support operations.

Methodology / Approach
The project implements and compares three different classification strategies using transformer models:

Zero-Shot Classification:
Using facebook/bart-large-mnli, this method classifies unseen ticket data without requiring any training. It directly compares ticket text with candidate labels using a pre-trained natural language inference model.

Few-Shot Prompting:
Using gpt2, this approach constructs a prompt with a few labeled examples. The model continues the pattern to predict the category of a new ticket. This is helpful when only a few examples are available and training is not feasible.

Fine-Tuned BERT Classifier:
A custom classifier was trained using a BERT model (bert-base-uncased) fine-tuned on labeled support ticket data. The training was done on Google Colab with GPU acceleration to speed up the process. After training, the model achieved good accuracy and F1 score and was saved using model.save_pretrained() for local deployment.

Key Results / Observations
The zero-shot classifier is flexible and performs well out-of-the-box but may struggle with domain-specific variations.

The few-shot approach works well when the examples are carefully selected but is less stable and depends heavily on prompt design.

The fine-tuned BERT model delivered the best performance after being trained on a labeled dataset, achieving high accuracy (e.g., >84%) and strong F1 score.

Using a GPU-enabled environment (Google Colab) significantly reduced training time and improved experimentation speed.

Python 3.12.7 | packaged by Anaconda, Inc. | (main, Oct  4 2024, 13:17:27) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> # Combined Python Code from Google Colab Notebook
... # Original Notebook: https://colab.research.google.com/drive/1Ejti-RtJyNaE6Wbz_oHURWXvF7PKyk16?usp=sharing
... 
... # --- Original Colab Block 1: Install Libraries ---
... !pip install transformers datasets accelerate evaluate torch pandas scikit-learn matplotlib seaborn -q
... 
... # --- Original Colab Block 2: Import Libraries ---
... import torch
... import pandas as pd
... import numpy as np
... import matplotlib.pyplot as plt
... import seaborn as sns
... from datasets import load_dataset
... from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
... from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
... import evaluate
... 
... # Set device for PyTorch
... device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
... print(f"Using device: {device}")
... 
... # --- Original Colab Block 3: Load Dataset ---
... # Load the IMDB dataset
... dataset = load_dataset("imdb")
... 
... # --- Original Colab Block 4: Explore Dataset ---
... print("Dataset structure:")
... print(dataset)
... 
... print("\nTraining set info:")
... print(dataset["train"])
... 
... print("\nTest set info:")
print(dataset["test"])

print("\nFirst 5 examples from training set:")
for i in range(5):
    print(f"Text: {dataset['train'][i]['text']}")
    print(f"Label: {dataset['train'][i]['label']}")
    print("-" * 30)

# Check label distribution
train_labels = [example['label'] for example in dataset['train']]
test_labels = [example['label'] for example in dataset['test']]

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.countplot(x=train_labels)
plt.title('Training Set Label Distribution')
plt.xlabel('Label')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
sns.countplot(x=test_labels)
plt.title('Test Set Label Distribution')
plt.xlabel('Label')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# --- Original Colab Block 5: Tokenization ---
# Load pre-trained tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)

# Apply tokenization to the entire dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Remove the original 'text' column as it's no longer needed after tokenization
tokenized_dataset = tokenized_dataset.remove_columns(["text"])

# Rename 'label' column to 'labels' as expected by the Trainer API
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

# Set the format to PyTorch tensors
tokenized_dataset.set_format("torch")

print("\nTokenized dataset structure:")
print(tokenized_dataset)
print("\nFirst tokenized example from training set:")
print(tokenized_dataset["train"][0])

# --- Original Colab Block 6: Data Collator ---
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# --- Original Colab Block 7: Load Pre-trained Model ---
# Load pre-trained model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
print("\nModel loaded successfully.")

# --- Original Colab Block 8: Training Arguments ---
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch", # Evaluate at the end of each epoch
    save_strategy="epoch",       # Save model at the end of each epoch
    load_best_model_at_end=True, # Load the best model at the end of training
    metric_for_best_model="f1",  # Metric to use for early stopping/best model selection
    report_to="none" # Disable reporting to external services like W&B
)

# --- Original Colab Block 9: Evaluation Metrics ---
# Load the evaluate metric
metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, predictions, average="weighted")
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average="weighted")
    recall = recall_score(labels, predictions, average="weighted")
    return {"f1": f1, "accuracy": accuracy, "precision": precision, "recall": recall}

# --- Original Colab Block 10: Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# --- Original Colab Block 11: Train Model ---
print("\nStarting model training...")
trainer.train()
print("Model training complete.")

# --- Original Colab Block 12: Evaluate Model ---
print("\nEvaluating model on the test set...")
eval_results = trainer.evaluate()
print("Evaluation Results:")
for key, value in eval_results.items():
    print(f"{key}: {value:.4f}")

# --- Original Colab Block 13: Inference ---
# Example inference
text_to_classify_positive = "This movie was absolutely fantastic! I loved every minute of it."
text_to_classify_negative = "Terrible film, completely boring and a waste of time."

# Tokenize the input text
inputs_positive = tokenizer(text_to_classify_positive, return_tensors="pt", truncation=True).to(device)
inputs_negative = tokenizer(text_to_classify_negative, return_tensors="pt", truncation=True).to(device)

# Perform inference
with torch.no_grad():
    outputs_positive = model(**inputs_positive)
    outputs_negative = model(**inputs_negative)

# Get predicted class (0 or 1)
predictions_positive = torch.argmax(outputs_positive.logits, dim=-1).item()
predictions_negative = torch.argmax(outputs_negative.logits, dim=-1).item()

# Map label to sentiment
id_to_label = {0: "Negative", 1: "Positive"}

print(f"\nText: '{text_to_classify_positive}'")
print(f"Predicted Sentiment: {id_to_label[predictions_positive]}")

print(f"\nText: '{text_to_classify_negative}'")
print(f"Predicted Sentiment: {id_to_label[predictions_negative]}")

# --- Original Colab Block 14: Save Model ---
# Define paths to save the model and tokenizer
output_model_path = "./fine_tuned_sentiment_model"
output_tokenizer_path = "./fine_tuned_sentiment_tokenizer"

# Save the fine-tuned model
trainer.save_model(output_model_path)
# Save the tokenizer
tokenizer.save_pretrained(output_tokenizer_path)

print(f"\nModel saved to: {output_model_path}")
print(f"Tokenizer saved to: {output_tokenizer_path}")

# Optional: Load the saved model and tokenizer to verify
# from transformers import pipeline
# loaded_tokenizer = AutoTokenizer.from_pretrained(output_tokenizer_path)
# loaded_model = AutoModelForSequenceClassification.from_pretrained(output_model_path)
#
# # Create a pipeline for easy inference
# classifier = pipeline("sentiment-analysis", model=loaded_model, tokenizer=loaded_tokenizer, device=0 if torch.cuda.is_available() else -1)
#
# print("\nTesting loaded model pipeline:")
# print(classifier("This is an amazing product!"))
# print(classifier("I hated this experience."))

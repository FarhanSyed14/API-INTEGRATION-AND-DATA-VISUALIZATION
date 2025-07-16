Fine-tuning a Pre-trained Model for Text Classification
This repository contains a Google Colab notebook that demonstrates how to fine-tune a pre-trained transformer model (DistilBERT) for a text classification task using the Hugging Face transformers library. The notebook uses the imdb dataset for sentiment analysis (classifying movie reviews as positive or negative).

Table of Contents
Project Overview

Features

Setup and Running in Google Colab

Dataset

Model

Evaluation

Dependencies

Combined Code

Project Overview
The goal of this project is to provide a clear and executable example of fine-tuning a transformer model for text classification. It covers the entire machine learning pipeline from data loading and preprocessing to model training, evaluation, and inference.

Features
Environment Setup: Installs all necessary libraries for Hugging Face transformers, datasets, accelerate, and evaluate.

Dataset Loading & Exploration: Demonstrates how to load a dataset from the Hugging Face datasets hub and perform basic data exploration.

Tokenization: Shows how to use a pre-trained tokenizer to convert text into numerical tokens suitable for transformer models.

Data Preparation: Utilizes DataCollatorWithPadding for efficient batching of variable-length sequences.

Model Loading: Loads a pre-trained AutoModelForSequenceClassification for the classification task.

Trainer API: Leverages the Hugging Face Trainer API for streamlined model training and evaluation.

Custom Metrics: Defines a custom function to compute evaluation metrics (accuracy, F1-score, precision, recall).

Model Training & Evaluation: Executes the fine-tuning process and evaluates the model's performance on the test set.

Inference: Provides an example of how to use the fine-tuned model for predictions on new text.

Model Saving: Shows how to save the trained model and tokenizer for future use.

Setup and Running in Google Colab
The easiest way to run this project is directly in Google Colab:

Open the Notebook: Click on the Colab link: https://colab.research.google.com/drive/1Ejti-RtJyNaE6Wbz_oHURWXvF7PKyk16?usp=sharing

Run All Cells: Go to Runtime -> Run all in the Colab menu.

GPU Acceleration: Ensure you have a GPU runtime enabled for faster training. Go to Runtime -> Change runtime type and select GPU as the hardware accelerator.

Dataset
Name: imdb

Description: A large movie review dataset for binary sentiment classification.

Classes: 0 (negative) and 1 (positive).

Splits: train and test.

Model
Base Model: distilbert-base-uncased

Task: Sequence Classification (sentiment analysis)

Evaluation
The model's performance is evaluated using the following metrics:

Accuracy: The proportion of correctly classified samples.

F1-score: The harmonic mean of precision and recall, useful for imbalanced datasets.

Precision: The proportion of true positive predictions among all positive predictions.

Recall: The proportion of true positive predictions among all actual positive samples.

Dependencies
The notebook automatically installs the required libraries:

transformers

datasets

accelerate

evaluate

torch

pandas

scikit-learn

matplotlib

seaborn

Combined Code
For convenience, all the code from the Google Colab notebook has been combined into a single Python script, which can be found in the combined-code section below. This script can be run in any Python environment with the necessary dependencies installed.
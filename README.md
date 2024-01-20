# Science LLM Chat Bot

## Introduction

The Science Chat Bot is a conversational AI project that uses the GPT-2 language model to answer science-related questions. This repository contains two main scripts - `train.py` for fine-tuning the GPT-2 model on a science-related dataset and `app.py` for deploying the chatbot in a Streamlit web application.

## Project Components

1. Training Script (`train.py`):
   - Purpose: Fine-tunes the GPT-2 model on a science-related dataset.
   - Dependencies: pandas, torch, transformers.
   - Usage:
     ```bash
     python train.py
     ```
   - Output: Trained model and tokenizer saved in a file named `your_model.pkl`.

2. Streamlit App (`app.py`):
   - Purpose: Deploys the trained GPT-2 model in a user-friendly web application.
   - Dependencies: streamlit, torch, transformers.
   - Usage:
     ```bash
     streamlit run app.py
     ```
   - Access: Open the provided link in a web browser to interact with the Science Chat Bot.



## Sample Usage

1. Run `train.py` to fine-tune the GPT-2 model.
2. Run `app.py` to deploy the Streamlit app.
3. Ask science-related questions in the app and receive generated responses.

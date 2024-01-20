import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import pickle

def main():
    st.title("Science Chat Bot")
    st.markdown('Ask Any Science Related Questions..')
    st.write("Made ❤️ with Rosary Abilash")
    with open("your_model.pkl", "rb") as f:
        model, tokenizer = pickle.load(f)
        user_input = st.text_input("Ask a question:")
    
    if user_input:
        # Tokenize the user input
        input_ids = tokenizer.encode(user_input, return_tensors="pt")

        # Generate a response
        with torch.no_grad():
            output = model.generate(input_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2)

        response = tokenizer.decode(output[0], skip_special_tokens=True)

        # Display the generated response
        st.write("Chatbot Response:", response)

if __name__ == "__main__":
    main()


# import streamlit as st
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# import torch

# def main():
#     st.title("GPT-2 Chatbot")

#     # Replace "path/to/your/saved_model" with the actual path to your saved model
#     model_path = "path/to/your/saved_model"
#     model = GPT2LMHeadModel.from_pretrained(model_path)
#     tokenizer = GPT2Tokenizer.from_pretrained(model_path)

#     user_input = st.text_input("Ask a question:")
    
#     if user_input:
#         # Tokenize the user input
#         input_ids = tokenizer.encode(user_input, return_tensors="pt")

#         # Generate a response
#         with torch.no_grad():
#             output = model.generate(input_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2)

#         response = tokenizer.decode(output[0], skip_special_tokens=True)

#         # Display the generated response
#         st.text_area("Chatbot Response:", response)

# if __name__ == "__main__":
#     main()


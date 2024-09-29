import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

# Assuming your tokenizer is also saved, or recreate it here if necessary
# Load the pre-trained model
model = load_model('my_model.h5')

with open('training_set.txt', 'r', encoding='utf-8') as file:
    training_text = file.read()

tokenizer = Tokenizer()

# Fit the Tokenizer on the training data
tokenizer.fit_on_texts([training_text])

def generate_text(seed_text, next_words, model, max_sequence_len, tokenizer):
    for _ in range(next_words):
        # Tokenize the current seed text
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        # Pad the sequence
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        # Predict the next word token
        predicted = model.predict(token_list, verbose=0)
        predicted_index = np.argmax(predicted, axis=-1)[0]
        # Convert token to word
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        # Append to the seed text
        seed_text += " " + output_word
    return seed_text

# Streamlit interface setup
st.title('Text Generation App')
input_sentence = st.text_input("Enter a seed sentence:")
num_words = st.number_input("Enter the number of words to generate:", min_value=1, value=5)
max_sequence_len = 50  # Adjust this based on the sequence length used during training

if st.button('Generate Text'):
    if input_sentence:
        generated_text = generate_text(input_sentence, num_words, model, max_sequence_len, tokenizer)
        st.write("Generated Text:", generated_text)
    else:
        st.warning("Please enter a seed sentence.")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
import numpy as np
import json
import os

# Load IMDB dataset with raw text
(x_train_raw, y_train), (x_test_raw, y_test) = keras.datasets.imdb.load_data(num_words=None)  # Load full dataset
word_index = keras.datasets.imdb.get_word_index()  # Get word index mapping

# Reverse word index to convert integer sequences back to text
index_to_word = {index + 3: word for word, index in word_index.items()}
index_to_word[0], index_to_word[1], index_to_word[2] = "<PAD>", "<START>", "<UNK>"

# Convert sequences back to text
def decode_review(sequence):
    return " ".join([index_to_word.get(i, "<UNK>") for i in sequence])

# Convert training & test data to raw text
x_train_text = [decode_review(seq) for seq in x_train_raw]
x_test_text = [decode_review(seq) for seq in x_test_raw]

# Tokenizer setup (explicitly define vocabulary)
num_words = 10000  # Limit vocabulary size
max_length = 200   # Maximum sequence length

tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
tokenizer.fit_on_texts(x_train_text)  # Train tokenizer on training data

# Convert text to sequences using Tokenizer
x_train = tokenizer.texts_to_sequences(x_train_text)
x_test = tokenizer.texts_to_sequences(x_test_text)

# Pad sequences to ensure uniform input size
x_train = pad_sequences(x_train, maxlen=max_length, padding='post', truncating='post')
x_test = pad_sequences(x_test, maxlen=max_length, padding='post', truncating='post')

print(f"Training data shape: {x_train.shape}")  # Expect (25000, 200)
print(f"Testing data shape: {x_test.shape}")    # Expect (25000, 200)

model = keras.Sequential([
    layers.Embedding(input_dim=num_words, output_dim=32, input_length=max_length),
    layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Dense(16, activation="tanh"),
    layers.Dense(1, activation="sigmoid")
])

# Compile the model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=5,  # Increase epochs for better training
    validation_data=(x_test, y_test)
)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Save the trained model
model.save("imdb_sentiment_model.h5")

# Save the tokenizer as JSON
tokenizer_json = tokenizer.to_json()
with open("tokenizer.json", "w") as f:
    f.write(tokenizer_json)

print("Model and Tokenizer saved successfully.")



"""

This file is responsible for training the sentiment analysis model.

We get movie reviews 📜
    TensorFlow provides a dataset (IMDB), where reviews are already converted into numbers.
    We take only the 10,000 most common words for our vocabulary.

We make sure all reviews are the same length 📏
    Some reviews are long, some are short.
    We make them all 200 words by adding padding or cutting off extra words.

We build a deep learning model 🧠
    First, we use an Embedding layer to convert words into meaningful number patterns.
    Then, we use two Bidirectional LSTM layers (a special type of memory layer) to learn patterns in the reviews.
    Finally, we add a Dense (fully connected) layer to make a decision:
        Positive review? (1)
        Negative review? (0)

We train the model 🏋️
    The model reads all training reviews multiple times (called epochs) and improves itself.

We test the model ✅
    After training, we check how well it performs on unseen reviews.

We save the model for later use 💾
    So that we don’t have to train it again!

"""


# ----------------------------------------------
# Output
# ----------------------------------------------

"""

Epoch 1/5
196/196 ━━━━━━━━━━━━━━━━━━━━ 181s 872ms/step - accuracy: 0.6570 - loss: 0.5882 - val_accuracy: 0.8370 - val_loss: 0.3778
Epoch 2/5
196/196 ━━━━━━━━━━━━━━━━━━━━ 165s 845ms/step - accuracy: 0.8923 - loss: 0.2808 - val_accuracy: 0.7976 - val_loss: 0.4384
Epoch 3/5
196/196 ━━━━━━━━━━━━━━━━━━━━ 207s 1s/step - accuracy: 0.9232 - loss: 0.2142 - val_accuracy: 0.8382 - val_loss: 0.3808
Epoch 4/5
196/196 ━━━━━━━━━━━━━━━━━━━━ 211s 1s/step - accuracy: 0.9436 - loss: 0.1609 - val_accuracy: 0.8360 - val_loss: 0.3940
Epoch 5/5
196/196 ━━━━━━━━━━━━━━━━━━━━ 169s 861ms/step - accuracy: 0.9603 - loss: 0.1207 - val_accuracy: 0.8337 - val_loss: 0.5595
782/782 ━━━━━━━━━━━━━━━━━━━━ 77s 98ms/step - accuracy: 0.8339 - loss: 0.5581  

Test Accuracy: 83.37%
Test Loss: 0.5595

Model and Tokenizer saved successfully.

"""
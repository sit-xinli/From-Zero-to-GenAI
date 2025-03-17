import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json

# Load trained model
model = tf.keras.models.load_model("imdb_sentiment_model.h5")
print("Model loaded successfully.")

# Load the tokenizer JSON correctly
with open("tokenizer.json", "r") as f:
    tokenizer_json = f.read()  # Read as string

tokenizer = tokenizer_from_json(tokenizer_json)  # Corrected


# Define preprocessing function
def preprocess_text(text, tokenizer, max_length=200):
    sequence = tokenizer.texts_to_sequences([text])  # Convert text to sequence
    padded = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    return padded

# Sample reviews for testing
new_reviews = [
    "This movie was fantastic! The story was engaging and the characters were well developed.",
    "Absolutely terrible. Waste of time. The acting was so bad and the plot was boring.",
    "It was an okay-okay movie. Nothing special",
    "It was a good movie. Enjoyed it a lot",
    "It was the worst movie I've ever seen. I hated everything about it."
]

# Process and predict sentiment for each review
for review in new_reviews:
    processed_review = preprocess_text(review, tokenizer)
    prediction = model.predict(processed_review)[0][0]
    
    # Classify as Positive or Negative
    sentiment = "Positive" if prediction >= 0.5 else "Negative"
    print(f"\nReview: {review}")
    print(f"Predicted Sentiment: {sentiment} ({prediction:.4f})")



"""

👉 This file is responsible for testing the trained model.

We load the saved model 📂
    No need to train again. We directly use the model we trained in the previous file.

We load the Tokenizer 🔡
The Tokenizer helps convert new text reviews into numbers.

We take new movie reviews (written as text) ✍️
    Example: "This movie was fantastic! The story was engaging and the characters were well developed."

We convert the review into numbers 🔢
    The model only understands numbers, so we use the Tokenizer to convert words into the same number format we used in training.

We make sure the review is 200 words long 📏
    Again, padding is used to match the training format.

We ask the model to predict sentiment 🧐
    The model gives a number between 0 and 1:
        Closer to 1 → Positive review 😊
        Closer to 0 → Negative review 😡

We print the result 📢

"""


# ----------------------------------------------
# Output
# ----------------------------------------------

"""

Model loaded successfully.
1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 1s/step

Review: This movie was fantastic! The story was engaging and the characters were well developed.
Predicted Sentiment: Positive (0.8211)
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 76ms/step

Review: Absolutely terrible. Waste of time. The acting was so bad and the plot was boring.
Predicted Sentiment: Negative (0.0067)
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 122ms/step

Review: It was an okay-okay movie. Nothing special
Predicted Sentiment: Negative (0.2266)
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 108ms/step

Review: It was a good movie. Enjoyed it a lot
Predicted Sentiment: Positive (0.9641)
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 138ms/step

Review: It was the worst movie I've ever seen. I hated everything about it.
Predicted Sentiment: Negative (0.0256)

"""
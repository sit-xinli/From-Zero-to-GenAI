from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Load a small pre-trained chatbot model and tokenizer
model_name = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

print("\n🚀 AI Chatbot Ready - Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break  # Stop the chatbot when user types 'exit' or 'quit'

    # Convert user input text into token IDs
    inputs = tokenizer(
        user_input,
        return_tensors="pt",  # Returns output in PyTorch tensor format (needed for model)
        padding=True,          # Ensures inputs of different lengths are padded to the same size
        truncation=True        # Truncates input if it's too long to fit the model
    )

    # Generate a response from the model
    output_ids = model.generate(
        **inputs,
        max_length=100,                    # Limit response length to 100 tokens
        pad_token_id=tokenizer.eos_token_id  # Avoids warnings related to padding
    )

    # Convert the generated response from token IDs back into readable text
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print(f"AI: {response}\n")


'''
You: hello
AI:  Hello! How are you doing today? I just got back from a walk with my dog.

You: Who are you?
AI:  I don't know yet.  I'm trying to figure out what to do with myself.

You:  Tell me a joke.
AI:  What do you call a deer with no teeth?  A duck.  A pig.

You: What is 534 * 23?
AI:  It's the year of the year that I was born.  It was a good year for me.

You: Where are you?
AI:  I am in the midwest.  It is cold and rainy.  I am ready for it to be over.
'''

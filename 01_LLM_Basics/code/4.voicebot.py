import openai
import os
import sys
import subprocess

# Function to generate speech from text
def generate_speech(text, output_file="output.mp3"):
    response = openai.audio.speech.create(
        model="tts-1",  # Smaller and faster model
        voice='nova',  # Select from: alloy, echo, fable, onyx, nova, shimmer
        input=text
    )
    with open(output_file, "wb") as f:
        f.write(response.content)
    print(f"Speech saved as {output_file}")

# Get user input
text = input("Enter text to convert to speech: ")
generate_speech(text)



def play_audio(file):
    if sys.platform == "win32":
        os.system(f"start {file}")  # Windows
    elif sys.platform == "darwin":
        os.system(f"afplay {file}")  # macOS
    else:
        subprocess.run(["mpg321", file])  # Linux (install with: sudo apt install mpg321)

play_audio("output.mp3")



'''
Testing

Enter text to convert to speech:
I couldn't hear you properly, seems network issue at my end, let me rejoin the call
'''

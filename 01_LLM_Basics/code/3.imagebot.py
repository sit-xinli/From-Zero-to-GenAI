import openai
import requests
from PIL import Image
from io import BytesIO

# Function to generate an image using OpenAI's DALL·E
def generate_image(prompt):
    response = openai.images.generate(
        model="dall-e-2",  # You can use "dall-e-3" for better results
        prompt=prompt,
        n=1,
        size="256x256"  # Small size for quick generation
    )
    image_url = response.data[0].url
    return image_url

# Get user input
prompt = input("Enter an image description: ")
image_url = generate_image(prompt)

# Download and display the image
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))
image.show()


'''
Testing

Enter an image description:
A dog is a marathon runner having nike shoes and about to win
'''

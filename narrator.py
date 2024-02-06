import os
from openai import OpenAI
import base64
import json
import time
import simpleaudio as sa
import errno
from elevenlabs import generate, play, set_api_key, voices

client = OpenAI()

set_api_key(os.environ.get("ELEVENLABS_API_KEY"))

def encode_image(image_path):
    while True:
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except IOError as e:
            if e.errno != errno.EACCES:
                # Not a "file in use" error, re-raise
                raise
            # File is being written to, wait a bit and retry
            time.sleep(0.1)


def play_audio(text):
    audio = generate(text, voice="Szy1HGeSt7N9YMB4blKq")

    unique_id = base64.urlsafe_b64encode(os.urandom(30)).decode("utf-8").rstrip("=")
    dir_path = os.path.join("narration", unique_id)
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, "audio.wav")

    with open(file_path, "wb") as f:
        f.write(audio)

    play(audio)

def analyze_image(base64_image):
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "system",
                "content": """
                You are a Minecraft professional. Analyze the 
                """
            },
            {
                "role": "user",
                "content": f"data:image/jpeg;base64,{base64_image}"
            }
        ],
        max_tokens=100
    )
    response_text = response.choices[0].message.content
    return response_text


def main():
    while True:
        # path to your image
        image_path = os.path.join(os.getcwd(), "./frames/frame.jpg")

        # getting the base64 encoding
        base64_image = encode_image(image_path)

        # analyze posture
        print("👀 David is watching...")
        analysis = analyze_image(base64_image)

        print("🎙️ David says:")
        print(analysis)

        play_audio(analysis)

        # wait for 5 seconds
        time.sleep(5)

if __name__ == "__main__":
    main()


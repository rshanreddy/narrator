import os
import time
import cv2
from PIL import Image
import numpy as np
import base64
import errno
from flask import Flask, jsonify, render_template
from openai import OpenAI
from elevenlabs import generate, set_api_key

app = Flask(__name__)
client = OpenAI()
set_api_key(os.environ.get("ELEVENLABS_API_KEY"))

# Folder for saving frames and audio
frames_folder = "frames"
narration_folder = "narration"

# Ensure directories exist
os.makedirs(frames_folder, exist_ok=True)
os.makedirs(narration_folder, exist_ok=True)

# Function to capture a single image
def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    time.sleep(2)

    ret, frame = cap.read()
    if ret:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        max_size = 250
        ratio = max_size / max(pil_img.size)
        new_size = tuple([int(x * ratio) for x in pil_img.size])
        resized_img = pil_img.resize(new_size, Image.LANCZOS)
        frame = cv2.cvtColor(np.array(resized_img), cv2.COLOR_RGB2BGR)
        path = os.path.join(frames_folder, "frame.jpg")
        cv2.imwrite(path, frame)
        print("📸 Image captured.")
    cap.release()

# Encode image to base64
def encode_image(image_path):
    while True:
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except IOError as e:
            if e.errno != errno.EACCES:
                raise
            time.sleep(0.1)

# Analyze the image
def analyze_image(base64_image):
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "system",
                "content": """
                You are the angry mean and scary Fletcher from the movie Whiplash.
                The image is of a guy sitting in his home office.
                Look at what he's're doing and tell them how he should do it differently to maximize his productivity to the very utmost.
                Don't repeat yourself and keep it to 2 sentences.
                """
            },
            {
                "role": "user",
                "content": f"data:image/jpeg;base64,{base64_image}"
            }
        ],
        max_tokens=50
    )
    return response.choices[0].message.content

# Generate and save audio
def play_audio(text):
    audio = generate(text, voice="FymFzmXuLh2piu8Rs9it")
    unique_id = base64.urlsafe_b64encode(os.urandom(30)).decode("utf-8").rstrip("=")
    dir_path = os.path.join(narration_folder, unique_id)
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, "audio.wav")
    with open(file_path, "wb") as f:
        f.write(audio)
    return file_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process-image', methods=['GET'])
def process_image():
    capture_image()
    image_path = os.path.join(os.getcwd(), frames_folder, "frame.jpg")
    base64_image = encode_image(image_path)
    analysis = analyze_image(base64_image)
    audio_file_path = play_audio(analysis)
    return jsonify(analysis=analysis, audio_file_path=audio_file_path)

if __name__ == '__main__':
    app.run(debug=True)

import base64
from io import BytesIO
from flask import Flask, render_template, redirect, url_for, request
import cv2
import numpy as np
from PIL import Image
from base64 import b64encode
from run import main 
app = Flask(__name__)

class Camera:
    def __init__(self):
        self.video_capture = cv2.VideoCapture(0)

    def capture_image(self):
        success, frame = self.video_capture.read()
        return success, frame

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture')
def capture():
    return render_template('capture.html')

# Initialize processed_image variable
processed_image = None

@app.route('/process_and_show', methods=['POST'])
def process_and_show():
    # Get the image file from the POST request
    image_file = request.files['image']
    
    # Convert the image file to a NumPy array
    nparr = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cv2.imwrite("static/input_image.jpg",image)
    # Call the external image processing function
    processed_image = main(image)

    # Convert the processed image to bytes using PIL
    processed_image_pil = Image.fromarray(processed_image)
    processed_image_bytesio = BytesIO()
    processed_image_pil.save(processed_image_bytesio, format='PNG')
    processed_image_bytes = processed_image_bytesio.getvalue()

    # Encode the processed image using base64
    encoded_image = base64.b64encode(processed_image_bytes).decode('utf-8')
    processed_image = encoded_image
    return render_template('result.html')

@app.route('/change_glasses', methods=['POST'])
def change_glasses():
    # Get the image file from the POST request
    image_file = request.json['glass_name']
    
    # Convert the image file to a NumPy array
    image=cv2.imread("static/input_image.jpg")
    # Call the external image processing function
    processed_image = main(image,image_file)

    

    
    return render_template('result.html')

@app.route('/result')
def result():
    return render_template('result.html')



if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=False)

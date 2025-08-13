from flask import Flask, jsonify, send_file, request
from gradio_client import Client, handle_file
from flask_cors import CORS
import cv2
import io
import base64
from PIL import Image
import json
import numpy as np

def decompress_arr(carr):
    arr = []
    index = 0

    for i in range(len(carr)):
        if type(carr[i]) is int:
            arr.append(carr[i])
            index += 1
        else:
            l = int(carr[i][1:])
            for j in range(l):
                arr.append(0)
            index += l
    return arr

app = Flask(__name__)
CORS(app) 
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 50 MB

SPACE_ID = "shivamkunkolikar/unet-inpaint-space"

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/uploads", methods=['POST'])
def uploads():
    # image = cv2.imread('aircraft.jpeg')
    # _, buffer = cv2.imencode('.png', image)
    # io_buf = io.BytesIO(buffer)


    if 'image' not in request.files:
        return "No image part", 400
    if 'mask' not in request.form:
        return "No mask part", 400

    image_file = request.files['image']
    mask_data = json.loads(request.form['mask'])  

    mask = decompress_arr(mask_data)
    mask = np.array(mask, dtype=np.uint8)
    mask = np.resize(mask, (512, 512))
    mask = Image.fromarray(mask, mode='L')

    image_file.save('image.png')
    mask.save('mask.png')

    client = Client(SPACE_ID)

    image_handle = handle_file('image.png')
    mask_handle  = handle_file('mask.png')

    result = client.predict(
        image=image_handle,
        mask=mask_handle,
        api_name="/predict"
    )

    img = Image.open(result)
    # img.show(title="Inpainted Output")
    img.save('output.png')


    image_path = 'output.png'  # Update this to your image file

    return send_file(image_path, mimetype='image/png')
    
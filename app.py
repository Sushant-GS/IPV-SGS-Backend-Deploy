from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import cv2
import io
import requests
import base64

app = Flask(__name__)
CORS(app) 

HF_API_URL = "https://shivamkunkolikar-unet-inpaint-space.hf.space/"

@app.route('/')
def home():
    return 'Object Removal Backend is Live'

@app.route('/uploads', methods=['POST'])
def inpaint():
    try:
        if 'image' not in request.files or 'mask' not in request.form:
            return jsonify({'error': 'Image file or mask array is missing'}), 400
        
        image_file = request.files['image']
        mask_array_flat = request.form['mask']

        mask_array = np.array(eval(mask_array_flat), dtype=np.uint8)

        img_stream = io.BytesIO(image_file.read())
        image_pil = Image.open(img_stream).convert("RGB")
        image_np = np.array(image_pil)

        if mask_array.ndim == 1:
            mask_array = mask_array.reshape((image_np.shape[0], image_np.shape[1]))

        if mask_array.shape != image_np.shape[:2]:
            mask_array = cv2.resize(mask_array, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)

        _, img_encoded = cv2.imencode(".png", image_np)
        _, mask_encoded = cv2.imencode(".png", mask_array)

        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        mask_base64 = base64.b64encode(mask_encoded).decode('utf-8')

        hf_payload = {
            "data": [
                f"data:image/png;base64,{img_base64}",
                f"data:image/png;base64,{mask_base64}"
            ]
        }

        response = requests.post(HF_API_URL, json=hf_payload)
        if response.status_code != 200:
            return jsonify({'error': 'Failed to get response from HF API'}), 500

        output_data = response.json()
        result_img_base64 = output_data['data'][0].split(",")[1]

        return jsonify({'processed_image_base64': result_img_base64}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, port=8000)

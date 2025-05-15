from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import cv2
import base64
import requests
from gradio_client import Client, handle_file
import json
import tempfile
import os

app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

HF_API_URL = "https://shivamkunkolikar-unet-inpaint-space.hf.space/"  
# HF_API_URL = "shivamkunkolikar/unet-inpaint-space"

@app.route('/')
def home():
    return 'Object Eraser Backend is Live'

@app.route('/uploads', methods=['POST'])
def inpaint():
    print("1) Entering the inpaint function")
    try:
        if 'image' not in request.files or 'mask' not in request.form:
            return jsonify({'error': 'Image file or mask data is missing'}), 400

        image_file = request.files['image']
        img_stream = io.BytesIO(image_file.read())
        try:
            image_pil = Image.open(img_stream).convert("RGB")
        except Exception as e:
            return jsonify({'error': f'Failed to open image: {str(e)}'}), 400
        image_np = np.array(image_pil)
        h, w = image_np.shape[:2]
        print(f"Original image shape: {image_np.shape}")

        # Resize image to 512x512
        image_resized = cv2.resize(image_np, (512, 512), interpolation=cv2.INTER_AREA)
        print(f"Resized image shape: {image_resized.shape}")

        # Get mask
        mask_array_flat = request.form['mask']
        print("Received mask length:", len(mask_array_flat))
        print("Sample mask string (first 100 chars):", mask_array_flat[:100])

        try:
            mask_array = np.array(json.loads(mask_array_flat), dtype=np.uint8)
        except json.JSONDecodeError as e:
            return jsonify({'error': f'Failed to decode mask JSON: {str(e)}'}), 400

        try:
            mask_array = mask_array.reshape(600, 600)  # Reshape to 2D
        except Exception as e:
            return jsonify({'error': f'Failed to reshape mask to 600x600: {str(e)}'}), 400
        print(f"Original mask shape: {mask_array.shape}")

        # Resize mask to 512x512
        mask_resized = cv2.resize(mask_array, (512, 512), interpolation=cv2.INTER_NEAREST)
        mask_resized = np.where(mask_resized > 0, 255, 0).astype(np.uint8)  # Ensure binary
        print(f"Resized mask shape: {mask_resized.shape}")

        # Save to temporary files
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_image:
            Image.fromarray(image_resized).save(temp_image, format='PNG')
            temp_image_path = temp_image.name

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_mask:
            Image.fromarray(mask_resized).save(temp_mask, format='PNG')
            temp_mask_path = temp_mask.name

        print("2) Image and mask processed, sending to Gradio API")

        # Initialize Gradio client
        try:
            client = Client(HF_API_URL)
        except Exception as e:
            try:
                response = requests.get(f"{HF_API_URL}/info")
                print(f"API Response Status: {response.status_code}")
                print("failed to connect to the client\n\n\n")
                # print(f"API Response Content: {response.text}")
            except Exception as req_e:
                print(f"Failed to fetch API info: {str(req_e)}")
            return jsonify({'error': f'Failed to initialize Gradio client: {str(e)}'}), 500

        result = client.predict(
            image=handle_file(temp_image_path),
            mask=handle_file(temp_mask_path),
            api_name="/predict"
        )
        print("3) Received result from Gradio API:", result)

        os.unlink(temp_image_path)
        os.unlink(temp_mask_path)

        try:
            result_pil = Image.open(result).convert("RGB")
        except Exception as e:
            return jsonify({'error': f'Failed to open Gradio output: {str(e)}'}), 500

        # Convert to base64 for frontend
        buffered = io.BytesIO()
        result_pil.save(buffered, format="PNG")
        result_img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        print("4) Sending processed image to frontend")
        return jsonify({'processed_image_base64': result_img_base64}), 200

    except Exception as e:
        print("Unhandled error:", str(e))
        return jsonify({'error': 'Internal Server Error', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000)









# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from PIL import Image
# import numpy as np
# import cv2
# import io
# import requests
# import base64
# from gradio_client import Client 

# app = Flask(__name__)
# CORS(app) 

# HF_API_URL = "https://shivamkunkolikar-unet-inpaint-space.hf.space/"

# @app.route('/')
# def home():
#     return 'Object Removal Backend is Live'

# @app.route('/uploads', methods=['POST'])
# def inpaint():
#     try:
#         if 'image' not in request.files or 'mask' not in request.form:
#             return jsonify({'error': 'Image file or mask array is missing'}), 400
        
#         image_file = request.files['image']
#         mask_array_flat = request.form['mask']

#         mask_array = np.array(eval(mask_array_flat), dtype=np.uint8)

#         img_stream = io.BytesIO(image_file.read())
#         image_pil = Image.open(img_stream).convert("RGB")
#         image_np = np.array(image_pil)

#         if mask_array.ndim == 1:
#             mask_array = mask_array.reshape((image_np.shape[0], image_np.shape[1]))

#         if mask_array.shape != image_np.shape[:2]:
#             mask_array = cv2.resize(mask_array, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)

#         _, img_encoded = cv2.imencode(".png", image_np)
#         _, mask_encoded = cv2.imencode(".png", mask_array)

#         img_base64 = base64.b64encode(img_encoded).decode('utf-8')
#         mask_base64 = base64.b64encode(mask_encoded).decode('utf-8')

#         hf_payload = {
#             "data": [
#                 f"data:image/png;base64,{img_base64}",
#                 f"data:image/png;base64,{mask_base64}"
#             ]
#         }

#         response = requests.post(HF_API_URL, json=hf_payload)
#         if response.status_code != 200:
#             return jsonify({'error': 'Failed to get response from HF API'}), 500

#         output_data = response.json()
#         result_img_base64 = output_data['data'][0].split(",")[1]

#         return jsonify({'processed_image_base64': result_img_base64}), 200

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=False, port=8000)

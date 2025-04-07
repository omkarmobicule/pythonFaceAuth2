from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import base64
import cv2
import io
from PIL import Image
from deepface import DeepFace

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests (Android)

def decode_image(base64_string):
    """Convert base64 string to OpenCV image safely."""
    try:
        base64_string = base64_string.replace("\n", "").replace("\r", "")  # Clean up
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        image = image.convert("RGB")  # Convert to RGB to prevent issues
        return np.array(image)
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

@app.route('/compare_faces', methods=['POST'])
def compare_faces():
    try:
        data = request.json
        img1_base64 = data.get('image1', '')
        img2_base64 = data.get('image2', '')

        # Decode images
        img1 = decode_image(img1_base64)
        img2 = decode_image(img2_base64)

        if img1 is None or img2 is None:
            return jsonify({"error": "Invalid image data"}), 400

        # Convert images to proper format for DeepFace
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

        # Compare faces using DeepFace (FaceNet model)
        result = DeepFace.verify(img1_rgb, img2_rgb, model_name="Facenet", enforce_detection=False)

        return jsonify({
            "match": result["verified"],
            "similarity": f"{(1 - result['distance']) * 100:.2f}%"
        })

    except Exception as e:
        print(f"Error in face comparison: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

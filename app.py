from flask import Flask, request, jsonify
from deepface import DeepFace
import base64
import io
import cv2
import numpy as np
from PIL import Image

app = Flask(__name__)

def decode_image(base64_string):
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = np.array(image)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"[ERROR] Failed to decode image: {e}")
        return None

def extract_face(image):
    try:
        faces = DeepFace.extract_faces(image, detector_backend="opencv", enforce_detection=False)
        if not faces:
            return None
        area = faces[0]["facial_area"]
        x, y, w, h = area["x"], area["y"], area["w"], area["h"]
        face_crop = image[y:y+h, x:x+w]
        return cv2.resize(face_crop, (160, 160))
    except Exception as e:
        print(f"[ERROR] Face extraction failed: {e}")
        return None

import traceback

@app.route('/compare_faces', methods=['POST'])
def compare_faces():
    try:
        data = request.json
        img1_base64 = data.get('image1', '')
        img2_base64 = data.get('image2', '')

        img1 = decode_image(img1_base64)
        img2 = decode_image(img2_base64)

        face1 = extract_face(img1)
        face2 = extract_face(img2)

        if img1 is None or img2 is None:
            return jsonify({"error": "Invalid image data"}), 400

        resultFacenet512 = DeepFace.verify(
            face1, face2, model_name="Facenet512", detector_backend="opencv", enforce_detection=False
        )

        similarity = (1 - resultFacenet512['distance']) * 100

        return jsonify({
            "match": resultFacenet512['verified'],
            "similarity": f"{similarity:.2f}"
        })

    except Exception as e:
        print("[ERROR]", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)

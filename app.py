import os
from flask import Flask, request, jsonify
from deepface import DeepFace
import cv2
import base64
import numpy as np

app = Flask(__name__)

os.environ["DEEPFACE_HOME"] = "."  # Set DeepFace home directory
weights_dir = os.path.join(os.environ["DEEPFACE_HOME"], "weights")
print(f"[DEBUG] DEEPFACE_HOME={os.environ['DEEPFACE_HOME']}")
print(f"[DEBUG] weights_dir={weights_dir}")

# Check and create directory structure
if not os.path.exists(weights_dir):
    print(f"[INIT] Creating weights directory at: {weights_dir}")
    os.makedirs(weights_dir, exist_ok=True)


# Load the model
print("[INIT] Loading FaceNet model...")
try:
    models = {
        "Facenet": DeepFace.build_model("Facenet")
    }
    print("[INIT] FaceNet model loaded successfully.")
except Exception as e:
    print(f"[INIT ERROR] Failed to load FaceNet model: {e}")
    raise

# Decode base64 string to OpenCV image
def decode_image(base64_str):
    try:
        print("[decode_image] Decoding base64 image...")
        img_data = base64.b64decode(base64_str)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        print("[decode_image] Image decoded successfully.")
        return img
    except Exception as e:
        print(f"[decode_image ERROR] {e}")
        return None

# Skip face detection (assume already cropped)
def extract_face(img):
    print("[extract_face] Skipping detection, returning original image.")
    return img

# Health check route
@app.route('/ping', methods=['GET'])
def ping():
    return "pong", 200

# Face comparison route
@app.route('/compare_faces', methods=['POST'])
def compare_faces():
    try:
        print("[compare_faces] Received request.")
        print(f"[compare_faces] Request from: {request.remote_addr}")

        data = request.json
        img1_base64 = data.get('image1', '')
        img2_base64 = data.get('image2', '')

        print(f"[compare_faces] Length of image1 base64: {len(img1_base64)}")
        print(f"[compare_faces] Length of image2 base64: {len(img2_base64)}")

        img1 = decode_image(img1_base64)
        img2 = decode_image(img2_base64)

        if img1 is None or img2 is None:
            print("[compare_faces] One or both images are invalid.")
            return jsonify({"error": "Invalid image data"}), 400

        print("[compare_faces] Running DeepFace.verify...")

        resultFacenet = DeepFace.verify(
            img1, img2,
            model_name="Facenet", model=models["Facenet"],
            detector_backend="skip", enforce_detection=False
        )

        similarity = (1 - resultFacenet['distance']) * 100
        print(f"[compare_faces] Verification result: Match = {resultFacenet['verified']}, Similarity = {similarity:.2f}%")

        response = {
            "match": resultFacenet['verified'],
            "similarity": f"{similarity:.2f}"
        }

        print("[compare_faces] Returning response to client.")
        return jsonify(response)

    except Exception as e:
        print(f"[compare_faces ERROR] {e}")
        return jsonify({
            "error": "Internal Server Error",
            "details": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

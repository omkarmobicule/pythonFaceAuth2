from flask import Flask, request, jsonify
from deepface import DeepFace
import cv2
import base64
import numpy as np

app = Flask(__name__)

# Load model once and log
print("[INIT] Loading FaceNet model...")
models = {
    "Facenet": DeepFace.build_model("Facenet")
}
print("[INIT] FaceNet model loaded successfully.")

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

def extract_face(img):
    print("[extract_face] Skipping detection, returning original image.")
    return img  # assuming input is already a cropped face

@app.route('/compare_faces', methods=['POST'])
def compare_faces():
    try:
        print("[compare_faces] Received request.")

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

        return jsonify({
            "match": resultFacenet['verified'],
            "similarity": f"{similarity:.2f}"
        })

    except Exception as e:
        print(f"[compare_faces ERROR] {e}")
        return jsonify({"error": str(e)}), 500

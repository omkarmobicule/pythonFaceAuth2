from flask import Flask, request, jsonify
from deepface import DeepFace
import base64, io, cv2, numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Force DeepFace to use a fixed cache directory (if possible, outside ephemeral paths)
os.environ["DEEPFACE_HOME"] = "./.deepface"  # Local folder to store weights
weights_dir = os.path.join(os.environ["DEEPFACE_HOME"], "weights")
os.makedirs(weights_dir, exist_ok=True)
# Load models only once
print("Preloading models...")
models = {
    "Facenet512": DeepFace.build_model("Facenet512")
}
print("Models loaded.")

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
        faces = DeepFace.extract_faces(image, detector_backend="mtcnn", enforce_detection=False)
        if len(faces) == 0:
            return None
        face_data = faces[0]
        facial_area = face_data["facial_area"]
        x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
        face_crop = image[y:y+h, x:x+w]
        return cv2.resize(face_crop, (160, 160))
    except Exception as e:
        print(f"[ERROR] Face extraction failed: {e}")
        return None

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


        resultFacenet = DeepFace.verify(
            face1, face2, model_name="Facenet", model=models["Facenet"],
            detector_backend="skip", enforce_detection=False
        )


        similarity = (1 - resultFacenet['distance']) * 100
        return jsonify({
            "match": resultFacenet['verified'],
            "similarity": f"{similarity:.2f}"
        })

    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)


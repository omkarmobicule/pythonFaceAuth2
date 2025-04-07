from flask import Flask, request, jsonify
from deepface import DeepFace
import base64
import io
import cv2
import numpy as np
from PIL import Image

app = Flask(__name__)

def decode_image(base64_string):
    """Convert base64 string to a NumPy image array."""
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = np.array(image)  # Convert to NumPy array
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to OpenCV BGR format
    except Exception as e:
        print(f"[ERROR] Failed to decode image: {e}")
        return None


def extract_and_show_face(image_path):
    """Extract face from an image and display it."""
    image = cv2.imread(image_path)

    if image is None:
        print(f"[ERROR] Image not loaded. Check the file path: {image_path}")
        return

    try:
        # Detect faces
        faces = DeepFace.extract_faces(image, detector_backend="mtcnn", enforce_detection=False)

        if not faces:
            print("[ERROR] No face detected.")
            return

        for face_data in faces:
            # Extract bounding box
            facial_area = face_data["facial_area"]
            x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]

            # Draw bounding box on image
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Show the detected face with a bounding box
        cv2.imshow("Detected Face", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"[ERROR] Face extraction failed: {e}")

def extract_face(image):
    """Extract and return only the detected face from an image."""
    try:
        faces = DeepFace.extract_faces(image, detector_backend="mtcnn", enforce_detection=False)

        if len(faces) == 0:
            return None  # No face found

        face_data = faces[0]  # Take the first detected face
        facial_area = face_data["facial_area"]

        x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
        face_crop = image[y:y+h, x:x+w]  # Crop the face

        return cv2.resize(face_crop, (160, 160))  # Resize for consistency
    except Exception as e:
        print(f"[ERROR] Face extraction failed: {e}")
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

        # extract_and_show_face(img1)
        # extract_and_show_face(img2)
        face1 = extract_face(img1)
        face2 = extract_face(img2)
        if img1 is None or img2 is None:
            return jsonify({"error": "Invalid image data"}), 400

        # Use DeepFace with Facenet512 model and proper preprocessing
        resultFacenet512 = DeepFace.verify(
            face1, face2, model_name="Facenet512", detector_backend="opencv", enforce_detection=False
        )
        resultFacenet = DeepFace.verify(
            face1, face2, model_name="Facenet", detector_backend="opencv", enforce_detection=False
        )
        resultOpenFace = DeepFace.verify(
            face1, face2, model_name="OpenFace", detector_backend="opencv", enforce_detection=False
        )
        matchFacenet512 = resultFacenet512['verified']
        matchFacenet = resultFacenet['verified']
        # matchOpenFace = resultOpenFace['verified']
        print(resultFacenet)
        print(resultFacenet512)
        print(resultOpenFace)
        similarity = (1 - resultFacenet['distance']) * 100  # Convert distance to percentage

        return jsonify({
            "match": matchFacenet512,
            "similarity": f"{similarity:.2f}"
        })

    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
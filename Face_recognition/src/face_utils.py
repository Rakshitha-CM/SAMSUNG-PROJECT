import os
import pickle
import cv2
import face_recognition
import numpy as np

ENCODINGS_FILE = os.path.join("data", "encodings.pkl")


# ---------------- Load & Save Faces ----------------
def load_known_faces(file_path=ENCODINGS_FILE):
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return {}   # return empty dict if no faces yet
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_known_faces(data, file_path=ENCODINGS_FILE):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


# ---------------- Register New Face ----------------
def register_face(img, name):
    known_faces = load_known_faces()

    # Convert BGR (OpenCV) to RGB (face_recognition expects RGB)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    encodings = face_recognition.face_encodings(rgb_img)

    if len(encodings) == 0:
        return False, "No face detected! Please upload a clear face image."
    elif len(encodings) > 1:
        return False, "Multiple faces detected! Upload an image with only one face."

    # Save the encoding with the given name
    known_faces[name] = encodings[0]
    save_known_faces(known_faces)

    return True, f"✅ Face for '{name}' registered successfully!"


# ---------------- Recognize Faces ----------------
def recognize_faces(frame):
    known_faces = load_known_faces()
    known_names = list(known_faces.keys())
    known_encodings = list(known_faces.values())

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    results = []
    for encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, encoding)
        name = "Unknown"

        if True in matches:
            match_index = np.argmin(face_recognition.face_distance(known_encodings, encoding))
            name = known_names[match_index]

        results.append((name, (top, right, bottom, left)))

    return results

import face_recognition
import os
from PIL import Image, ImageDraw

known_faces_dir = "known_faces"
known_encodings = []
known_names = []

def load_known_faces():
    for filename in os.listdir(known_faces_dir):
        image = face_recognition.load_image_file(f"{known_faces_dir}/{filename}")
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(filename.split('.')[0])

load_known_faces()

def recognize_face(image_path):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    if not face_encodings:
        return {"status": "no_face_detected"}

    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    result_status = "intruder"
    result_name = None

    for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.5)
        name = "Intruso"
        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]
            result_status = "recognized"
            result_name = name

        draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0), width=3)
        draw.text((left + 6, bottom + 5), name, fill=(255, 255, 255))

    del draw
    annotated_path = image_path.replace(".jpg", "_annotated.jpg").replace(".jpeg", "_annotated.jpg")
    pil_image.save(annotated_path)

    return {
        "status": result_status,
        "name": result_name,
        "image_path": annotated_path
    }

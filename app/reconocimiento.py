from flask import Flask, Response, jsonify
import cv2
import face_recognition
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Dirección IP del ESP32-CAM
CAMERA_IP = "192.168.75.42"  # Reemplaza con la IP real de tu ESP32-CAM
VIDEO_URL = f"http://{CAMERA_IP}/stream"

# Carga los rostros conocidos desde el archivo
try:
    with open("encodings.pickle", "rb") as f:
        known_data = pickle.load(f)
    print("[INFO] Datos de entrenamiento cargados correctamente.")
except Exception as e:
    print("[ERROR] No se pudo cargar encodings.pickle:", e)
    known_data = {'encodings': [], 'names': []}

# Estado de detección
unknown_detected = False

def generate_frames():
    global unknown_detected
    cap = cv2.VideoCapture(VIDEO_URL)

    if not cap.isOpened():
        print("[ERROR] No se puede abrir el stream de la ESP32.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Convertir a RGB para face_recognition
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, boxes)

        names = []
        unknown_detected = False

        for encoding in encodings:
            matches = face_recognition.compare_faces(known_data['encodings'], encoding)
            name = "Unknown"

            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                for i in matchedIdxs:
                    name = known_data['names'][i]
                    counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)

            if name == "Unknown":
                unknown_detected = True

            names.append(name)

        # Dibujar cajas y nombres
        for ((top, right, bottom, left), name) in zip(boxes, names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # Codificar como JPEG para enviar como stream
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/video")
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/check_for_unknowns")
def check_for_unknowns():
    return jsonify({"unknownDetected": unknown_detected})

if __name__ == "__main__":
    print("[INFO] Servidor Flask iniciado.")
    app.run(host="0.0.0.0", port=5000, debug=True)

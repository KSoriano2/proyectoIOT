from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import face_recognition
import base64
import pickle
import os
from datetime import datetime
import json

app = Flask(__name__)
CORS(app)  # Permitir CORS para todas las rutas

class FaceRecognitionSystem:
    def __init__(self):
        self.known_faces = []
        self.known_names = []
        self.encodings_file = "face_encodings.pkl"
        self.load_encodings()
    
    def load_encodings(self):
        """Cargar encodings guardados"""
        try:
            if os.path.exists(self.encodings_file):
                with open(self.encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_faces = data['encodings']
                    self.known_names = data['names']
                print(f"Cargados {len(self.known_names)} rostros conocidos")
            else:
                print("No se encontró archivo de encodings, iniciando sistema vacío")
        except Exception as e:
            print(f"Error cargando encodings: {e}")
            self.known_faces = []
            self.known_names = []
    
    def save_encodings(self):
        """Guardar encodings"""
        try:
            data = {
                'encodings': self.known_faces,
                'names': self.known_names
            }
            with open(self.encodings_file, 'wb') as f:
                pickle.dump(data, f)
            print("Encodings guardados correctamente")
        except Exception as e:
            print(f"Error guardando encodings: {e}")
    
    def add_person(self, name, image_base64):
        """Agregar nueva persona al sistema"""
        try:
            # Decodificar imagen
            image_data = base64.b64decode(image_base64)
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detectar y codificar rostro
            face_locations = face_recognition.face_locations(rgb_image)
            
            if len(face_locations) == 0:
                return {"success": False, "message": "No se detectó ningún rostro en la imagen"}
            
            if len(face_locations) > 1:
                return {"success": False, "message": "Se detectaron múltiples rostros. Use una imagen con un solo rostro"}
            
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            if len(face_encodings) > 0:
                # Verificar si la persona ya existe
                if name in self.known_names:
                    # Actualizar encoding existente
                    index = self.known_names.index(name)
                    self.known_faces[index] = face_encodings[0]
                else:
                    # Agregar nueva persona
                    self.known_faces.append(face_encodings[0])
                    self.known_names.append(name)
                
                self.save_encodings()
                return {"success": True, "message": f"Persona '{name}' agregada/actualizada correctamente"}
            
            return {"success": False, "message": "No se pudo codificar el rostro"}
            
        except Exception as e:
            return {"success": False, "message": f"Error procesando imagen: {str(e)}"}
    
    def recognize_face(self, image_base64):
        """Reconocer rostro en imagen"""
        try:
            # Decodificar imagen
            image_data = base64.b64decode(image_base64)
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detectar rostros
            face_locations = face_recognition.face_locations(rgb_image)
            
            if len(face_locations) == 0:
                return {
                    "status": "no_face",
                    "name": "",
                    "confidence": 0.0,
                    "message": "No se detectó ningún rostro"
                }
            
            # Codificar rostros encontrados
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            results = []
            for face_encoding in face_encodings:
                if len(self.known_faces) == 0:
                    results.append({
                        "status": "unknown",
                        "name": "Desconocido",
                        "confidence": 0.0,
                        "message": "No hay rostros registrados en el sistema"
                    })
                    continue
                
                # Comparar con rostros conocidos
                matches = face_recognition.compare_faces(self.known_faces, face_encoding, tolerance=0.5)
                face_distances = face_recognition.face_distance(self.known_faces, face_encoding)
                
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    confidence = (1 - face_distances[best_match_index]) * 100
                    
                    if matches[best_match_index] and confidence > 60:
                        results.append({
                            "status": "known",
                            "name": self.known_names[best_match_index],
                            "confidence": round(confidence, 2),
                            "message": f"Persona reconocida: {self.known_names[best_match_index]}"
                        })
                    else:
                        results.append({
                            "status": "unknown",
                            "name": "Desconocido",
                            "confidence": round(confidence, 2),
                            "message": "Persona no reconocida"
                        })
                else:
                    results.append({
                        "status": "unknown",
                        "name": "Desconocido",
                        "confidence": 0.0,
                        "message": "Error en el reconocimiento"
                    })
            
            # Retornar el primer resultado (o el mejor si hay múltiples)
            return results[0] if results else {
                "status": "error",
                "name": "",
                "confidence": 0.0,
                "message": "Error procesando imagen"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "name": "",
                "confidence": 0.0,
                "message": f"Error: {str(e)}"
            }

# Instancia global del sistema de reconocimiento
face_system = FaceRecognitionSystem()

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Sistema de Reconocimiento Facial ESP32-CAM",
        "status": "active",
        "known_persons": len(face_system.known_names),
        "persons": face_system.known_names,
        "endpoints": {
            "recognize": "/recognize (POST)",
            "add_person": "/add_person (POST)", 
            "persons": "/persons (GET)",
            "delete_person": "/person/<name> (DELETE)",
            "health": "/health (GET)"
        }
    })

@app.route('/recognize', methods=['POST'])
def recognize_face():
    """Endpoint para reconocimiento facial desde ESP32-CAM"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({"error": "No se proporcionó imagen"}), 400
        
        device_id = data.get('device_id', 'unknown')
        result = face_system.recognize_face(data['image'])
        
        # Log para debugging
        print(f"[{datetime.now()}] Reconocimiento desde {device_id}: {result['status']} - {result['name']}")
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"Error en reconocimiento: {str(e)}"}), 500

@app.route('/add_person', methods=['POST'])
def add_person():
    """Endpoint para agregar nuevas personas al sistema"""
    try:
        data = request.get_json()
        
        if not data or 'name' not in data or 'image' not in data:
            return jsonify({"error": "Se requieren 'name' e 'image'"}), 400
        
        result = face_system.add_person(data['name'], data['image'])
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"Error agregando persona: {str(e)}"}), 500

@app.route('/persons', methods=['GET'])
def get_persons():
    """Obtener lista de personas registradas"""
    return jsonify({
        "count": len(face_system.known_names),
        "persons": face_system.known_names
    })

@app.route('/person/<name>', methods=['DELETE'])
def delete_person(name):
    """Eliminar persona del sistema"""
    try:
        if name in face_system.known_names:
            index = face_system.known_names.index(name)
            face_system.known_names.pop(index)
            face_system.known_faces.pop(index)
            face_system.save_encodings()
            return jsonify({"success": True, "message": f"Persona '{name}' eliminada correctamente"})
        else:
            return jsonify({"success": False, "message": f"Persona '{name}' no encontrada"}), 404
    except Exception as e:
        return jsonify({"error": f"Error eliminando persona: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check para Render"""
    return jsonify({
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "known_persons": len(face_system.known_names)
    })

@app.route('/test', methods=['GET'])
def test():
    """Endpoint de prueba"""
    return jsonify({
        "message": "Servidor Flask funcionando correctamente",
        "timestamp": datetime.now().isoformat()
    })

# Manejo de errores
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint no encontrado"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Error interno del servidor"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    print(f"Iniciando servidor Flask en puerto {port}")
    print(f"Sistema de reconocimiento facial listo")
    print(f"Personas registradas: {len(face_system.known_names)}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from face_utils import recognize_face
from cloudinary_utils import upload_to_cloudinary
import shutil
import os

app = FastAPI()

@app.post("/recognize/")
async def recognize(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = recognize_face(temp_path)

    if result["status"] == "no_face_detected":
        os.remove(temp_path)
        return {"status": "no_face_detected"}

    # Si quieres devolver solo imagen anotada directamente
    return FileResponse(result["image_path"], media_type="image/jpeg")

    # --- O ---
    # Si prefieres subirla a Cloudinary (requiere configuración activa):
    # url = upload_to_cloudinary(result["image_path"])
    # os.remove(temp_path)
    # os.remove(result["image_path"])
    # return {"status": result["status"], "name": result["name"], "image_url": url}

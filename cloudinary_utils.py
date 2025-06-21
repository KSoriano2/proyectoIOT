import cloudinary
import cloudinary.uploader

cloudinary.config(
  cloud_name="TU_CLOUD_NAME",
  api_key="TU_API_KEY",
  api_secret="TU_API_SECRET"
)

def upload_to_cloudinary(image_path):
    result = cloudinary.uploader.upload(image_path)
    return result.get("secure_url")

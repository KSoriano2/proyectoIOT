FROM python:3.10-slim

# Dependencias del sistema para face_recognition y dlib
RUN apt-get update && \
    apt-get install -y build-essential cmake \
    libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev \
    libboost-all-dev git wget unzip && \
    apt-get clean

# Crear carpeta para la app
WORKDIR /app

# Copiar los archivos del proyecto
COPY . .

# Instalar dependencias de Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Exponer el puerto
EXPOSE 10000

# Comando para ejecutar la app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]

FROM python:3.11-slim

# Instala dependencias del sistema
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libboost-all-dev \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    libpython3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Crea el directorio de la app
WORKDIR /app

# Copia archivos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app
WORKDIR /app/app

# Expone el puerto para Render
EXPOSE 10000

CMD ["python", "reconocimiento.py"]

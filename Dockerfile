# Gunakan Python base image
FROM python:3.10-slim

# Install libGL dan dependensi sistem
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set direktori kerja
WORKDIR /app

# Copy semua file ke image
COPY . .

# Install dependensi Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Jalankan app Flask
CMD ["python", "app.py"]

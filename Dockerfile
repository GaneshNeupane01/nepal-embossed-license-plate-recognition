FROM python:3.10-slim

WORKDIR /app

# System dependencies (if easyocr / opencv need extras, add here)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

ENV PYTHONUNBUFFERED=1

# Hugging Face Spaces usually expose port 7860
EXPOSE 7860

CMD ["streamlit", "run", "streamlit/app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]

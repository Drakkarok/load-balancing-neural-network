FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY trainer.py .

# Create models directory for shared model files
RUN mkdir -p /app/models

CMD ["python", "trainer.py"]
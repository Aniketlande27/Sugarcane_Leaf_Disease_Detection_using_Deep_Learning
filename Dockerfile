FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements*.txt backend/
RUN pip install --no-cache-dir -r backend/requirements.txt
COPY backend backend
COPY frontend frontend
WORKDIR /app/backend
EXPOSE 8000
# Put your .keras model + class_names.json in backend/model/ BEFORE building.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

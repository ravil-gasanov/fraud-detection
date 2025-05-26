FROM python:3.13-slim

WORKDIR /app

# install system dependencies
COPY requirements-production.txt ./
RUN pip install -r requirements-production.txt

# Copy the application code
COPY . .
RUN pip install -e .

EXPOSE 8080

CMD ["uvicorn", "fraud_detection.predict:app", "--host", "0.0.0.0", "--port", "8080"]
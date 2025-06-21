FROM python:3.13-slim

WORKDIR /app

# install system dependencies
COPY requirements-production.txt ./
RUN pip install -r requirements-production.txt

# Copy the application code
COPY . .
RUN pip install -e .

EXPOSE 8000

CMD ["uvicorn", "fraud_detection.predict:app", "--host", "0.0.0.0", "--port", "8000"]
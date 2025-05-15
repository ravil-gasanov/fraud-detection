FROM python:3.13-slim

WORKDIR /app

COPY . .

RUN pip install pandas==2.2.3 scikit-learn==1.6.1 FastAPI==0.115.12 uvicorn==0.34.2
RUN pip install -e .

EXPOSE 8080

CMD ["uvicorn", "fraud_detection.predict:app", "--host", "0.0.0.0", "--port", "8080"]
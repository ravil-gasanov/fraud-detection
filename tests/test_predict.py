import json
import requests

url = "http://localhost:8000/"


response = requests.get(url)

print(response.json())

data = json.dumps(
    {
        "V4": [1.0],
        "V11": [1241241.0],
        "V7": [-125152.0],
        "Amount": [500000.0],
    }
)


response = requests.post(url=url + "predict", data=data, timeout=10)
print(response.text)

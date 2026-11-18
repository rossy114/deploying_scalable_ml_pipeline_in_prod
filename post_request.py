"""Send an example request to a running API (defaults to a local server)."""
import sys

import requests

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"

record = {
    "age": 45, "workclass": "Private", "fnlgt": 160000, "education": "Doctorate",
    "education_num": 16, "marital_status": "Married-civ-spouse", "occupation": "Prof-specialty",
    "relationship": "Husband", "race": "White", "sex": "Male", "capital_gain": 15024,
    "capital_loss": 0, "hours_per_week": 50, "native_country": "United-States",
}

response = requests.post(f"{URL}/predict", json=record, timeout=10)
print(response.status_code, response.json())

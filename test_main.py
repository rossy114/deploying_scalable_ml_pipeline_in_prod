from fastapi.testclient import TestClient
import pytest
from main import app
# from starter.main import app
import json
import pandas as pd
import os 


client = TestClient(app)

# path=(os.path.dirname(os.getcwd()))
# print(path)


def test_get():
    response = client.get("/welcome")
    assert response.status_code == 200


#Tests Salary <= 50K
def test_post_less_then():
    input_dict = {
				  "age": 54,
				  "workclass": "Private",
				  "fnlgt": 77516,
				  "education": "Masters",
				  "education_num": 9,
				  "marital_status": "Never-married",
				  "occupation": "Handlers-cleaners",
				  "relationship": "Not-in-family",
				  "race": "Black",
				  "sex": "Male",
				  "capital_gain": 2667,
				  "capital_loss": 0,
				  "hours_per_week": 40,
				  "native_country": "United-States"
                    }
    response = client.post("/predict", json=input_dict)
    assert response.status_code == 200
    assert json.loads(response.text)["prediction"] == "Salary <= 50k"

#Tests Salary => 50K
def test_post_greater_then():
    input_dict = {
				  "age": 34,
				  "workclass": "State-gov",
				  "fnlgt": 77516,
				  "education": "Masters",
				  "education_num": 14,
				  "marital_status": "Never-married",
				  "occupation": "Prof-specialty",
				  "relationship": "Not-in-family",
				  "race": "Black",
				  "sex": "Male",
				  "capital_gain": 18084,
				  "capital_loss": 0,
				  "hours_per_week": 40,
				  "native_country": "United-States"
                    }
    response = client.post("/predict", json=input_dict)
    assert response.status_code == 200
    assert json.loads(response.text)["prediction"] == "Salary => 50k"
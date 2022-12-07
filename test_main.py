from fastapi.testclient import TestClient
import pytest
from deploying_scalable_ml_pipeline_in_prod.main import app

import json
import pandas as pd
import os 


client = TestClient(app)

# path=(os.path.dirname(os.getcwd()))
# print(path)


def test_get():
    response = client.get("/welcome")
    assert response.status_code == 200

def test_get2():
    response = client.get("/")
    assert response.status_code == 200

def test_get3():
    response = client.get("/a")
    assert response.status_code == 404



#Tests Salary <= 50K
# def test_post_less_then():
#     input_dict = {
# 				  "age": 54,
# 				  "workclass": "Private",
# 				  "fnlgt": 77516,
# 				  "education": "Masters",
# 				  "education_num": 9,
# 				  "marital_status": "Never-married",
# 				  "occupation": "Handlers-cleaners",
# 				  "relationship": "Not-in-family",
# 				  "race": "Black",
# 				  "sex": "Male",
# 				  "capital_gain": 2667,
# 				  "capital_loss": 0,
# 				  "hours_per_week": 40,
# 				  "native_country": "United-States",
#                     }
#     response = client.post("/predict", json=input_dict)
#     assert response.status_code == 200
#     assert json.loads(response.text)["prediction"] == "Salary <= 50k"

# #Tests Salary => 50K
# def test_post_greater_then():
#     input_dict = {
# 				  "age": 34,
# 				  "workclass": "State-gov",
# 				  "fnlgt": 77516,
# 				  "education": "Masters",
# 				  "education_num": 14,
# 				  "marital_status": "Never-married",
# 				  "occupation": "Prof-specialty",
# 				  "relationship": "Not-in-family",
# 				  "race": "Black",
# 				  "sex": "Male",
# 				  "capital_gain": 18084,
# 				  "capital_loss": 0,
# 				  "hours_per_week": 40,
# 				  "native_country": "United-States"
#                     }
#     response = client.post("/predict", json=input_dict)
#     assert response.status_code == 200
#     assert json.loads(response.text)["prediction"] == "Salary => 50k"

# if __name__ == "__main__":
#     test_get()
#     test_post_less_then()
#     test_post_greater_then()
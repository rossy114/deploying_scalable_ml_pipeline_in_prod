"""API tests."""
import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # the context manager runs the startup hook that loads the model
        yield c


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_unknown_route(client):
    assert client.get("/does-not-exist").status_code == 404


def test_predict_low_income(client):
    record = {
        "age": 19, "workclass": "Private", "fnlgt": 168294, "education": "HS-grad",
        "education_num": 9, "marital_status": "Never-married", "occupation": "Handlers-cleaners",
        "relationship": "Own-child", "race": "White", "sex": "Male", "capital_gain": 0,
        "capital_loss": 0, "hours_per_week": 25, "native_country": "United-States",
    }
    response = client.post("/predict", json=record)
    assert response.status_code == 200
    assert response.json()["prediction"] == "<=50K"


def test_predict_high_income(client):
    record = {
        "age": 45, "workclass": "Private", "fnlgt": 160000, "education": "Doctorate",
        "education_num": 16, "marital_status": "Married-civ-spouse", "occupation": "Prof-specialty",
        "relationship": "Husband", "race": "White", "sex": "Male", "capital_gain": 15024,
        "capital_loss": 0, "hours_per_week": 50, "native_country": "United-States",
    }
    response = client.post("/predict", json=record)
    assert response.status_code == 200
    assert response.json()["prediction"] == ">50K"


def test_predict_rejects_bad_input(client):
    assert client.post("/predict", json={"age": "not a number"}).status_code == 422

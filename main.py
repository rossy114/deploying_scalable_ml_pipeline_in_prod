"""FastAPI app serving the census income classifier."""
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from ml.config import CAT_FEATURES
from ml.data import process_data
from ml.model import inference, load_artifacts

artifacts = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at startup instead of on every request.
    artifacts["model"], artifacts["encoder"], artifacts["lb"] = load_artifacts()
    yield
    artifacts.clear()


app = FastAPI(
    title="Census Income Classifier",
    description="Predicts whether a person earns more than $50K a year from census attributes.",
    version="1.1.0",
    lifespan=lifespan,
)


class CensusInput(BaseModel):
    """One person's census record. JSON uses snake_case; the model uses the CSV's hyphenated names."""

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education_num": 13,
                "marital_status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital_gain": 2174,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States",
            }
        },
    )

    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")


class Prediction(BaseModel):
    prediction: str


@app.get("/")
async def root():
    return {"message": "Census income classifier. POST a record to /predict, or see /docs."}


@app.post("/predict", response_model=Prediction)
async def predict(record: CensusInput):
    df = pd.DataFrame([record.model_dump(by_alias=True)])
    X, _, _, _ = process_data(
        df, categorical_features=CAT_FEATURES, training=False, encoder=artifacts["encoder"]
    )
    label = artifacts["lb"].inverse_transform(inference(artifacts["model"], X))[0]
    return {"prediction": label}

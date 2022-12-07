import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from deploying_scalable_ml_pipeline_in_prod.ml.model import inference
from deploying_scalable_ml_pipeline_in_prod.ml.data import process_data



#Initial a FastAPI instance
app = FastAPI()

# pydantic models
class Data_Input(BaseModel):
    age : int = 34
    workclass : str =  "State-gov"
    fnlgt : int = 77516
    education : str = "Divorced"
    education_num : int = 14
    marital_status : str = "Married-civ-spouse"
    occupation : str = "Handlers-cleaners"
    relationship : str = "Not-in-family"
    race : str = "White"
    sex : str = "Male"
    capital_gain : int = 18084
    capital_loss : int = 0
    hours_per_week : int = 40
    native_country : str = "United-States"

class Data_Output(BaseModel):
    #Prediction will be either >=50K or <=50K 
    prediction: str 


#The Welcome page
@app.get("/")
async def root():
    return {"Welcome"}

# routes
@app.get("/welcome")
async def welcome():
    return {"Welcome": "to the Model!"}


@app.post("/predict", response_model=Data_Output, status_code=200)
def get_prediction(df_temp: Data_Input):
    #Reading the input data
    age = df_temp.age
    workclass = df_temp.workclass
    fnlgt = df_temp.fnlgt
    education = df_temp.education
    education_num = df_temp.education_num
    marital_status = df_temp.marital_status
    occupation = df_temp.occupation
    relationship = df_temp.relationship
    race = df_temp.race
    sex = df_temp.sex
    capital_gain = df_temp.capital_gain
    capital_loss = df_temp.capital_loss
    hours_per_week = df_temp.hours_per_week
    native_country = df_temp.native_country
    #Converted the inputs into Dataframe to be processed 
    df = pd.DataFrame([{"age" : age,
                        "workclass" : workclass,
                        "fnlgt" : fnlgt,
                        "education" : education,
                        "education-num" : education_num,
                        "marital-status" : marital_status,
                        "occupation" : occupation,
                        "relationship" : relationship,
                        "race" : race,
                        "sex" : sex,
                        "capital-gain" : capital_gain,
                        "capital-loss" : capital_loss,
                        "hours-per-week" : hours_per_week,
                        "native-country" : native_country}])
    # Process the data with the process_data function.
    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
] 
    model = pd.read_pickle(r"model.pkl")
    encoder = pd.read_pickle(r"encoder.pkl") 

    X_processed, _, _, _ = process_data(df, categorical_features=cat_features, label = None, training=False, encoder=encoder)
    prediction = inference(model, X_processed)
    
    if prediction == 0:
        prediction = "Salary <= 50k"
    elif prediction == 1:
        prediction = "Salary => 50k"
    #Response dictionary
    response_object = {"prediction": prediction}
    return response_object

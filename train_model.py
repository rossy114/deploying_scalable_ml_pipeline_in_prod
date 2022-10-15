import pandas as pd
from sklearn.model_selection import train_test_split

from starter.ml.data import process_data
from starter.ml.model import train_model
import os


path="nd0821-c3-starter-code/starter"
df = pd.read_csv(os.path.join(path, "data/raw/census.csv"))
df.columns = df.columns.str.strip()
df = df.drop_duplicates()

train_set, test_set = train_test_split(df, test_size=0.20, random_state=42, stratify=df.salary)


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


X_train, y_train, encoder, lb = process_data(
train_set, categorical_features=cat_features, label="salary", training=True
    )
# Train and save a model.
model = train_model(X_train, y_train)


pd.to_pickle(model, os.path.join(path, "model.pkl"))
pd.to_pickle(model, os.path.join(path, "encoder.pkl"))
pd.to_pickle(model, os.path.join(path, "lb.pkl"))

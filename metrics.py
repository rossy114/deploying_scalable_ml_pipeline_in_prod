import pandas as pd
from ml.data import process_data
from ml.model import compute_model_metrics, inference
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, LabelEncoder
import os



# Add code to load in the data, model and encoder
df = pd.read_csv("data/raw/census.csv")
df.columns = df.columns.str.strip()
df = df.drop_duplicates()
model = pd.read_pickle(r"model.pkl")
encoder = pd.read_pickle(r"encoder.pkl") 
# lb = pd.read_pickle(r"lb.pkl")

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

_ , test_set = train_test_split(df, test_size=0.20, random_state=42, stratify=df.salary)
lb = LabelEncoder() 
test_set['salary'] = lb.fit_transform(test_set['salary'])


X_test, y_test, _, _ = process_data(
                test_set,
                cat_features,
                label= None, encoder=encoder, lb=lb, training=False)
y_preds=inference(model, X_test)
y =X_test.iloc[:,-1:]
lb = LabelEncoder() 
y = lb.fit_transform(np.ravel(y))
prc, rcl, fb = compute_model_metrics(y, y_preds)

print(pcr, rcl, fb)
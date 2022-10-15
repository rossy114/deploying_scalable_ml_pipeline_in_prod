import pandas as pd
from starter.ml.data import process_data
from starter.ml.model import compute_model_metrics
import os

path="nd0821-c3-starter-code/starter"

# Add code to load in the data, model and encoder
df = pd.read_csv(os.path.join(path, "data/raw/census.csv"))
df.columns = df.columns.str.strip()
df = df.drop_duplicates()
model = pd.read_pickle(r"starter/model/model.pkl")
encoder = pd.read_pickle(r"starter/model/encoder.pkl") 
# lb = pd.read_pickle(r"starter/model/lb.pkl")

_ , test_set = train_test_split(df, test_size=0.20, random_state=42, stratify=df.salary)


slice_metrics = []
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

# for cls in test_set[cat_features].unique():
#     # df_temp = test_set[test_set[cat_features] == cls]

#     # X_test, y_test, _, _ = process_data(
#     #     df_temp,
#     #     categorical_features=src.common_functions.get_cat_features(),
#     #     label="salary", encoder=encoder, lb=lb, training=False)

#     #     y_preds = model.predict(X_test)

#     #     precision, recall, fb = compute_model_metrics(y_test, y_preds)

#     #     line = "[%s->%s] Precision: %s " \
#     #                "Recall: %s FBeta: %s" % (cat_features, cls, precision, recall, fb)
#     #     logging.info(line)
#     #     slice_values.append(line)

for cat in cat_features:
        for cls in test_set[cat].unique():
            df_temp = test_set[test_set[cat] == cls]
            
#             lb = LabelEncoder() 
#             y = lb.fit_transform(np.ravel(y))
            slice_metrics = []
            X_test, y_test, _, _ = process_data(
                df_temp,
                cat_features,
                label= None, encoder=encoder, lb=lb, training=False)

#             y_preds = model.predict(X_test)
            y_preds=inference(model, X_test)
            y =df_temp.iloc[:,-1:]
            lb = LabelEncoder() 
            y = lb.fit_transform(np.ravel(y))
            prc, rcl, fb = compute_model_metrics(y, y_preds)
            line = "[%s->%s] Precision: %s " \
                   "Recall: %s FBeta: %s" % (cat, cls, prc, rcl, fb)
#             logging.info(line)
            slice_metrics.append(line)
            
with open('data/model/slice_output.txt', 'w') as out:
    for slice_value in slice_values:
        out.write(slice_value + '\n')


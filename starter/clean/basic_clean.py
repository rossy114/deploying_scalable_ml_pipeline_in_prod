"""
Basic cleaning
"""
import pandas as pd

def clean():
	df = pd.read_csv("data//raw/census.csv")

	df.columns = df.columns.str.strip()

	df.to_csv("data/clean/census.csv", index=False)



# df = df.drop_duplicates()
# len(df)-len(df.drop_duplicates())
"""
Basic cleaning module test
"""
import pandas as pd
import pytest
import src.basic_clean


@pytest.fixture
def data():
    """
    Get dataset
    """
    df = pd.read_csv("data/raw/census.csv")
    df.columns = df.columns.str.strip()
    df = src.basic_clean.__clean(df)
    return df


def no_whitespcar(data):
    """
    Data is assumed to have no duplicates
    """
    assert len(df)-len(df.drop_duplicates())


def no_whitespcar(data):
    """
    Data is assumed to have no whitespace
    """
    assert df.workclass.str.isspace().nunique()

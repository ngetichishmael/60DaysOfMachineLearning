#!/urs/bin/env python3
"""Naive bayes implementation using scikit-learn"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score
)

import pandas as pd

def load_data(file_path):
    """
    Load a CSV file into a DataFrame, perform preliminary checks,
    and return the DataFrame.

    args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The DataFrame created from the CSV file.
    """

    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # shape of the DataFrame
        print("Shape of the DataFrame:", df.shape)

        # check for missing values
        print("\nMissing values in each column:")
        print(df.isnull().sum())

        # column datatypes
        print("\nData types of each column:")
        print(df.dtypes)

        # dataframe info
        print("\nDataFrame Info:")
        df.info()

        return df

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
    except pd.errors.ParserError:
        print("Error: There was an issue parsing the file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

        # Return None if there's an error
        return None



# Call the functions
load_data("../dataset/loan_data.csv")

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

def wrangle(df, col, exclude_col=None, skew_threshold=0.75):
    """
    Preprocesses the DataFrame by:
    - transforming the `purpose` column into categorical using get_dummies,
    - applying transformations to features with high skewness to
    normalize distributions

    Parameters:
    df (pd.DataFrame): The DataFrame to preprocess
    col (str): The column to transform into categorical using get_dummies
    skew_threshold (float): apply transformation if skewness is above this

    Returns:
    pd.DataFrame: The preprocessed DataFrame.
    """

    # Check if `purpose` column exists in the DataFrame
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not in df")
    
    # encode `purpose` using get_dummies into a categorical column
    df = pd.get_dummies(df, columns=[col], drop_first=True)

    # Check for skewness in numerical columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    # exclude the target column - it's numerical
    if exclude_col and exclude_col in numeric_columns:
        numeric_columns = numeric_columns.drop(exclude_col)

    skewness = df[numeric_columns].skew()

    # Apply transformations to features with high skewness
    for column in skewness.index:
        if skewness[column] > skew_threshold:
            # Apply log transformation if all values are positive
            if df[column].min() > 0:
                df[column] = np.log(df[column])
                print(f"'{column}' log transformed ({skewness[column]:.2f})")
            else:
                # Apply square root transformation if values include zero or negative
                df[column] = np.sqrt(df[column] - df[column].min())
                print(f"'{column}' square root transformed ({skewness[column]:.2f})")
    
    return df

# Call the functions
df = load_data("../dataset/loan_data.csv")

preprocessed_df = wrangle(df, col="purpose", exclude_col="not.fully.paid")

print(preprocessed_df.columns)

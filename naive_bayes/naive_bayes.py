#!/urs/bin/env python3
"""Naive bayes implementation using scikit-learn"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    classification_report
)


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

    Args:
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

def split_and_scale(df, target_column, test_size=0.2, random_state=42):
    """
    Splits preprocessed df into train and test sets
    Applies standard scaling to the feature sets.

    Args:
        df (pd.DataFrame): preprocessed df
        target_column (str): target column
        test_size (float): dataset portion for the test set
        random_state (int): seed for reproducibility

    Returns:
        X_train_scaled (pd.DataFrame): scaled training feature set
        X_test_scaled (pd.DataFrame): scaled testing feature set
        y_train (pd.Series): training target set
        y_test (pd.Series): testing target set
    """
    
    # split the DataFrame into features (X) and target (y)
    X = df.drop(columns=[target_column], axis=1)
    y = df[target_column]

    # split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
                                                        X, y,
                                                        test_size=test_size,
                                                        random_state=random_state)

    # initialize the StandardScaler
    scaler = StandardScaler()

    # fit and transform the feature set
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # convert the scaled arrays back to DataFrames
    X_train_scaled = pd.DataFrame(X_train_scaled,
                                  columns=X_train.columns,
                                  index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled,
                                 columns=X_test.columns,
                                 index=X_test.index)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def build_and_evaluate_model(X_train, X_test, y_train, y_test):
    """
    Builds a Gaussian Naive Bayes model
    Fits the model to the training data
    Evaluates it on the test data

    Args:
        X_train (pd.DataFrame): scaled training feature set
        X_test (pd.DataFrame): scaled testing feature set
        y_train (pd.Series): training target set
        y_test (pd.Series): testing target set

    Returns:
        model (GaussianNB): the trained Gaussian Naive Bayes model
        evaluation_results (dict): A dictionary containing the evaluation metrics
    """
    
    # initialize the Gaussian Naive Bayes model
    model = GaussianNB()

    # fit the model on the training data
    model.fit(X_train, y_train)

    # predict the target values for the test data
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Store evaluation results in a dictionary
    evaluation_results = {
        'accuracy_score': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'f1_score': f1
    }

    # return the trained model and evaluation results
    return model, evaluation_results

# Call the functions
df = load_data("../dataset/loan_data.csv")

preprocessed_df = wrangle(df, col="purpose", exclude_col="not.fully.paid")

X_train_scaled, X_test_scaled, y_train, y_test = split_and_scale(preprocessed_df, target_column='not.fully.paid')

model, results = build_and_evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test)
print(results)

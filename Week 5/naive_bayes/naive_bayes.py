#!/urs/bin/env python3
"""Naive bayes implementation using scikit-learn"""

import pandas as pd
from io import StringIO
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
        file_path (str): The path to the CSV file

    Returns:
        tuple (pd.DataFrame, dict): 
        - DataFrame created from the CSV file
        - dictionary of metadata
    """

    try:
        # load the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # perform preliminary checks and gather information
        shape = df.shape
        missing_values = df.isnull().sum()
        data_types = df.dtypes

        # creating a buffer to capture info output
        buffer = StringIO()
        df.info(buf=buffer)
        data_info = buffer.getvalue()

        # return the DataFrame and a dictionary with the metadata
        return df, {
            'shape': shape,
            'missing_values': missing_values,
            'data_types': data_types,
            'data_info': data_info
        }

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

def wrangle(df, col):
    """
    Preprocesses the DataFrame by:
    - transforming the `purpose` column into categorical using get_dummies

    Args:
        df (pd.DataFrame): The DataFrame to preprocess
        col (str): The column to transform into categorical using get_dummies

    Returns:
    pd.DataFrame: The preprocessed DataFrame.
    """

    # Check if `purpose` column exists in the DataFrame
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not in df")
    
    # encode `purpose` using get_dummies into a categorical column
    df = pd.get_dummies(df, columns=[col], drop_first=True)

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
        evaluation_results (dict): dictionary containing evaluation metrics
    """
    
    # initialize the Gaussian Naive Bayes model
    model = GaussianNB()

    # fit the model on the training data
    model.fit(X_train, y_train)

    # predict the target values for the test data
    y_pred = model.predict(X_test)

    # calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    # store evaluation results in a dictionary
    evaluation_results = {
        'accuracy_score': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'f1_score': f1
    }

    # return the trained model and evaluation results
    return model, evaluation_results

"""Call the functions"""
# load the data
df, metadata = load_data("../../dataset/loan_data.csv")
if df is not None:
    print("Shape of the DataFrame:", metadata['shape'])
    print("\nMissing values in each column:")
    print(metadata['missing_values'])
    print("\nData types of each column:")
    print(metadata['data_types'])
    print("\nDataFrame Info:")
    print(metadata['data_info'])
print("--------------------------------")

# preprocess df
preprocessed_df = wrangle(df, col="purpose")

# Splitting the data
X_train_scaled, X_test_scaled, y_train, y_test = \
    split_and_scale(preprocessed_df, target_column='not.fully.paid')
print("Shapes after splitting:")
print(X_train_scaled.shape)
print(X_test_scaled.shape)
print(y_train.shape)
print(y_test.shape)
print("--------------------------------")

# Model and evaluation
model, results = build_and_evaluate_model(X_train_scaled,
                                          X_test_scaled,
                                          y_train,
                                          y_test)

print(f"Accuracy Score: {results['accuracy_score']:.4f}")
print("Confusion Matrix:")
print(results['confusion_matrix'])
print("Classification Report:")
print(results['classification_report'])
print(f"F1 Score: {results['f1_score']:.4f}")

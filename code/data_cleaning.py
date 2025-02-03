"""
data_cleaning.py

Contains functions for handling data cleaning, date conversions,
and duplicate removal.
"""

import pandas as pd

def fill_missing_dob(dataframe, year_column, dob_column):
    """
    Fills missing date of births in the specified column using
    the year of birth. The missing dates are filled as January 15th
    of the corresponding year.

    Parameters:
    - dataframe: pd.DataFrame, the dataframe containing the data
    - year_column: str, the name of the column containing the year of birth
    - dob_column: str, the name of the column where missing dates of birth
      will be filled

    Returns:
    - pd.DataFrame: The dataframe with missing dates of birth filled
    """
    for index, row in dataframe[dataframe[dob_column].isnull()].iterrows():
        year = row[year_column]
        if pd.notnull(year):
            dataframe.loc[index, dob_column] = f"{int(year)}-01-15"
    return dataframe


def convert_dates_datatype(dataframe, date_columns):
    """
    Converts specified columns to datetime datatype.

    Parameters:
    - dataframe: pd.DataFrame, the dataframe containing the data
    - date_columns: list, the names of the columns to convert to datetime

    Returns:
    - pd.DataFrame: The dataframe with converted datetime columns
    """
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)
    return dataframe


def drop_duplicates(dataframe):
    """
    Drops duplicate rows from the dataframe.

    Parameters:
    - dataframe: pd.DataFrame, the dataframe containing the data

    Returns:
    - pd.DataFrame: The dataframe with duplicates removed
    """
    dataframe.drop_duplicates(inplace=True)
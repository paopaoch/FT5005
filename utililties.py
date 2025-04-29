import pandas as pd
from companies import *
import numpy as np

import pandas as pd

class Winsorizer():
    """
    A class wrapper for winsorizing a specific column in a dataframe.

    This version allows winsorization on a single specified column,
    while storing the calculated limits for future use.

    Mimics the sklearn API with fit, transform, and fit_transform methods.
    """
    def __init__(self, column: str, limits: list = None):
        self.column = column
        self.lower_lim = None
        self.upper_lim = None
        if limits:
            self.limits = limits[:2]
        else:
            self.limits = [0.05, 0.95]

    def fit(self, input_df: pd.DataFrame):
        """Compute the lower and upper quantile limits for the specified column."""
        if self.column not in input_df.columns:
            raise ValueError(f"Column '{self.column}' not found in dataframe.")

        col_data = input_df[self.column]
        self.lower_lim = col_data.quantile(self.limits[0])
        self.upper_lim = col_data.quantile(self.limits[1])
        return self

    def transform(self, input_df: pd.DataFrame):
        """Apply the winsorization to the specified column."""
        if self.upper_lim is None or self.lower_lim is None:
            raise LookupError("No limits were defined. Hint: Run a fit function first.")
        if self.column not in input_df.columns:
            raise ValueError(f"Column '{self.column}' not found in dataframe.")

        df = input_df.copy()
        df[self.column] = df[self.column].clip(lower=self.lower_lim, upper=self.upper_lim)
        return df

    def fit_transform(self, input_df: pd.DataFrame):
        """Convenience method to fit and transform in one step."""
        return self.fit(input_df).transform(input_df)


def log_transform(df, column):
    """signed log transformation on a single dataframe column."""
    df = df.copy()
    df[column] = np.sign(df[column]) * np.log1p(np.abs(df[column]))
    return df


def winsorize_and_log_transform(df, column, limits=(0.01, 0.99), apply_log=True, winsorizer: Winsorizer=None):
    """
    Performs winsorization followed by signed log transformation on a single dataframe column.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe containing the column to transform
    
    column : str
        The name of the column to transform
    
    limits : tuple, default (0.01, 0.99)
        A tuple of (lower, upper) limits for winsorization
        
    apply_log : bool, default True
        Whether to apply signed log transformation after winsorization
        
    winsorizer : Winsorizer, default None
        A pre-fitted Winsorizer instance. If None, a new Winsorizer will be created with the specified limits.
        
    Returns:
    --------
    dict
        A dictionary containing:
        - 'transformed': The transformed pandas Series (does not modify the original dataframe)
        - 'winsorizer': The Winsorizer instance used for the transformation
    """
    # Extract column as a DataFrame with single column to use with Winsorizer
    col_df = df[[column]].copy()
    
    # Use provided winsorizer or create a new one
    if winsorizer is None:
        winsorizer = Winsorizer(limits=[limits[0], limits[1]])
        transformed_df = winsorizer.fit_transform(col_df)
    else:
        transformed_df = winsorizer.transform(col_df)
    
    # Extract the transformed values as a series
    transformed = transformed_df[column]
    
    # Apply signed log transformation if requested
    if apply_log:
        transformed = np.sign(transformed) * np.log1p(np.abs(transformed))
    
    # Return both the transformed series and the winsorizer
    return {
        'transformed': transformed,
        'winsorizer': winsorizer
    }


def transform_multiple_columns(df, columns, limits_list=None, apply_log=True):
    """
    Performs winsorization followed by signed log transformation on multiple columns of a dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe containing the columns to transform
    
    columns : list
        List of column names to transform
    
    limits_list : list or None, default None
        List of (lower, upper) tuples defining the limits for each column.
        If None, default limits (0.01, 0.99) will be used for all columns.
        Must be the same length as columns if provided.
        
    apply_log : bool or list, default True
        Whether to apply signed log transformation after winsorization.
        If bool, the same value will be used for all columns.
        If list, must be the same length as columns.
        
    Returns:
    --------
    dict
        A dictionary containing:
        - 'transformed_df': A new dataframe with the transformed columns
        - 'winsorizers': A dictionary mapping column names to their Winsorizer instances
    """
    # Make a copy of the input dataframe
    result_df = df.copy()
    
    # Standardize limits_list
    if limits_list is None:
        limits_list = [(0.01, 0.99)] * len(columns)
    elif len(limits_list) != len(columns):
        raise ValueError("limits_list must have the same length as columns")
    
    # Standardize apply_log
    if isinstance(apply_log, bool):
        apply_log_list = [apply_log] * len(columns)
    elif len(apply_log) != len(columns):
        raise ValueError("If apply_log is a list, it must have the same length as columns")
    else:
        apply_log_list = apply_log
    
    # Dictionary to store Winsorizer instances
    winsorizers = {}
    
    # Apply transformation to each column
    for i, column in enumerate(columns):
        # Check if column exists
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in dataframe")
        
        # Apply transformation
        result = winsorize_and_log_transform(
            df, 
            column, 
            limits=limits_list[i], 
            apply_log=apply_log_list[i]
        )
        
        # Store transformed values and Winsorizer
        result_df[column] = result['transformed']
        winsorizers[column] = result['winsorizer']
    
    return {
        'transformed_df': result_df,
        'winsorizers': winsorizers
    }


def interpolate_missing_values(df, feature, limit_direction='both'):
    """
    Fill missing values in the feature column.
    
    Parameters:
    -----------
    df : DataFrame
        The input dataframe
    feature : str
        Column name to fill
    limit_direction : str, default 'both'
        If 'forward', interpolate and forward fill NaNs after valid values
        If 'backward', interpolate and backward fill NaNs before valid values
        If 'both', interpolate then apply both forward and backward fill
    
    Returns:
    --------
    DataFrame with filled values
    """
    df = df.copy()
    
    if limit_direction == 'forward':
        # First interpolate
        df[feature] = df[feature].interpolate(method='linear')
        # Then forward fill any remaining NaNs after valid values
        df[feature] = df[feature].ffill()
    elif limit_direction == 'backward':
        # First interpolate
        df[feature] = df[feature].interpolate(method='linear')
        # Then backward fill any remaining NaNs before valid values
        df[feature] = df[feature].bfill()
    else:  # 'both' or any other value
        # Interpolate first
        df[feature] = df[feature].interpolate(method='linear')
        # Then apply both forward and backward fill
        df[feature] = df[feature].ffill().bfill()
    
    return df


def fill_with_industry_average(df: pd.DataFrame, ticker, size_array, feature):
    """
    Fill NaN values in the feature column with the industry average for the given ticker.
    """
    # Get the industry average for the given ticker for each quarter
    df = df.copy()
    # select quarters with ticker data
    quarters = df[df['tic'] == ticker]['datacqtr'].unique()
    mask = (df['tic'] != ticker) & (df['datacqtr'].isin(quarters)) & (df['tic'].isin(size_array))
    industry_avg = df[mask].groupby('datacqtr')[feature].mean()
    industry_avg = industry_avg.reset_index()
    industry_avg["tic"] = ticker
    df = df[df['tic'] != ticker]
    df = pd.concat([df, industry_avg], ignore_index=True)
    return df


def get_bank_peers(ticker):
    """
    Find the size of the bank based on the ticker.
    """
    if ticker in large_banks:
        return large_banks
    elif ticker in medium_banks:
        return medium_banks
    elif ticker in small_banks:
        return small_banks
    else:
        raise ValueError(f"Ticker {ticker} not found in any bank size category.")


def check_if_all_nan_by_ticker(df, feature):
    """
    Check if all values in the 'Net Interest Income' column are NaN for each ticker.
    """
    empty_tickers = []
    for ticker in df['tic'].unique():
        if df[df['tic'] == ticker][feature].isna().all():
            empty_tickers.append(ticker)
    return empty_tickers
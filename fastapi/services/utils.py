from typing import Any, List, Callable
from ast import literal_eval

import pandas as pd

def parse_int(value: Any) -> int:
    if pd.isna(value):
        return None
    try:
        return int(value)
    except ValueError:
        return value


def parse_boolean(value: Any) -> bool:
    if pd.isna(value):
        return None
    value = str(value).lower().strip()
    if value in ['true', 'yes', 'y', '1']:
        return True
    elif value in ['false', 'no', 'n', '0']:
        return False
    else:
        try:
            return bool(float(value))
        except ValueError:
            return value
        
def convert_to_list(x: Any, to_type: Callable[[Any], Any]=lambda x: x) -> List:
    list_types = (list, tuple, set)
    if isinstance(x, list_types):
        x = list(x)
        return [to_type(item) for item in x]
    if isinstance(x, str):
        try:
            x = list(literal_eval(x))
            return [to_type(item) for item in x]
        except (ValueError, SyntaxError):
            return x
    return x
        
# Reference: https://stackoverflow.com/questions/55562696/how-to-replace-missing-values-with-group-mode-in-pandas
def fast_mode(df: pd.DataFrame, key_columns: List[str], target_column: str):
    """
    Calculate a column mode, by group, ignoring null values.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame over which to calcualate the mode.
    key_columns : list of str
        Columns to groupby for calculation of mode.
    target_column : str
        Column for which to calculate the mode.

    Returns
    -------
    pandas.DataFrame
        One row for the mode of value_col per key_cols group. If ties,
        returns the one which is sorted first.
    """
    return (df.groupby(key_columns + [target_column]).size()
              .to_frame('counts').reset_index()
              .sort_values('counts', ascending=False)
              .drop_duplicates(subset=key_columns)).drop(columns='counts')
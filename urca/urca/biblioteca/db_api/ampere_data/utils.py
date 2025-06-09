import pandas as pd


def dateStr2Timestamp(date: str):
    return pd.to_datetime(date, format='%Y%m%d')

def dateTimestamp2Str(date) -> str:
    if isinstance(date, pd.Series):
        return date.dt.strftime('%Y%m%d')
    return date.strftime('%Y%m%d')
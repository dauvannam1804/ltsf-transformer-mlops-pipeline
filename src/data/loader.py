import pandas as pd
import numpy as np

def load_stock_csv(path: str):
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    df["daily_return"] = df["close"].pct_change()
    df["close_log"] = np.log(df["close"])

    df["daily_return"].fillna(0, inplace=True)

    return df

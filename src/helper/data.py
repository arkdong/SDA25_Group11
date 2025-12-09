import polars as pl
from datetime import datetime


def load_btc(csv_path):
    """
    Load BTC OHLCV CSV data using Polars, ensure chronological order,
    and optionally return one of six equal-sized parts.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.

    part : int or None
        If 1–6, return the corresponding 1/6 slice.
        If None, return the full dataset.

    Returns
    -------
    pl.DataFrame
    """

    schema = {
        "timestamp": pl.Datetime,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Float64,
        "tes": pl.Int64
    }

    df = (
        pl.read_csv(
            csv_path,
            schema=schema,
            has_header=True,
            separator=","
        )
        .sort("timestamp")  # ensure chronological order
    )


    return df

def load_data_sentiment(csv_path):
    schema = {
        "timestamp": pl.Datetime,
        "text": pl.Utf8,
        "text_en": pl.Utf8,
        "sentiment_score": pl.Float64,
        "sentiment_label": pl.Utf8
    }

    df = (
        pl.read_csv(
            csv_path,
            schema=schema,
            has_header=True,
            separator=","
        )
        .sort("timestamp")  # ensure time order
    )
    return df
def load_btc_2018_2019(csv_path, part=None):
    """
    Load BTC OHLCV CSV data using Polars, ensure chronological order,
    and optionally return one of six equal-sized parts.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.

    part : int or None
        If 1–6, return the corresponding 1/6 slice.
        If None, return the full dataset.

    Returns
    -------
    pl.DataFrame
    """

    schema = {
        "timestamp": pl.Datetime,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Float64,
    }

    df = (
        pl.read_csv(
            csv_path,
            schema=schema,
            has_header=True,
            separator=","
        )
        .sort("timestamp")  # ensure chronological order
    )

    # If no slicing requested
    if part is None:
        return df

    # Validate requested part
    if not (1 <= part <= 6):
        raise ValueError("part must be an integer between 1 and 6, or None.")

    # Fixed absolute boundaries for the year
    year_start = datetime(2018, 1, 1)
    year_end   = datetime(2019, 1, 1)

    total_span = year_end - year_start
    step = total_span / 6

    slice_start = year_start + step * (part - 1)
    slice_end   = year_start + step * part

    df_part = df.filter(
        (pl.col("timestamp") >= slice_start) &
        (pl.col("timestamp") < slice_end)
    )

    return df_part
if __name__ == "__main__":
    df1 = load_btc("data/btc_training.csv")

    print(df1.head(2))
    df2 = load_data_sentiment("data/sentiment/tweets_1_sent.csv")
    print(df2.head(2))

    df1 = load_btc_2018_2019("data/btc_2018_2019.csv", 5)
    df1.write_csv("data/btc_5.csv")
    print(df1.head(2))
    df1 = load_btc_2018_2019("data/btc_2018_2019.csv", 6)
    df1.write_csv("data/btc_6.csv")
    print(df1.head(2))
    # df2 = load_btc_2018_2019("data/btc_2018_2019.csv", 2)
    # df3 = load_btc_2018_2019("data/btc_2018_2019.csv", 3)
    # df4 = load_btc_2018_2019("data/btc_2018_2019.csv", 4)
    # df_merge = pl.concat([df1, df2, df3, df4])
    # df_merge.write_csv("data/btc_training.csv")
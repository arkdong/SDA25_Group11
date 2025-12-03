import polars as pl
from datetime import datetime


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

def load_btc(csv_path):
    """
    Load OHLCV dataset of bitcoin with Unix timestamps into a Polars DataFrame.

    Expected columns:
        Timestamp (Unix time)
        Open
        High
        Low
        Close
        Volume

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame with proper schema and timestamp parsed to Datetime.
    """

    schema = {
        "timestamp": pl.Float64,   # Unix timestamp in seconds
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Float64,
    }

    df = pl.read_csv(
        csv_path,
        schema=schema,
        has_header=True,
        separator=","
    )

    # Convert Unix timestamp → Datetime
    df = df.with_columns(
        pl.from_epoch(pl.col("timestamp"), time_unit="s").alias("timestamp")
    )

    return df

def load_btc_2018_2019(csv_path):
    schema = {
        "timestamp": pl.Datetime,   # Unix timestamp in seconds
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Float64,
    }

    df = pl.read_csv(
        csv_path,
        schema=schema,
        has_header=True,
        separator=","
    )

    return df

def load_google_search(csv_path):
    schema = {
        "date": pl.Datetime,   # Unix timestamp in seconds
        "Scale_['bitcoin']": pl.Float64
    }

    df = pl.read_csv(
        csv_path,
        schema=schema,
        has_header=True,
        separator=","
    )

    return df


if __name__ == "__main__":
    df1 = load_btc_2018_2019("data/btc_2018_2019.csv")
    print(df1.head(2))
    df2 = load_data_sentiment("data/sentiment/tweets_1_sent.csv")
    print(df2.head(2))

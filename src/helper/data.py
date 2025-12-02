import polars as pl

def load_16M_tweet(columns, csv_path):
    """
    Load the 16M Tweet dataset CSV using a predefined schema.

    Parameters
	    - columns : list[str]
	        List of column names to select. If empty, all columns are returned.
	    - csv_path : str
	        Path to the tweet CSV file based on the current location.

    Returns
	    - pl.DataFrame
	        A Polars DataFrame containing either all the columns or the specified subset.
    """
    schema = {
        "id": pl.Utf8,            # Tweet ID (string or pl.Int64 if numeric)
        "user": pl.Utf8,          # Username
        "fullname": pl.Utf8,      # Full name of the user
        "url": pl.Utf8,           # Tweet URL
        "timestamp": pl.Datetime, # Timestamp of the tweet
        "replies": pl.Int64,      # Number of replies
        "likes": pl.Int64,        # Number of likes
        "retweets": pl.Int64,     # Number of retweets
        "text": pl.Utf8,          # Tweet content
    }

    df = pl.read_csv(
        csv_path,
        separator=";",
        schema=schema,
        has_header=True
    )

    # If columns is empty, return all columns
    if not columns:
        return df

    return df.select(columns)

def load_tweet_with_sentiment(columns, csv_path):
    """
    Load a sentiment-labeled tweet CSV file using a predefined schema.

    Parameters
        - columns : list[str]
            List of column names to select. If empty, all columns are returned.
        - csv_path : str
            Path to the sentiment tweet CSV file (comma-separated).

    Returns
        - pl.DataFrame
            A Polars DataFrame containing either all the columns or the specified subset.
    """
    schema = {
        "Date": pl.Datetime,  # Timestamp of the tweet
        "text": pl.Utf8,      # Tweet content
        "Sentiment": pl.Utf8  # Sentiment label (e.g., Positive/Negative)
    }

    df = pl.read_csv(
        csv_path,
        separator=",",
        schema=schema,
        has_header=True
    )

    # Convert all column names to lowercase
    df = df.rename({col: col.lower() for col in df.columns})

    if not columns:
        return df

    return df.select(columns)

def load_reddit_comments(columns, csv_path):
    """
    Load a Reddit comments CSV file using a fixed schema.

    Parameters:
        - columns : list[str]
            List of column names to select. If empty, all columns are returned.
        - csv_path : str
            Path to the Reddit comments CSV file.

    Returns
        - pl.DataFrame
            A Polars DataFrame containing either all columns or the specified subset.
    """

    schema = {
        "index": pl.Int64,             # Added index column (first column in CSV)
        "timestamp": pl.Datetime,       # Datetime of the comment
        "date": pl.Utf8,               # Day of the comment (string)
        "author": pl.Utf8,             # Reddit username
        "subreddit": pl.Utf8,          # Subreddit name
        "created_utc": pl.Int64,       # Unix timestamp (int)
        "score": pl.Float64,           # Upvotes/Downvotes count
        "controversiality": pl.Int64,  # Controversiality score
        "text": pl.Utf8                # Comment text
    }

    df = pl.read_csv(
        csv_path,
        separator=",",
        schema=schema,
        has_header=True
    )

    # If columns is empty, return all columns
    if not columns:
        return df

    return df.select(columns)

def store_tweets_2018_2019(csv_path):
    # 1. Load the subset of columns you need
    df = load_16M_tweet(["timestamp", "text"], "data/tweets.csv")
    filtered = df.filter(
            (pl.col("timestamp") >= pl.datetime(2018, 1, 1)) &
            (pl.col("timestamp") <  pl.datetime(2019, 1, 1))
        )
    filtered.write_csv(csv_path)

def load_timestamp_text_2018_2019(csv_path):
    """
    Load a CSV containing 'timestamp' and 'text' columns using Polars.

    Expected format:
        2018-09-13T19:36:04.000000,"Some text..."

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.

    Returns
    -------
    pl.DataFrame
    """
    schema = {
        "timestamp": pl.Datetime,   # ISO8601 with microseconds
        "text": pl.Utf8
    }

    df = pl.read_csv(
        csv_path,
        schema=schema,
        has_header=True,
        separator=","
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


if __name__ == "__main__":
    # df = load_timestamp_text_2018_2019("data/reddit_2018_2019.csv")
    # print(df.head(2))
    df = load_btc("data/btc.csv")
    print(df.head(2))
    df = load_btc_2018_2019("data/btc_2018_2019.csv")
    print(df.head(2))

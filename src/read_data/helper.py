import polars as pl

def load_16M_tweet(columns, csv_path):
    schema = {
        "id": pl.Utf8,            # Or pl.Int64 if numeric
        "user": pl.Utf8,
        "fullname": pl.Utf8,
        "url": pl.Utf8,
        "timestamp": pl.Datetime, # or pl.Utf8 if raw text
        "replies": pl.Int64,
        "likes": pl.Int64,
        "retweets": pl.Int64,
        "text": pl.Utf8,
    }

    df = pl.read_csv(
        csv_path,
        separator=";",
        schema=schema,
        has_header=True
    ).select(columns)

    return df

if __name__ == "__main__":
    df = load_16M_tweet(["id", "timestamp", "text"], "data/tweets.csv")
    print(df.head(2))

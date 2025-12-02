import polars as pl
from nltk.sentiment import SentimentIntensityAnalyzer
from langdetect import detect, LangDetectException
from data import load_16M_tweet

# crypto / bitcoin / hype slang lexicon (extend or tweak as you like)
HYPE_LEXICON = {
    "hodl": 3.0,
    "hodling": 3.0,
    "hodler": 2.5,
    "moon": 2.5,
    "mooning": 3.0,
    "tothemoon": 3.0,
    "lambo": 3.0,
    "lamboooo": 3.5,
    "bullish": 2.5,
    "bearish": -2.5,
    "rekt": -3.0,
    "bagholder": -2.0,
    "fomo": 1.5,
    "shitcoin": -2.5,
    "scam": -3.0,
    "pump": 1.5,
    "pumping": 2.0,
    "dump": -2.0,
    "dumping": -2.5,
    "whale": 1.5,
    "diamondhands": 3.0,
    "paperhands": -2.0,
    "🚀": 3.0,
}


def build_vader_with_hype(hype_lexicon: dict | None = None) -> SentimentIntensityAnalyzer:
    """
    Create a VADER SentimentIntensityAnalyzer and extend its lexicon
    with custom hype words.
    """
    sia = SentimentIntensityAnalyzer()
    if hype_lexicon is None:
        hype_lexicon = HYPE_LEXICON

    # Update / add hype words to VADER's lexicon
    sia.lexicon.update({k.lower(): v for k, v in hype_lexicon.items()})
    return sia

def safe_detect_lang(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return "unknown"
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Pick a multilingual-to-English model; check HF docs for other options
TRANSLATOR_MODEL = "Helsinki-NLP/opus-mt-mul-en"

translation_pipe = pipeline(
    "translation",
    model=TRANSLATOR_MODEL,
    tokenizer=TRANSLATOR_MODEL,
)


def translate_to_english(text: str, src_lang: str | None = None) -> str:
    """
    Translate `text` to English using a local seq2seq model.
    If the text is already English, just return it.
    """
    if not isinstance(text, str) or not text.strip():
        return text

    # Detect language if not provided
    lang = src_lang or safe_detect_lang(text)

    if lang == "en":
        return text

    try:
        out = translation_pipe(text, max_length=256)
        return out[0]["translation_text"]
    except Exception as e:
        # Fallback: if translation fails, return original
        # (better than crashing the whole pipeline)
        return text


def add_sentiment_score(
    df: pl.DataFrame,
    text_col: str = "text",
    hype_lexicon: dict | None = None,
    new_col: str = "sentiment_score",
    translation_col: str = "text_en",
    log_every: int = 100,  # for 16M rows you'll want this fairly large
) -> pl.DataFrame:
    """
    Add:
      - a translated-to-English text column
      - a sentiment score column (VADER + hype lexicon)
    to a Polars DataFrame.

    - df: Polars DataFrame with a text column.
    - text_col: original text column name.
    - hype_lexicon: optional custom lexicon dict; if None, uses HYPE_LEXICON.
    - new_col: name of the sentiment score column.
    - translation_col: name of the English text column.
    - log_every: print progress every N rows.
    """
    analyzer = build_vader_with_hype(hype_lexicon)
    total = df.height

    # ---------- 1) Translate & store in a separate column ----------
    counter = {"n": 0}
    print(f"Starting translation for {total:,} texts...", flush=True)

    def _translate(text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            translated = ""
        else:
            lang = safe_detect_lang(text)
            translated = translate_to_english(text, src_lang=lang)

        counter["n"] += 1
        n = counter["n"]
        if log_every and n % log_every == 0:
            pct = n / total * 100
            print(
                f"Translated {n:,}/{total:,} texts ({pct:.1f}%)",
                flush=True,
            )
        return translated

    df = df.with_columns(
        pl.col(text_col)
        .map_elements(_translate, return_dtype=pl.Utf8)
        .alias(translation_col)
    )

    print("Translation finished. Starting sentiment scoring...", flush=True)

    # ---------- 2) Sentiment from translated text ----------
    counter["n"] = 0  # reset

    def _score(text_en: str) -> float:
        if not isinstance(text_en, str) or not text_en.strip():
            value = 0.0
        else:
            value = analyzer.polarity_scores(text_en)["compound"]

        counter["n"] += 1
        n = counter["n"]
        if log_every and n % log_every == 0:
            pct = n / total * 100
            print(
                f"Scored {n:,}/{total:,} texts ({pct:.1f}%)",
                flush=True,
            )
        return value

    df = df.with_columns(
        pl.col(translation_col)
        .map_elements(_score, return_dtype=pl.Float64)
        .alias(new_col)
    )

    print("Sentiment scoring finished.", flush=True)
    return df

if __name__ == "__main__":
    # 1. Load the subset of columns you need
    df = load_16M_tweet(["timestamp", "text"], "data/tweets.csv")
    print(df.head(2))

    # 2. Add translation + sentiment score
    tweets_with_sentiment = add_sentiment_score(
        df.limit(1000),
        text_col="text",
        new_col="sentiment_score",
        translation_col="text_en",
        log_every=100_000,  # tweak depending on how spammy you want logs
    )

    # 3. Save the resulting DataFrame to CSV
    #    (you might prefer Parquet for speed/size: .write_parquet(...))
    output_path = "data/tweets_with_sentiment.csv"
    tweets_with_sentiment.write_csv(output_path)
    print(f"Saved enriched data with sentiment and translation to {output_path}")

    # 4. Resample and plot sentiment over time
    df_resampled = (
        tweets_with_sentiment.with_columns(
            pl.col("timestamp")
            .dt.truncate("1m")  # choose "10s", "5m", "1h", etc
            .alias("ts_bin")
        )
        .group_by("ts_bin", maintain_order=True)
        .agg(
            pl.col("sentiment_score").mean().alias("sentiment_mean")
        )
        .sort("ts_bin")
    )

    pdf = df_resampled.to_pandas()

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(pdf["ts_bin"], pdf["sentiment_mean"])
    plt.xlabel("Timestamp")
    plt.ylabel("Sentiment Score")
    plt.title("Sentiment Over Time")
    plt.grid(True)
    plt.show()

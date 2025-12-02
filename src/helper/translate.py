import math
import logging
from pathlib import Path

import polars as pl
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from langdetect import detect, LangDetectException

from helper import load_16M_tweet

# ----------------- Logging setup -----------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ----------------- Model setup -------------------

model_name = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)

# ----------------- Helpers -----------------------


def safe_detect_lang(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return "unknown"
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def translate_local(text: str, tgt_lang: str = "en") -> str:
    """
    Translate a single text string to `tgt_lang`.
    Includes basic error handling so a single bad sample
    doesn't crash the whole run.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    try:
        src_lang = safe_detect_lang(text)
        # Fallback if detection fails
        if src_lang == "unknown":
            src_lang = "en"

        tokenizer.src_lang = src_lang
        encoded = tokenizer(text, return_tensors="pt", truncation=True)
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.get_lang_id(tgt_lang),
        )
        return tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )[0]
    except Exception as e:
        logging.exception("Translation failed for text starting with: %r", text[:50])
        # Fallback: return original text so we don't lose data
        return text


def translate_chunk(
    df: pl.DataFrame,
    chunk_idx: int,
    start_row: int,
    end_row: int,
    out_dir: Path,
    tgt_lang: str = "en",
) -> None:
    """
    Translate a slice of the DataFrame [start_row, end_row)
    and store it as a separate CSV.
    """
    out_path = out_dir / f"tweets_translated_chunk_{chunk_idx:02d}.csv"

    if out_path.exists():
        logging.info(
            "Chunk %d already exists at %s. Skipping.",
            chunk_idx,
            out_path,
        )
        return

    logging.info(
        "Processing chunk %d: rows [%d, %d)",
        chunk_idx,
        start_row,
        end_row,
    )

    # Slice the chunk
    length = end_row - start_row
    df_chunk = df.slice(start_row, length)

    # Add translated column
    df_chunk = df_chunk.with_columns(
        pl.col("text")
        .map_elements(lambda s: translate_local(s, tgt_lang=tgt_lang), return_dtype=pl.Utf8)
        .alias("text_en")
    )

    # Save chunk
    df_chunk.write_csv(out_path)
    logging.info(
        "Finished chunk %d. Rows: %d. Saved to %s",
        chunk_idx,
        df_chunk.height,
        out_path,
    )


def combine_chunks(out_dir: Path, n_chunks: int, combined_name: str = "tweets_translated_full.csv") -> None:
    """
    Combine all chunk CSVs into one big CSV file.
    Only runs if all chunk files exist.
    """
    combined_path = out_dir / combined_name
    if combined_path.exists():
        logging.info("Combined file %s already exists. Skipping combine.", combined_path)
        return

    chunk_files = [
        out_dir / f"tweets_translated_chunk_{i:02d}.csv"
        for i in range(n_chunks)
    ]

    missing = [str(f) for f in chunk_files if not f.exists()]
    if missing:
        logging.warning(
            "Cannot combine chunks; missing %d chunk(s): %s",
            len(missing),
            ", ".join(missing),
        )
        return

    logging.info("Combining %d chunk files into %s", n_chunks, combined_path)

    # For big data, using scan_csv is more memory-friendly
    lazy_frames = [pl.scan_csv(str(f)) for f in chunk_files]
    lf_all = pl.concat(lazy_frames)
    lf_all.sink_csv(str(combined_path))

    logging.info("Combined file written to %s", combined_path)


# ----------------- Main script -------------------


if __name__ == "__main__":
    NUM_CHUNKS = 10
    OUTPUT_DIR = Path("data/translated")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("Loading tweets DataFrame...")
    df = load_16M_tweet(["timestamp", "text"], "data/tweets.csv")
    logging.info("Loaded DataFrame with %d rows and %d columns.", df.height, len(df.columns))

    n_rows = df.height
    chunk_size = math.ceil(n_rows / NUM_CHUNKS)
    logging.info("Splitting into %d chunks. Chunk size ≈ %d rows.", NUM_CHUNKS, chunk_size)

    for chunk_idx in range(NUM_CHUNKS):
        start_row = chunk_idx * chunk_size
        if start_row >= n_rows:
            logging.info(
                "Chunk %d start index %d is beyond total rows %d. Stopping.",
                chunk_idx,
                start_row,
                n_rows,
            )
            break

        end_row = min(start_row + chunk_size, n_rows)

        try:
            translate_chunk(
                df=df,
                chunk_idx=chunk_idx,
                start_row=start_row,
                end_row=end_row,
                out_dir=OUTPUT_DIR,
                tgt_lang="en",
            )
        except Exception:
            logging.exception("Unexpected error while processing chunk %d.", chunk_idx)
            # If something crashes, you can just rerun the script;
            # finished chunks will be skipped because their CSVs already exist.
            break

    # Try to combine all chunks into one big CSV
    combine_chunks(OUTPUT_DIR, NUM_CHUNKS)

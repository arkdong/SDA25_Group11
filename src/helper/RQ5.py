import polars as pl
import matplotlib.pyplot as plt
import numpy as np

import helper.RQ4 as RQ4
import helper.data as data


def build_daily_return_model_df(
    btc_daily: pl.DataFrame,
    sent_daily: pl.DataFrame,
    trends_daily: pl.DataFrame | None = None,
    target_col: str = "absret_daily",
) -> pl.DataFrame:

    # Join BTC, sentiment, and trend data on date.
    df = btc_daily.join(sent_daily, on="date", how="left").sort("date")
    df = df.join(trends_daily, on="date", how="left").sort("date")

    # Choose level features to lag.
    level_cols = ["twitter_mean", "reddit_mean"]  # from sentiment_disagreement_daily output

    # Disagreement features to lag.
    disagree_cols = [
        "D_mad", "D_gap", "D_var", "D_std", "D_iqr",
        "D_mad_7d", "D_gap_7d", "D_var_7d", "D_std_7d", "D_iqr_7d",
    ]


    # Trend features to lag.
    trend_cols = ["SVI", "d_SVI", "SVI_7d"]

    # Build lag(1) columns for all predictors.
    lag_cols = level_cols + disagree_cols + trend_cols

    df = df.with_columns(
        [pl.col(c).shift(1).alias(f"{c}_lag1") for c in lag_cols] +
        [
            pl.col(target_col).shift(1).alias(f"{target_col}_lag1"),
            pl.col(target_col).alias("y"),
        ]
    )

    # Drop rows with any missing candidate features for fair comparisons.
    required = ["y", f"{target_col}_lag1"] + [f"{c}_lag1" for c in lag_cols]
    df = df.drop_nulls(subset=required)

    return df

def prepare_google_trends(df: pl.DataFrame,
                          date_col: str = "date",
                          value_col: str = "gt") -> pl.DataFrame:
    # Normalize Google Trends to daily date + 7-day rolling mean.
    out = (
        df.select([
            pl.col(date_col).cast(pl.Date).alias("date"),
            pl.col(value_col).cast(pl.Float64).alias("gt"),
        ])
        .sort("date")
        .with_columns(
            pl.col("gt").rolling_mean(window_size=7, min_samples=1).alias("gt_7d")
        )
    )
    return out


def rq5_models():
    # Define model specs for RQ5 comparisons.
    return {
        # price-only baselines
        "Baseline_const": [],
        "Price_AR1": ["absret_daily_lag1"],

        # add each sentiment family
        "AR1_plus_Twitter": ["absret_daily_lag1", "twitter_mean_lag1"],
        "AR1_plus_Reddit":  ["absret_daily_lag1", "reddit_mean_lag1"],
        "AR1_plus_Trends":  ["absret_daily_lag1", "SVI", "d_SVI", "SVI_7d"],

        # disagreement-only (choose a small set to keep interpretation clean)
        "AR1_plus_D_GAP_D_MAD_7D_D_GAP_7D": [
            "absret_daily_lag1",
            "D_gap_lag1",
            "D_mad_7d",
            "D_gap_7d"
        ],

        "AR1_plus_D_STD_D_IQR_D_MAD_7D_D_GAP_7D": [
            "absret_daily_lag1",
            "D_std_lag1",
            "D_iqr_lag1",
            "D_mad_7d",
            "D_gap_7d"
        ],

        "AR1_plus_D_VAR_D_STD_D_IQR_D_MAD_7D_D_GAP_7D": [
            "absret_daily_lag1",
            "D_var_lag1",
            "D_std_lag1",
            "D_iqr_lag1",
            "D_mad_7d",
            "D_gap_7d"
        ],


        # everything
        "AR1_plus_ALL": [
            "absret_daily_lag1",
            "twitter_mean_lag1",
            "reddit_mean_lag1",
            "SVI",
            "d_SVI",
            "SVI_7d",
            "D_var_lag1",
            "D_std_lag1",
            "D_gap_lag1",
            "D_iqr_lag1",
            "D_mad_7d",
            "D_gap_7d"
        ],
    }

def training_model_rq5(training_df, validation_df):
    # Fit each model spec and compute validation metrics.
    MODELS = rq5_models()
    results = {}
    metrics = []

    for name, x_cols in MODELS.items():
        out = RQ4.fit_predict(training_df, validation_df, x_cols, y_col="y", log_y=False, hac_lags=7)
        results[name] = out

        # Same metrics style as RQ4.
        metrics.append({
            "model": name,
            "rmse": RQ4.rmse(out["y_validation_raw"], out["predit_validation_raw"]),
            "mae":  RQ4.mae(out["y_validation_raw"], out["predit_validation_raw"]),
        })

    return metrics, results

def report_best_models_rq5(metrics):
    # Pick the best model by RMSE and MAE.
    best_rmse = min(metrics, key=lambda d: d["rmse"])
    best_mae  = min(metrics, key=lambda d: d["mae"])
    print("Best RMSE:", best_rmse)
    print("Best MAE :", best_mae)
    return {"rmse": best_rmse["model"], "mae": best_mae["model"]}

def model_prepare(tweets_1to5, reddit_1to5, btc_1to5, gt_1to5):
    # Build sentiment + disagreement features.
    sent_1to5 = RQ4.sentiment_disagreement_daily(reddit_1to5, tweets_1to5)

    # Build BTC daily metrics.
    btc_daily_1to5 = RQ4.btc_daily_instability_from_1m(btc_1to5, require_full_day=True)

    # Build modeling DF for RQ5.
    model_1to5_ret = build_daily_return_model_df(
        btc_daily_1to5, sent_1to5, trends_daily=gt_1to5, target_col="absret_daily"
    )

    # Slice train/val.
    train_start, _     = RQ4.part_bounds(1)
    _, train_end       = RQ4.part_bounds(4)
    val_start, val_end = RQ4.part_bounds(5)

    ret_train_df = RQ4.keep_full_days_inside(model_1to5_ret, train_start, train_end)
    ret_val_df   = RQ4.keep_full_days_inside(model_1to5_ret, val_start, val_end)
    return ret_train_df, ret_val_df


def apply_common_style(title: str, xlabel: str = "Date", ylabel: str = ""):
    # Shared style for matplotlib plots.
    plt.title(title, fontsize=10)
    plt.xlabel(xlabel, fontsize=8)
    plt.ylabel(ylabel, fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.gcf().autofmt_xdate()
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

def plot_all_model_predictions(
    val_df,
    results: dict,
    y_key: str = "y_validation_raw",
    pred_key: str = "predit_validation_raw",
    date_col: str = "date",
    title: str = "RQ5: Validation predictions vs actual",
):
    # Extract dates and model names.
    dates = val_df.select(date_col).to_series().to_list()

    model_names = list(results.keys())

    # Use first model for actual series.
    y = np.asarray(results[model_names[0]][y_key], dtype=float)

    # Daily returns (actual vs predictions).
    plt.figure(figsize=(18, 6))
    plt.plot(dates, y, linewidth=2, alpha=0.8, label="Actual")

    for m in model_names:
        if m == "Price_AR1":
            p = np.asarray(results[m][pred_key], dtype=float)
            plt.plot(dates, p, linewidth=2, alpha=0.8, label=m)
        else:
            p = np.asarray(results[m][pred_key], dtype=float)
            plt.plot(dates, p, alpha=0.5, label=m)
    apply_common_style(title, xlabel="Date", ylabel="Daily return (absret_daily)")


    # Cumulative returns (actual vs predictions).
    plt.figure(figsize=(18, 6))
    cum_y = np.cumsum(y)
    plt.plot(dates, cum_y, linewidth=2, alpha=0.8, label="Actual (cum)")

    for m in model_names:
        if m == "Price_AR1":
            p = np.asarray(results[m][pred_key], dtype=float)
            plt.plot(dates, np.cumsum(p), label=f"{m} (cum)", alpha=0.8, linewidth=2)
        else:
            p = np.asarray(results[m][pred_key], dtype=float)
            plt.plot(dates, np.cumsum(p), label=f"{m} (cum)", alpha=0.5)

    apply_common_style(title=title + " — cumulative", xlabel="Date", ylabel="Cumulative log return")


def load_google(path):
    # Load and normalize Google Trends data for daily joins.
    google_1to5 = data.load_google_search(path)
    google_1to5 = google_1to5.rename({"Scale_['bitcoin']": "SVI"})

    # Standardize date field.
    google_1to5 = google_1to5.with_columns(
        pl.col("date").cast(pl.Datetime)
    )

    # Aggregate to daily SVI and compute deltas/rolling mean.
    google_daily = (
        google_1to5
        .with_columns(pl.col("date").dt.truncate("1d").alias("date"))
        .group_by("date")
        .agg(pl.col("SVI").mean().alias("SVI"))
        .sort("date")
        .with_columns([
            (pl.col("SVI") - pl.col("SVI").shift(1)).alias("d_SVI"),
            pl.col("SVI").rolling_mean(window_size=7, min_samples=1).alias("SVI_7d"),
            pl.col("date").cast(pl.Date),
        ])
    )

    return google_daily.select(["date", "SVI", "d_SVI", "SVI_7d"])


if __name__ == "__main__":
    model_prepare()

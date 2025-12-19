import polars as pl
# import helper.RQ4 as RQ4
# import helper.data as data
import RQ4
import data
import matplotlib.pyplot as plt
import numpy as np



def build_daily_return_model_df(
    btc_daily: pl.DataFrame,
    sent_daily: pl.DataFrame,
    trends_daily: pl.DataFrame | None = None,
    target_col: str = "absret_daily",
) -> pl.DataFrame:
    """
    RQ5 table:
      y_t = return(t)
      X_t includes lagged (t-1) price-only + sentiment + trends + disagreement
    """

    df = btc_daily.join(sent_daily, on="date", how="left").sort("date")

    df = df.join(trends_daily, on="date", how="left").sort("date")

    # ---- choose the feature columns you want to lag ----
    level_cols = ["twitter_mean", "reddit_mean"]  # from sentiment_disagreement_daily output

    disagree_cols = [
        "D_mad", "D_gap", "D_var", "D_std", "D_iqr",
        "D_mad_7d", "D_gap_7d", "D_var_7d", "D_std_7d", "D_iqr_7d",
    ]


    trend_cols = ["SVI", "d_SVI", "SVI_7d"]

    # ---- build lag(1) columns ----
    lag_cols = level_cols + disagree_cols + trend_cols

    df = df.with_columns(
        [pl.col(c).shift(1).alias(f"{c}_lag1") for c in lag_cols] +
        [
            pl.col(target_col).shift(1).alias(f"{target_col}_lag1"),
            pl.col(target_col).alias("y"),
        ]
    )

    # IMPORTANT: for fair model comparison, drop rows where ANY candidate feature is missing
    required = ["y", f"{target_col}_lag1"] + [f"{c}_lag1" for c in lag_cols]
    df = df.drop_nulls(subset=required)

    return df

def prepare_google_trends(df: pl.DataFrame,
                          date_col: str = "date",
                          value_col: str = "gt") -> pl.DataFrame:
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
    MODELS = rq5_models()
    results = {}
    metrics = []

    for name, x_cols in MODELS.items():
        out = RQ4.fit_predict(training_df, validation_df, x_cols, y_col="y", log_y=False, hac_lags=7)
        results[name] = out

        # same metrics style as RQ4
        metrics.append({
            "model": name,
            "rmse": RQ4.rmse(out["y_validation_raw"], out["predit_validation_raw"]),
            "mae":  RQ4.mae(out["y_validation_raw"], out["predit_validation_raw"]),
        })

    return metrics, results

def report_best_models_rq5(metrics):
    best_rmse = min(metrics, key=lambda d: d["rmse"])
    best_mae  = min(metrics, key=lambda d: d["mae"])
    print("Best RMSE:", best_rmse)
    print("Best MAE :", best_mae)
    return {"rmse": best_rmse["model"], "mae": best_mae["model"]}

def training_rq5():
    # 1) Load/concat like RQ4 (1-4 + part5)
    tweets_1to4 = data.load_data_sentiment("data/tweets_training.csv")
    tweets_5    = data.load_data_sentiment("data/sentiment/tweets_5_sent.csv")
    reddit_1to4 = data.load_data_sentiment("data/reddit_training.csv")
    reddit_5    = data.load_data_sentiment("data/sentiment/reddit_5_sent.csv")

    btc_1to4 = data.load_btc("data/btc_training.csv")
    btc_5    = data.load_btc("data/btc_5.csv")

    tweets_1to5 = pl.concat([tweets_1to4, tweets_5]).sort("timestamp")
    reddit_1to5 = pl.concat([reddit_1to4, reddit_5]).sort("timestamp")
    btc_1to5    = pl.concat([btc_1to4, btc_5]).sort("timestamp")

    # 2) Sentiment + disagreement (already includes twitter_mean/reddit_mean)
    sent_1to5 = RQ4.sentiment_disagreement_daily(reddit_1to5, tweets_1to5)

    # 3) Google trends (example)
    # gt_raw = pl.read_csv("data/google_trends_1to5.csv")  # adapt path
    # gt_1to5 = prepare_google_trends(gt_raw, date_col="date", value_col="value")

    gt_1to5 = load_google()
    # 4) BTC daily (has absret_daily)
    btc_daily_1to5 = RQ4.btc_daily_instability_from_1m(btc_1to5, require_full_day=True)

    # 5) Build modeling DF for RQ5
    model_1to5_ret = build_daily_return_model_df(
        btc_daily_1to5, sent_1to5, trends_daily=gt_1to5, target_col="absret_daily"
    )

    # 6) Slice train/val
    train_start, _     = RQ4.part_bounds(1)
    _, train_end       = RQ4.part_bounds(4)
    val_start, val_end = RQ4.part_bounds(5)

    ret_train_df = RQ4.keep_full_days_inside(model_1to5_ret, train_start, train_end)
    ret_val_df   = RQ4.keep_full_days_inside(model_1to5_ret, val_start, val_end)

    # 7) Train + select best
    metrics, results = training_model_rq5(ret_train_df, ret_val_df)
    best = report_best_models_rq5(metrics)
    plot_all_model_predictions(ret_val_df, results, max_models=10)

    return metrics, results, best


def apply_common_style(title: str, xlabel: str = "Date", ylabel: str = ""):
    # General Style for Matplotlib
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
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    max_models: int | None = 12,
    sort_by_rmse: bool = True,
    title: str = "RQ5: Validation predictions vs actual",
):
    """
    val_df: Polars DF used as validation set (must have date_col)
    results: dict from your training_model_rq5, mapping model_name -> out (from fit_predict)
    """

    # Extract dates
    dates = val_df.select(date_col).to_series().to_list()

    # Build model list
    model_names = list(results.keys())
    if include is not None:
        model_names = [m for m in model_names if m in include]
    if exclude is not None:
        model_names = [m for m in model_names if m not in exclude]

    # Optionally sort models by RMSE on validation
    if sort_by_rmse:
        def _rmse(m):
            y = np.asarray(results[m][y_key], dtype=float)
            p = np.asarray(results[m][pred_key], dtype=float)
            return float(np.sqrt(np.mean((y - p) ** 2)))
        model_names.sort(key=_rmse)

    if max_models is not None:
        model_names = model_names[:max_models]

    # Use "actual" from the first model (they should all share the same y vector)
    if not model_names:
        raise ValueError("No models to plot (empty after include/exclude).")

    y = np.asarray(results[model_names[0]][y_key], dtype=float)

    # ---- Plot 1: daily returns (actual vs predictions) ----
    plt.figure(figsize=(18, 4))
    plt.plot(dates, y, linewidth=2, alpha=0.8, label="Actual")

    for m in model_names:
        if m == "Price_AR1":
            p = np.asarray(results[m][pred_key], dtype=float)
            plt.plot(dates, p, linewidth=2, alpha=0.8, label=m)
        else:
            p = np.asarray(results[m][pred_key], dtype=float)
            plt.plot(dates, p, alpha=0.5, label=m)
    apply_common_style(title, xlabel="Date", ylabel="Daily return (absret_daily)")
    # plt.title(title)
    # plt.xlabel()
    # plt.ylabel()
    # plt.xticks(rotation=45)
    # plt.legend(loc="best", fontsize="small")
    # plt.tight_layout()
    # plt.show()

    # ---- Plot 2: cumulative returns (actual vs predictions) ----
    plt.figure(figsize=(18, 4))
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
    # plt.title(title + " — cumulative")
    # plt.xlabel("Date")
    # plt.ylabel("Cumulative log return")
    # plt.xticks(rotation=45)
    # plt.legend(loc="best", fontsize="small")
    # plt.tight_layout()
    # plt.show()



def test():
    part_1 = data.load_google_search("data/part_1.csv")
    part_1 = part_1.rename({"date": "timestamp"})
    part_1 = part_1.rename({"Scale_['bitcoin']": "SVI"})

def load_google():
    google_1to5 = data.load_google_search("data/google_1to5.csv")
    google_1to5 = google_1to5.rename({"Scale_['bitcoin']": "SVI"})

    google_1to5 = google_1to5.with_columns(
        pl.col("date").cast(pl.Datetime)
    )

    google_daily = (
        google_1to5
        .with_columns(pl.col("date").dt.truncate("1d").alias("date"))
        .group_by("date")
        .agg(pl.col("SVI").mean().alias("SVI"))
        .sort("date")
        .with_columns([
            (pl.col("SVI") - pl.col("SVI").shift(1)).alias("d_SVI"),
            pl.col("SVI").rolling_mean(window_size=7, min_samples=1).alias("SVI_7d"),
            pl.col("date").cast(pl.Date),  # <<< THIS LINE FIXES YOUR JOIN
        ])
    )

    return google_daily.select(["date", "SVI", "d_SVI", "SVI_7d"])





if __name__ == "__main__":
    # load_google()
    training_rq5()
    # training_rq5_weekly()
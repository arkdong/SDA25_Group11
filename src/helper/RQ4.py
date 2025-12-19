
import helper.data as data
import helper.RQ4_EDA as EDA

import polars as pl
import pandas as pd
import math
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import statsmodels.api as sm
import seaborn as sns



# Normalize sentiment data to date/score columns.
def prepare(df: pl.DataFrame):
    # Keep timestamp + sentiment and derive daily date/score columns.
    return (
        df.select(["timestamp", "sentiment_score"])
             .drop_nulls("sentiment_score")
             .with_columns(
                 pl.col("timestamp").dt.date().alias("date"),
                 pl.col("sentiment_score").alias("score"),
         )
    )


# Compute per-day sentiment summary stats for a platform.
def daily_stats(df: pl.DataFrame, prefix: str) -> pl.DataFrame:
    # Base daily scores with nulls removed.
    base = df.select(["date", "score"]).drop_nulls(["score"])

    # Daily median and absolute deviation.
    df2 = base.with_columns(pl.col("score").median().over("date").alias("med"))
    df2 = df2.with_columns((pl.col("score") - pl.col("med")).abs().alias("abs_dev"))

    # Quartiles for IQR.
    q75 = pl.col("score").quantile(0.75, interpolation="nearest")
    q25 = pl.col("score").quantile(0.25, interpolation="nearest")

    # Aggregate daily summary metrics.
    return (
        df2.group_by("date")
           .agg([
               pl.col("score").mean().alias(f"{prefix}mean"),
               pl.col("score").median().alias(f"{prefix}median"),
               pl.col("abs_dev").median().alias(f"{prefix}mad"),
               pl.col("score").var().alias(f"{prefix}var"),
               pl.col("score").std().alias(f"{prefix}std"),
               (q75 - q25).alias(f"{prefix}iqr"),
               pl.len().alias(f"{prefix}n"),
           ])
           .sort("date")
    )


# Join platform stats and add absolute mean-gap column.
def mean_gap(tw: pl.DataFrame, rd: pl.DataFrame) -> pl.DataFrame:
    # Outer-join on date and add absolute mean gap.
    return (tw.join(rd, on="date", how="full")
              .with_columns((pl.col("twitter_mean") - pl.col("reddit_mean")).abs().alias("D_gap"))
              .sort("date")
    )


# Compute daily dispersion metrics over all posts combined.
def dispersion_all(tw: pl.DataFrame, rd: pl.DataFrame) -> pl.DataFrame:
    # Combine scores across platforms per day.
    all_scores = pl.concat(
        [tw.select(["date", "score"]), rd.select(["date", "score"])],
        how="vertical"
    ).drop_nulls(["score"])

    # Daily median and absolute deviations.
    df2 = all_scores.with_columns(pl.col("score").median().over("date").alias("med"))
    df2 = df2.with_columns((pl.col("score") - pl.col("med")).abs().alias("abs_dev"))

    # Quartiles for IQR.
    q75 = pl.col("score").quantile(0.75, interpolation="nearest")
    q25 = pl.col("score").quantile(0.25, interpolation="nearest")

    # Aggregate dispersion metrics for all posts.
    return (
        df2.group_by("date")
           .agg([
               pl.col("abs_dev").median().alias("D_mad"),
               pl.col("score").var().alias("D_var"),
               pl.col("score").std().alias("D_std"),
               (q75 - q25).alias("D_iqr"),
               pl.col("score").mean().alias("all_mean"),
               pl.len().alias("all_n"),
           ])
           .sort("date")
    )


# Add 7-day rolling means for disagreement metrics.
def add_weekly_rolling_disagreement(df: pl.DataFrame) -> pl.DataFrame:
    # Ensure date order for rolling windows.
    df = df.sort("date")

    disagreement_cols = ["D_mad", "D_var", "D_std", "D_iqr", "D_gap"]

    # Add rolling 7-day means for each disagreement feature.
    return df.with_columns(
        [
            pl.col(col)
              .rolling_mean(window_size=7, min_samples=1)
              .alias(f"{col}_7d")
            for col in disagreement_cols
        ]
    )


# Build daily sentiment disagreement features across platforms.
def sentiment_disagreement_daily(reddit_df: pl.DataFrame, tweets_df: pl.DataFrame) -> pl.DataFrame:

    # Normalize inputs to date/score.
    rd = prepare(reddit_df)
    tw = prepare(tweets_df)

    # Per-platform daily summaries.
    rd_stats = daily_stats(rd, "reddit_")
    tw_stats = daily_stats(tw, "twitter_")

    # Mean gap and combined dispersion.
    gap_df = mean_gap(tw=tw_stats, rd=rd_stats)

    disp_df = dispersion_all(tw=tw, rd=rd)

    # Merge and add rolling features.
    temp = gap_df.join(disp_df, on="date", how="left").sort("date")
    return add_weekly_rolling_disagreement(temp)


# Aggregate 1-minute BTC data into daily instability measures.
def btc_daily_instability_from_1m(
    btc_df: pl.DataFrame,
    ts_col: str = "timestamp",
    require_full_day: bool = True,
    full_day_bars: int = 1440,
) -> pl.DataFrame:

    # Add date and compute intraday log returns.
    btc = (btc_df.with_columns(pl.col(ts_col).dt.date().alias("date")).sort(ts_col))

    btc = btc.with_columns(pl.col("close").log().alias("log_close"))
    btc = btc.with_columns(
        (pl.col("log_close") - pl.col("log_close").shift(1).over("date")).alias("logret_intra")
    )

    # Prepare variance components for daily aggregation.
    btc = btc.with_columns(pl.col("logret_intra").fill_null(0.0).pow(2).alias("logret2"))
    LN2 = float(math.log(2.0))
    SQRT_365 = float(math.sqrt(365.0))
    daily = (
        btc.group_by("date")
           .agg([
               pl.len().alias("n_bars"),
               pl.col("open").sort_by(ts_col).first().alias("open_d"),
               pl.col("high").max().alias("high_d"),
               pl.col("low").min().alias("low_d"),
               pl.col("close").sort_by(ts_col).last().alias("close_d"),
               pl.col("volume").sum().alias("volume_d"),
               pl.col("logret2").sum().alias("rv_var"),
               pl.col("logret_intra").is_not_null().sum().alias("n_returns"),
           ])
           .sort("date")
           .with_columns([
               pl.col("rv_var").sqrt().alias("rv"),
               ((pl.col("high_d") / pl.col("low_d")).log().pow(2) / (4.0 * LN2)).sqrt().alias("parkinson"),
           ])
    )

    # Daily close-to-close log returns.
    daily = daily.with_columns([
        (pl.col("close_d").log() - pl.col("close_d").log().shift(1)).alias("logret_daily"),
        (pl.col("close_d").log() - pl.col("close_d").log().shift(1)).abs().alias("absret_daily"),
    ])

    # Optionally keep only full 1440-bar days.
    if require_full_day:
        daily = daily.filter(pl.col("n_bars") == full_day_bars)

    return daily


# Assemble daily modeling table with lagged disagreement features.
def build_daily_model_df(
    btc_daily: pl.DataFrame,
    sent_daily: pl.DataFrame,
    target_col: str = "rv",
) -> pl.DataFrame:
    # Join BTC and sentiment and build lagged predictors.
    df = (
        btc_daily.join(sent_daily, on="date", how="left")
                 .sort("date")
                 .with_columns([
                     pl.col("D_mad").shift(1).alias("D_mad_lag1"),
                     pl.col("D_gap").shift(1).alias("D_gap_lag1"),
                     pl.col("D_var").shift(1).alias("D_var_lag1"),
                     pl.col("D_std").shift(1).alias("D_std_lag1"),
                     pl.col("D_iqr").shift(1).alias("D_iqr_lag1"),
                     pl.col(target_col).shift(1).alias(f"{target_col}_lag1"),
                 ])
                 .with_columns(pl.col(target_col).alias("y"))
                 .drop_nulls(subset=["y", "D_mad_lag1", "D_gap_lag1", f"{target_col}_lag1"])
    )
    return df


# Split 2018 into six equal date ranges.
def part_bounds(part: int):
    # Split 2018 into six equal slices and return bounds.
    year_start = datetime(2018, 1, 1)
    year_end   = datetime(2019, 1, 1)
    step = (year_end - year_start) / 6
    slice_start = year_start + step * (part - 1)
    slice_end   = year_start + step * part
    return slice_start, slice_end


# Keep only days fully contained in the slice window.
def keep_full_days_inside(df_daily: pl.DataFrame, slice_start, slice_end) -> pl.DataFrame:
    # Filter days whose full 24h window is inside the slice.
    return (
        df_daily
        .with_columns([
            pl.col("date").cast(pl.Datetime).alias("day_start"),
            (pl.col("date").cast(pl.Datetime) + pl.duration(days=1)).alias("day_end"),
        ])
        .filter((pl.col("day_start") >= slice_start) & (pl.col("day_end") <= slice_end))
        .drop(["day_start", "day_end"])
        .sort("date")
    )


# Build the full model-spec dictionary for regression runs.
def model_combination(ar_lag_col: str):

    # Candidate disagreement feature set.
    DISAGREE_FEATURES = [
        "D_mad_lag1",
        "D_gap_lag1",
        "D_var_lag1",
        "D_std_lag1",
        "D_iqr_lag1",
        "D_mad_7d",
        "D_gap_7d",
        "D_var_7d",
        "D_std_7d",
        "D_iqr_7d",
    ]

    MODELS = {
        "Baseline_const": [],
    }

    # Single-feature models.
    for f in DISAGREE_FEATURES:
        MODELS[f"D_only_{f.replace('_lag1','').upper()}"] = [f]

    # All multi-feature combos.
    import itertools
    for k in range(2, len(DISAGREE_FEATURES) + 1):
        for combo in itertools.combinations(DISAGREE_FEATURES, k):
            name = "D_" + "_".join([c.replace("_lag1","").upper() for c in combo])
            MODELS[name] = list(combo)

    # AR-only baseline.
    MODELS["Baseline_AR1"] = [ar_lag_col]

    # AR plus single feature.
    for f in DISAGREE_FEATURES:
        MODELS[f"AR1_plus_{f.replace('_lag1','').upper()}"] = [ar_lag_col, f]

    # AR plus multi-feature combos.
    for k in range(2, len(DISAGREE_FEATURES) + 1):
        for combo in itertools.combinations(DISAGREE_FEATURES, k):
            base_name = "_".join([c.replace("_lag1","").upper() for c in combo])
            MODELS[f"AR1_plus_{base_name}"] = [ar_lag_col] + list(combo)

    return MODELS


# Run selected EDA plots for sentiment and BTC data.
def plot_EDA():
    # Load training data and build daily frames for plotting.
    df_tweets = data.load_data_sentiment("data/tweets_training.csv")
    df_reddit = data.load_data_sentiment("data/reddit_training.csv")
    sent_train = sentiment_disagreement_daily(df_tweets, df_reddit)
    df_btc = data.load_btc("data/btc_training.csv")
    btc_train  = btc_daily_instability_from_1m(df_btc, require_full_day=True)
    model_train_rv = build_daily_model_df(btc_train, sent_train, target_col="rv")

    # Plot selected EDA figure(s).
    EDA.plot_btc_daily_instability(btc_train)


# Root-mean-squared error helper.
def rmse(y, yhat):
    # Root-mean-squared error.
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


# Mean absolute error helper.
def mae(y, yhat):
    # Mean absolute error.
    return float(np.mean(np.abs(y - yhat)))


# Build numpy design matrix with intercept.
def design_matrix(df_pl, x_cols):
    # Convert selected columns to numpy and add intercept.
    n = df_pl.height
    if len(x_cols) == 0:
        return np.ones((n, 1)), ["const"]
    X = df_pl.select(x_cols).to_numpy()
    X = sm.add_constant(X, has_constant="add")
    return X, ["const"] + x_cols


# Fit OLS on train and predict on validation, with optional log scale.
def fit_predict(training_df, validation_df, x_cols, y_col="y", log_y=True, hac_lags=7):
    # Extract response arrays and apply log if requested.
    y_training_raw = training_df[y_col].to_numpy()
    y_validation_raw = validation_df[y_col].to_numpy()

    if log_y:
        y_training = np.log(y_training_raw + 1e-12)
        y_validation = np.log(y_validation_raw + 1e-12)
    else:
        y_training = y_training_raw
        y_validation = y_validation_raw

    # Fit OLS and predict on validation.
    X_training, names = design_matrix(training_df, x_cols)
    X_validation, _ = design_matrix(validation_df, x_cols)

    res = sm.OLS(y_training, X_training).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    predit_validation = res.predict(X_validation)

    # Convert predictions back to raw scale.
    if log_y:
        predite_validation_raw = np.exp(predit_validation) - 1e-12
    else:
        predite_validation_raw = predit_validation

    return {
        "res": res,
        "names": names,
        "x_cols": x_cols,
        "y_validation": y_validation,
        "predit_validation": predit_validation,
        "y_validation_raw": y_validation_raw,
        "predit_validation_raw": predite_validation_raw,
    }


# Train a suite of models and collect validation metrics.
def training_model(training_df, validation_df, ar_lag_col):
    # Fit each model spec and track validation metrics.
    MODELS = model_combination(ar_lag_col)
    results = {}
    metrics = []

    for name, x_cols in MODELS.items():
        out = fit_predict(training_df, validation_df, x_cols, y_col="y", log_y=True, hac_lags=7)
        results[name] = out
        metrics.append({
            "model": name,
            "rmse_log": rmse(out["y_validation"], out["predit_validation"]),
            "mae_log": mae(out["y_validation"], out["predit_validation"]),
            "rmse_raw": rmse(out["y_validation_raw"], out["predit_validation_raw"]),
            "mae_raw": mae(out["y_validation_raw"], out["predit_validation_raw"]),
        })
    return metrics, results


# Print and return best model names by metric.
def report_best_models(metrics):
    # Compare metrics and report best model per criterion.
    criteria = {
        "rmse_raw": "RMSE (raw scale)",
        "rmse_log": "RMSE (log scale)",
        "mae_raw":  "MAE  (raw scale)",
        "mae_log":  "MAE  (log scale)",
    }

    best_models = {}

    print("\n=== Best Models on Validation Set ===")
    for key, label in criteria.items():
        best = min(metrics, key=lambda d: d[key])
        best_models[key] = best["model"]
        print(f"{label:20} → {best['model']}  (value = {best[key]:.6f})")
    print("=====================================\n")

    return best_models


# Prepare train/validation slices for a target metric.
def training_process(reddit_1to5, tweets_1to5, btc_1to5, target):
    # Build daily sentiment and BTC frames.
    sent_1to5 = sentiment_disagreement_daily(reddit_1to5, tweets_1to5)

    btc_daily_1to5 = btc_daily_instability_from_1m(btc_1to5, require_full_day=True)

    # Create modeling table and slice into train/val.
    model_1to5_rv = build_daily_model_df(btc_daily_1to5, sent_1to5, target_col=target)
    train_start, _        = part_bounds(1)
    _, train_end          = part_bounds(4)
    val_start, val_end    = part_bounds(5)

    train_df = keep_full_days_inside(model_1to5_rv, train_start, train_end)
    val_df   = keep_full_days_inside(model_1to5_rv, val_start, val_end)
    return train_df, val_df


# Format p-values with compact precision.
def fmt_p(p):
    # Compact p-value format for reports.
    return f"{p:.2e}" if p < 1e-4 else f"{p:.4f}"


# Build result tables and display-friendly formatting.
def results_table(*outs, model=None):
    # Assemble results into a DataFrame plus display formatting.
    rows = []
    for o in outs:
        rows.append({
            "model": model,
            "term": o["term"],
            "beta": o["beta"],
            "t": o["t"],
            "p": o["p"],
            "alpha": o["alpha"],
            "reject": "✅" if o["reject_H0"] else "❌",
        })
    df = pd.DataFrame(rows)

    df_disp = df.copy()
    df_disp["beta"] = df_disp["beta"].map(lambda x: f"{x:.4f}")
    df_disp["t"] = df_disp["t"].map(lambda x: f"{x:.3f}")
    df_disp["p"] = df_disp["p"].map(fmt_p)
    df_disp["alpha"] = df_disp["alpha"].map(lambda x: f"{x:.2f}")

    return df, df_disp


# Run a coefficient hypothesis test and return stats.
def coef_test(res, names, term, alpha=0.05, alternative="greater"):
    # Compute one- or two-sided p-value and decision.
    idx = names.index(term)
    beta = float(res.params[idx])
    t = float(res.tvalues[idx])
    p2 = float(res.pvalues[idx])

    if alternative == "two-sided":
        p = p2
    elif alternative == "greater":
        p = (p2 / 2.0) if beta > 0 else (1.0 - p2 / 2.0)
        reject = p < alpha
    elif alternative == "less":
        p = (p2 / 2.0) if beta < 0 else (1.0 - p2 / 2.0)
        reject = p < alpha
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    return {"term": term, "beta": beta, "t": t, "p": float(p), "alpha": alpha, "reject_H0": bool(reject)}


# Run hypothesis tests for selected models and plot results.
def hypothesis_test(result):
    # Select models, test key coefficients, and plot distributions.
    chosen1 = result["AR1_plus_D_GAP_D_MAD_7D_D_GAP_7D"]
    out1_1 = coef_test(chosen1["res"], chosen1["names"], "D_mad_7d", alpha=0.05, alternative="greater")
    out1_2 = coef_test(chosen1["res"], chosen1["names"], "D_gap_7d", alpha=0.05, alternative="greater")
    out1_3 = coef_test(chosen1["res"], chosen1["names"], "D_gap_lag1", alpha=0.05, alternative="greater")

    chosen2 = result["AR1_plus_D_STD_D_IQR_D_MAD_7D_D_GAP_7D"]
    out2_1 = coef_test(chosen2["res"], chosen2["names"], "D_std_lag1", alpha=0.05, alternative="greater")
    out2_2 = coef_test(chosen2["res"], chosen2["names"], "D_iqr_lag1", alpha=0.05, alternative="greater")
    out2_3 = coef_test(chosen2["res"], chosen2["names"], "D_mad_7d", alpha=0.05, alternative="greater")
    out2_4 = coef_test(chosen2["res"], chosen2["names"], "D_gap_7d", alpha=0.05, alternative="greater")

    chosen3 = result["AR1_plus_D_VAR_D_STD_D_IQR_D_MAD_7D_D_GAP_7D"]
    out3_1 = coef_test(chosen3["res"], chosen3["names"], "D_var_lag1", alpha=0.05, alternative="greater")
    out3_2 = coef_test(chosen3["res"], chosen3["names"], "D_std_lag1", alpha=0.05, alternative="greater")
    out3_3 = coef_test(chosen3["res"], chosen3["names"], "D_mad_7d", alpha=0.05, alternative="greater")
    out3_4 = coef_test(chosen3["res"], chosen3["names"], "D_gap_7d", alpha=0.05, alternative="greater")
    out3_5 = coef_test(chosen3["res"], chosen3["names"], "D_iqr_lag1", alpha=0.05, alternative="greater")

    df1, df1_disp = results_table(out1_1, out1_2, out1_3, model="AR1_plus_D_GAP_D_MAD_7D_D_GAP_7D")
    df2, df2_disp = results_table(out2_1, out2_2, out2_3, out2_4, model="AR1_plus_D_STD_D_IQR_D_MAD_7D_D_GAP_7D")
    df3, df3_disp = results_table(out3_1, out3_2, out3_3, out3_4, out3_5, model="AR1_plus_D_VAR_D_STD_D_IQR_D_MAD_7D_D_GAP_7D")

    df_all_disp = pd.concat([df1_disp, df2_disp, df3_disp], ignore_index=True)
    print("\nALL RESULTS:\n")
    print(df_all_disp.to_string(index=False))


    EDA.plot_null_test_normal(
    out1_1["t"],
    alpha=0.05,
    alternative="greater",
    title=r"$H_0$: $\beta=0$ vs $H_1$: $\beta>0$ for D_mad_7d (AR1_plus_D_GAP_D_MAD_7D_D_GAP_7D)"
    )

    EDA.plot_null_test_normal(
        out1_2["t"],
        alpha=0.05,
        alternative="greater",
        title=r"$H_0$: $\beta=0$ vs $H_1$: $\beta>0$ for D_gap_7d (AR1_plus_D_GAP_D_MAD_7D_D_GAP_7D)"
    )

if __name__ ==  "__main__":
    tweets_1to4 = data.load_data_sentiment("data/tweets_training.csv")
    tweets_5 = data.load_data_sentiment("data/sentiment/tweets_5_sent.csv")
    reddit_1to4 = data.load_data_sentiment("data/reddit_training.csv")
    reddit_5 = data.load_data_sentiment("data/sentiment/reddit_5_sent.csv")
    btc_1to4 = data.load_btc("data/btc_training.csv")
    btc_5 = data.load_btc("data/btc_5.csv")


    reddit_1to5 = pl.concat([reddit_1to4, reddit_5], how="vertical").sort("timestamp")
    tweets_1to5 = pl.concat([tweets_1to4, tweets_5], how="vertical").sort("timestamp")
    btc_1to5    = pl.concat([btc_1to4, btc_5], how="vertical").sort("timestamp")
    train_df,val_df = training_process(
        reddit_1to5=reddit_1to5,
        tweets_1to5=tweets_1to5,
        btc_1to5=btc_1to5,
        target="rv"
        )

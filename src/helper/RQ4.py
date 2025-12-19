
import helper.data as data
import helper.RQ4_EDA as EDA

# import data
# import RQ4_EDA as EDA

import polars as pl
import pandas as pd
import math
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import statsmodels.api as sm
import seaborn as sns



def prepare(df: pl.DataFrame):
    return (
        df.select(["timestamp", "sentiment_score"])
             .drop_nulls("sentiment_score")
             .with_columns(
                 pl.col("timestamp").dt.date().alias("date"),
                 pl.col("sentiment_score").alias("score"),
         )
    )


def daily_stats(df: pl.DataFrame, prefix: str) -> pl.DataFrame:
    """
    Return per-day stats for one platform.
    Output columns:
      ['date', 'reddit_mean', 'reddit_median', 'reddit_mad', 'reddit_var',
       'reddit_std', 'reddit_iqr', 'reddit_n']
    """
    base = df.select(["date", "score"]).drop_nulls(["score"])

    df2 = base.with_columns(pl.col("score").median().over("date").alias("med"))
    df2 = df2.with_columns((pl.col("score") - pl.col("med")).abs().alias("abs_dev"))

    q75 = pl.col("score").quantile(0.75, interpolation="nearest")
    q25 = pl.col("score").quantile(0.25, interpolation="nearest")

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


def mean_gap(tw: pl.DataFrame, rd: pl.DataFrame) -> pl.DataFrame:
    """
    Compute and return Twitter–Reddit mean gap:
      D_gap = |twitter_mean - reddit_mean|
    Keeps all columns from both stats tables (outer join on date).
    """
    return (tw.join(rd, on="date", how="full")
              .with_columns((pl.col("twitter_mean") - pl.col("reddit_mean")).abs().alias("D_gap"))
              .sort("date")
    )


def dispersion_all(tw: pl.DataFrame, rd: pl.DataFrame) -> pl.DataFrame:
    """
    Disagreement across ALL posts (Twitter + Reddit) per day:
      - D_mad, D_var, D_std, D_iqr
      - all_mean, all_n
    """
    all_scores = pl.concat(
        [tw.select(["date", "score"]), rd.select(["date", "score"])],
        how="vertical"
    ).drop_nulls(["score"])

    df2 = all_scores.with_columns(pl.col("score").median().over("date").alias("med"))
    df2 = df2.with_columns((pl.col("score") - pl.col("med")).abs().alias("abs_dev"))

    q75 = pl.col("score").quantile(0.75, interpolation="nearest")
    q25 = pl.col("score").quantile(0.25, interpolation="nearest")

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


def add_weekly_rolling_disagreement(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add 7-day rolling disagreement features:
      D_mad_7d, D_var_7d, D_std_7d, D_iqr_7d, D_gap_7d
    using a rolling mean over the last 7 days (including today).
    """
    df = df.sort("date")

    disagreement_cols = ["D_mad", "D_var", "D_std", "D_iqr", "D_gap"]

    return df.with_columns(
        [
            pl.col(col)
              .rolling_mean(window_size=7, min_samples=1)
              .alias(f"{col}_7d")
            for col in disagreement_cols
        ]
    )


def sentiment_disagreement_daily(reddit_df: pl.DataFrame, tweets_df: pl.DataFrame) -> pl.DataFrame:
    """
    Returns per-day sentiment features including:
      - Platform stats (mean/median/mad/var/std/iqr/n)
      - D_gap (abs mean gap)
      - All-post dispersion: D_mad, D_var, D_std, D_iqr
    """

    rd = prepare(reddit_df)
    tw = prepare(tweets_df)

    rd_stats = daily_stats(rd, "reddit_")
    tw_stats = daily_stats(tw, "twitter_")

    # Twitter-Reddit mean gap (absolute)
    gap_df = mean_gap(tw=tw_stats, rd=rd_stats)

    # Combined MAD and Disagreement across ALL posts (Twitter + Reddit)
    disp_df = dispersion_all(tw=tw, rd=rd)

    temp = gap_df.join(disp_df, on="date", how="left").sort("date")
    return add_weekly_rolling_disagreement(temp)


def btc_daily_instability_from_1m(
    btc_df: pl.DataFrame,
    ts_col: str = "timestamp",
    require_full_day: bool = True,
    full_day_bars: int = 1440,
) -> pl.DataFrame:
    """
    Computes daily price instability from 1-minute OHLCV:
      - rv: realized volatility = sqrt(sum intraday log-return^2)
      - parkinson: range-based daily estimator using daily high/low
      - absret_daily: abs(close-to-close daily log return)

    If require_full_day=True, keeps only days with exactly 1440 bars.
    """

    btc = (btc_df.with_columns(pl.col(ts_col).dt.date().alias("date")).sort(ts_col))

    # Intraday log returns within each day (order matters)
    btc = btc.with_columns(pl.col("close").log().alias("log_close"))
    btc = btc.with_columns(
        (pl.col("log_close") - pl.col("log_close").shift(1).over("date")).alias("logret_intra")
    )

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

    # daily close-to-close log return + abs return
    daily = daily.with_columns([
        (pl.col("close_d").log() - pl.col("close_d").log().shift(1)).alias("logret_daily"),
        (pl.col("close_d").log() - pl.col("close_d").log().shift(1)).abs().alias("absret_daily"),
    ])

    if require_full_day:
        daily = daily.filter(pl.col("n_bars") == full_day_bars)

    return daily


def build_daily_model_df(
    btc_daily: pl.DataFrame,
    sent_daily: pl.DataFrame,
    target_col: str = "rv",   # rv, parkinson, absret_daily
) -> pl.DataFrame:
    """
    Produces modeling table where day t target uses disagreement from day t-1:
      y_t = target(t)
      X_t includes D_mad(t-1), D_gap(t-1), D_var(t-1), D_std(t-1), D_iqr(t-1), rv(t-1)
    """
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


def part_bounds(part: int):
    year_start = datetime(2018, 1, 1)
    year_end   = datetime(2019, 1, 1)
    step = (year_end - year_start) / 6
    slice_start = year_start + step * (part - 1)
    slice_end   = year_start + step * part
    return slice_start, slice_end


def keep_full_days_inside(df_daily: pl.DataFrame, slice_start, slice_end) -> pl.DataFrame:
    """
    Keep ONLY days where the entire [00:00, 24:00) interval is inside [slice_start, slice_end).
    This avoids "split day" leakage when part boundaries cut through a day.
    """
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


def model_combination(ar_lag_col: str):
    """
    Build full model-spec dictionary using the correct AR(1) lag column.
    Example: ar_lag_col='rv_lag1', 'parkinson_lag1', or 'absret_daily_lag1'.
    """

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

    # Build MODEL dictionary
    MODELS = {
        "Baseline_const": [],  # intercept-only model
    }

    # 1) Single disagreement features
    for f in DISAGREE_FEATURES:
        MODELS[f"D_only_{f.replace('_lag1','').upper()}"] = [f]

    # 2) All combinations of disagreement features (size ≥ 2)
    import itertools
    for k in range(2, len(DISAGREE_FEATURES) + 1):
        for combo in itertools.combinations(DISAGREE_FEATURES, k):
            name = "D_" + "_".join([c.replace("_lag1","").upper() for c in combo])
            MODELS[name] = list(combo)

    # 3) Baseline AR model
    MODELS["Baseline_AR1"] = [ar_lag_col]

    # 4) AR(1) + single disagreement feature
    for f in DISAGREE_FEATURES:
        MODELS[f"AR1_plus_{f.replace('_lag1','').upper()}"] = [ar_lag_col, f]

    # 5) AR(1) + all disagreement combos (≥2)
    for k in range(2, len(DISAGREE_FEATURES) + 1):
        for combo in itertools.combinations(DISAGREE_FEATURES, k):
            base_name = "_".join([c.replace("_lag1","").upper() for c in combo])
            MODELS[f"AR1_plus_{base_name}"] = [ar_lag_col] + list(combo)

    return MODELS


def plot_EDA():
    df_tweets = data.load_data_sentiment("data/tweets_training.csv")
    df_reddit = data.load_data_sentiment("data/reddit_training.csv")
    sent_train = sentiment_disagreement_daily(df_tweets, df_reddit)
    df_btc = data.load_btc("data/btc_training.csv")
    btc_train  = btc_daily_instability_from_1m(df_btc, require_full_day=True)
    model_train_rv = build_daily_model_df(btc_train, sent_train, target_col="rv")

    # Sentiment EDA
    # EDA.plot_sentiment_distributions(df_tweets, df_reddit)
    # EDA.plot_daily_means(df_tweets, df_reddit)
    # EDA.plot_boxplots(df_tweets, df_reddit)
    # EDA.plot_scatter(df_tweets, df_reddit)

    # Disagreement EDA
    # EDA.plot_daily_sentiment_disagreement(sent_train, (15, 3))
    # EDA.plot_7d_sentiment_disagreement(sent_train, (15, 3))
    # EDA.plot_disagreement_normalized(sent_train, (15, 3))
    # EDA.plot_7d_disagreement_normalized(sent_train, (15, 3))

    # BTC EDA
    EDA.plot_btc_daily_instability(btc_train)

    # Training Dataset EDA
    # EDA.plot_model_feature_scatter(model_train_rv)
    # EDA.plot_model_correlations(model_train_rv)


def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mae(y, yhat):
    return float(np.mean(np.abs(y - yhat)))


def design_matrix(df_pl, x_cols):
    """Return (X, names) with intercept."""
    n = df_pl.height
    if len(x_cols) == 0:
        return np.ones((n, 1)), ["const"]
    X = df_pl.select(x_cols).to_numpy()
    X = sm.add_constant(X, has_constant="add")
    return X, ["const"] + x_cols


def fit_predict(training_df, validation_df, x_cols, y_col="y", log_y=True, hac_lags=7):
    """Fit OLS on train, predict val. Returns result + arrays in log and raw scale."""
    y_training_raw = training_df[y_col].to_numpy()
    y_validation_raw = validation_df[y_col].to_numpy()

    if log_y:
        y_training = np.log(y_training_raw + 1e-12)
        y_validation = np.log(y_validation_raw + 1e-12)
    else:
        y_training = y_training_raw
        y_validation = y_validation_raw

    X_training, names = design_matrix(training_df, x_cols)
    X_validation, _ = design_matrix(validation_df, x_cols)

    res = sm.OLS(y_training, X_training).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    predit_validation = res.predict(X_validation)

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


def training_model(training_df, validation_df, ar_lag_col):
    # Define your baseline + augmented specs
    MODELS = model_combination(ar_lag_col)
    # Fit all models (train) and evaluate on validation
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


def report_best_models(metrics):
    """
    Print the best models for each evaluation metric and return a dict
    mapping metric-key -> best-model-name.
    """
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


def training_process(reddit_1to5, tweets_1to5, btc_1to5, target):
    # Daily sentiment features (D_mad, D_gap)
    sent_1to5 = sentiment_disagreement_daily(reddit_1to5, tweets_1to5)

    # Daily BTC instability from 1m bars (rv etc.)
    btc_daily_1to5 = btc_daily_instability_from_1m(btc_1to5, require_full_day=True)

    # Join + lag features; y = today's rv
    model_1to5_rv = build_daily_model_df(btc_daily_1to5, sent_1to5, target_col=target)
    train_start, _        = part_bounds(1)
    _, train_end          = part_bounds(4)
    val_start, val_end    = part_bounds(5)

    train_df = keep_full_days_inside(model_1to5_rv, train_start, train_end)
    val_df   = keep_full_days_inside(model_1to5_rv, val_start, val_end)
    return train_df, val_df
    # rv_metrics, rv_result = training_model(rv_train_df, rv_val_df, "rv_lag1")
    # rv_best_models = report_best_models(rv_metrics)

    # EDA.plot_best_models_vs_baseline(rv_val_df, rv_result, rv_best_models, "rv")
    # EDA.plot_all_models_vs_actual(rv_val_df, rv_result, "rv")
    # return rv_result


def fmt_p(p):
    # nice p formatting
    return f"{p:.2e}" if p < 1e-4 else f"{p:.4f}"


def results_table(*outs, model=None):
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

    # formatting
    df_disp = df.copy()
    df_disp["beta"] = df_disp["beta"].map(lambda x: f"{x:.4f}")
    df_disp["t"] = df_disp["t"].map(lambda x: f"{x:.3f}")
    df_disp["p"] = df_disp["p"].map(fmt_p)
    df_disp["alpha"] = df_disp["alpha"].map(lambda x: f"{x:.2f}")

    return df, df_disp


def coef_test(res, names, term, alpha=0.05, alternative="greater"):
    """
    alternative:
      - "two-sided": H1 beta != 0
      - "greater"  : H1 beta > 0
      - "less"     : H1 beta < 0
    """
    idx = names.index(term)
    beta = float(res.params[idx])
    t = float(res.tvalues[idx])
    p2 = float(res.pvalues[idx])  # two-sided

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


def hypothesis_test(result):
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

    # EDA.plot_null_test_normal(
    #     out1_3["t"],
    #     alpha=0.05,
    #     alternative="greater",
    #     title=r"$H_0$: $\beta=0$ vs $H_1$: $\beta>0$ for D_gap_lag1 (AR1_plus_D_GAP_D_MAD_7D_D_GAP_7D)"
    # )

    # EDA.plot_null_test_normal(
    #     out2_1["t"],
    #     alpha=0.05,
    #     alternative="greater",
    #     title=r"$H_0$: $\beta=0$ vs $H_1$: $\beta>0$ for D_std_lag1 (AR1_plus_D_STD_D_IQR_D_MAD_7D_D_GAP_7D)"
    # )

    # EDA.plot_null_test_normal(
    #     out2_2["t"],
    #     alpha=0.05,
    #     alternative="greater",
    #     title=r"$H_0$: $\beta=0$ vs $H_1$: $\beta>0$ for D_iqr_lag1 (AR1_plus_D_STD_D_IQR_D_MAD_7D_D_GAP_7D)"
    # )

    # EDA.plot_null_test_normal(
    #     out2_3["t"],
    #     alpha=0.05,
    #     alternative="greater",
    #     title=r"$H_0$: $\beta=0$ vs $H_1$: $\beta>0$ for D_mad_7d (AR1_plus_D_STD_D_IQR_D_MAD_7D_D_GAP_7D)"
    # )

    # EDA.plot_null_test_normal(
    #     out2_4["t"],
    #     alpha=0.05,
    #     alternative="greater",
    #     title=r"$H_0$: $\beta=0$ vs $H_1$: $\beta>0$ for D_gap_7d (AR1_plus_D_STD_D_IQR_D_MAD_7D_D_GAP_7D)"
    # )

    # EDA.plot_null_test_normal(
    #     out3_1["t"],
    #     alpha=0.05,
    #     alternative="greater",
    #     title=r"$H_0$: $\beta=0$ vs $H_1$: $\beta>0$ for D_var_lag1 (AR1_plus_D_VAR_D_STD_D_IQR_D_MAD_7D_D_GAP_7D)"
    # )

    # EDA.plot_null_test_normal(
    #     out3_2["t"],
    #     alpha=0.05,
    #     alternative="greater",
    #     title=r"$H_0$: $\beta=0$ vs $H_1$: $\beta>0$ for D_std_lag1 (AR1_plus_D_VAR_D_STD_D_IQR_D_MAD_7D_D_GAP_7D)"
    # )

    # EDA.plot_null_test_normal(
    #     out3_3["t"],
    #     alpha=0.05,
    #     alternative="greater",
    #     title=r"$H_0$: $\beta=0$ vs $H_1$: $\beta>0$ for D_mad_7d (AR1_plus_D_VAR_D_STD_D_IQR_D_MAD_7D_D_GAP_7D)"
    # )

    # EDA.plot_null_test_normal(
    #     out3_4["t"],
    #     alpha=0.05,
    #     alternative="greater",
    #     title=r"$H_0$: $\beta=0$ vs $H_1$: $\beta>0$ for D_gap_7d (AR1_plus_D_VAR_D_STD_D_IQR_D_MAD_7D_D_GAP_7D)"
    # )

    # EDA.plot_null_test_normal(
    #     out3_5["t"],
    #     alpha=0.05,
    #     alternative="greater",
    #     title=r"$H_0$: $\beta=0$ vs $H_1$: $\beta>0$ for D_iqr_lag1 (AR1_plus_D_VAR_D_STD_D_IQR_D_MAD_7D_D_GAP_7D)"
    # )


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
    # plot_EDA()
    # result = training()
    # hypothesis_test(result)
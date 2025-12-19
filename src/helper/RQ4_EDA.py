import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

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

# Print basic sentiment stats and cross-platform correlation.
def print_basic_stats(df_tweets, df_reddit):
    # Prepare daily sentiment frames.
    tw = prepare(df_tweets)
    rd = prepare(df_reddit)

    # Descriptive statistics for both platforms.
    stats = pd.DataFrame({
        "Twitter": tw["score"].to_pandas().describe(),
        "Reddit": rd["score"].to_pandas().describe()
    })

    print("\n===== BASIC SENTIMENT STATISTICS =====\n")
    print(stats)
    print("\nDifference in means:", stats.loc["mean", "Twitter"] - stats.loc["mean", "Reddit"])

    # Daily means to compare trends.
    tw_pd = tw.group_by("date").agg(pl.col("score").mean().alias("mean")).sort("date").to_pandas()
    rd_pd = rd.group_by("date").agg(pl.col("score").mean().alias("mean")).sort("date").to_pandas()

    # Merge dates for correlation.
    merged = tw_pd.merge(rd_pd, on="date", suffixes=("_twitter", "_reddit"))

    # Correlation of daily means across platforms.
    corr = merged["mean_twitter"].corr(merged["mean_reddit"])
    print("\n===== DAILY CORRELATION =====")
    print(f"Correlation between Twitter and Reddit daily sentiment: {corr:.4f}")


# Apply a consistent matplotlib style and layout.
def apply_common_style(title: str, xlabel: str = "Date", ylabel: str = ""):
    # Shared styling for charts.
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

# Plot KDE distributions for raw sentiment scores.
def plot_sentiment_distributions(df_tweets: pl.DataFrame, df_reddit: pl.DataFrame):
    # Convert to pandas for seaborn.
    tw_pd = prepare(df_tweets).to_pandas()
    rd_pd = prepare(df_reddit).to_pandas()
    plt.figure(figsize=(14, 4))
    sns.kdeplot(tw_pd["score"], fill=True, label="Twitter")
    sns.kdeplot(rd_pd["score"], fill=True, label="Reddit")

    apply_common_style(
        "Sentiment Score Distribution (Twitter vs Reddit)",
        xlabel="Sentiment score",
        ylabel="Count",
    )

def plot_daily_means(df_tweets: pl.DataFrame, df_reddit: pl.DataFrame):
    # Compute daily mean sentiment per platform.
    tw_pd = prepare(df_tweets).group_by("date").agg(pl.col("score").mean().alias("mean_score")).sort("date").to_pandas()
    rd_pd = prepare(df_reddit).group_by("date").agg(pl.col("score").mean().alias("mean_score")).sort("date").to_pandas()

    # Plot time series of daily means.
    plt.figure(figsize=(14, 4))
    plt.plot(tw_pd["date"], tw_pd["mean_score"], linewidth=1, alpha=0.8, label="Twitter mean")
    plt.plot(rd_pd["date"], rd_pd["mean_score"], linewidth=1, alpha=0.8, label="Reddit mean")

    apply_common_style(
        "Daily Mean Sentiment (Twitter vs Reddit)",
        xlabel="Date",
        ylabel="Mean sentiment",
    )

# Compare sentiment distributions with boxplots.
def plot_boxplots(df_tweets, df_reddit):
    # Extract score arrays.
    tw = prepare(df_tweets).to_pandas()["score"]
    rd = prepare(df_reddit).to_pandas()["score"]

    data = [tw, rd]

    plt.figure(figsize=(6, 6))

    # Create boxplot.
    bp = plt.boxplot(
        data,
        tick_labels=["Twitter", "Reddit"],
        patch_artist=True,
        showfliers=False,     # Hide extreme outliers for clarity
        boxprops=dict(linewidth=1.5),
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        label=["Twitter", "Reddit"]
    )

    # Color the boxes.
    colors = ["#1DA1F2", "#FF4500"]  # Twitter blue, Reddit orange
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    apply_common_style(
        "Sentiment Score Distribution by Platform",
        xlabel="Date",
        ylabel="Sentiment Score",
    )

# Scatter daily mean sentiment between platforms.
def plot_scatter(df_tweets, df_reddit):
    # Compute daily means.
    tw_pd = prepare(df_tweets).group_by("date").agg(pl.col("score").mean().alias("mean")).sort("date").to_pandas()
    rd_pd = prepare(df_reddit).group_by("date").agg(pl.col("score").mean().alias("mean")).sort("date").to_pandas()

    # Merge dates.
    merged = tw_pd.merge(rd_pd, on="date", suffixes=("_twitter", "_reddit"))

    # Scatter plot.
    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=merged, x="mean_twitter", y="mean_reddit", s=40, alpha=0.8,label="tw and rd")
    apply_common_style(
        "Relationship between Daily Twitter and Reddit Sentiment",
        xlabel="Twitter daily mean",
        ylabel="Reddit daily mean",
    )

# Plot daily disagreement metrics over time.
def plot_daily_sentiment_disagreement(sent_daily: pl.DataFrame, figsize=None):
    # Sort to ensure temporal order.
    df = sent_daily.sort("date").to_pandas()
    if figsize is None:
        figsize = (12,4)
    # Plot daily disagreement series.
    plt.figure(figsize=figsize)
    plt.plot(df["date"], df["D_mad"], linewidth=1, alpha=0.8, label="D_mad")
    plt.plot(df["date"], df["D_gap"], linewidth=1, alpha=0.8, label="D_gap")
    plt.plot(df["date"], df["D_var"], linewidth=1, alpha=0.8, label="D_var")
    plt.plot(df["date"], df["D_std"], linewidth=1, alpha=0.8, label="D_std")
    plt.plot(df["date"], df["D_iqr"], linewidth=1, alpha=0.8, label="D_iqr")

    apply_common_style(
        "Daily Sentiment Disagreement (D_mad, D_gap, D_var, D_std, D_iqr)",
        xlabel="Date",
        ylabel="Value",
    )

# Plot 7-day rolling disagreement metrics over time.
def plot_7d_sentiment_disagreement(sent_daily: pl.DataFrame, figsize=None):
    # Sort to ensure temporal order.
    df = sent_daily.sort("date").to_pandas()
    if figsize is None:
        figsize = (12,4)
    # Plot rolling disagreement series.
    plt.figure(figsize=figsize)
    plt.plot(df["date"], df["D_mad_7d"], linewidth=1, alpha=0.8, label="D_mad_7d")
    plt.plot(df["date"], df["D_gap_7d"], linewidth=1, alpha=0.8, label="D_gap_7d")
    plt.plot(df["date"], df["D_var_7d"], linewidth=1, alpha=0.8, label="D_var_7d")
    plt.plot(df["date"], df["D_std_7d"], linewidth=1, alpha=0.8, label="D_std_7d")
    plt.plot(df["date"], df["D_iqr_7d"], linewidth=1, alpha=0.8, label="D_iqr_7d")

    apply_common_style(
        "7 Day Rolling Sentiment Disagreement (D_mad_7d, D_gap_7d, D_var_7d, D_std_7d, D_iqr_7d)",
        xlabel="Date",
        ylabel="Value",
    )

# Plot z-scored daily disagreement features.
def plot_disagreement_normalized(sent_daily: pl.DataFrame, figsize=None):
    # Compute z-scores for daily metrics.
    df = sent_daily.sort("date").to_pandas()
    features = ["D_mad", "D_gap", "D_var", "D_std", "D_iqr"]
    normalized = (df[features] - df[features].mean()) / df[features].std()

    if figsize is None:
        figsize = (12,4)
    # Plot normalized series.
    plt.figure(figsize=figsize)
    for f in features:
        plt.plot(df["date"], normalized[f], label=f, alpha=0.8)

    apply_common_style(
        "Normalized Daily Sentiment Disagreement (Z-scores)",
        xlabel="Date",
        ylabel="Normalized Value"
    )

# Plot z-scored 7-day rolling disagreement features.
def plot_7d_disagreement_normalized(sent_daily: pl.DataFrame, figsize=None):
    # Compute z-scores for rolling metrics.
    df = sent_daily.sort("date").to_pandas()
    features = ["D_mad_7d", "D_gap_7d", "D_var_7d", "D_std_7d", "D_iqr_7d"]
    normalized = (df[features] - df[features].mean()) / df[features].std()

    if figsize is None:
        figsize = (12,4)
    # Plot normalized series.
    plt.figure(figsize=figsize)
    for f in features:
        plt.plot(df["date"], normalized[f], label=f, alpha=0.8)

    apply_common_style(
        "Normalized 7D Rolling Daily Sentiment Disagreement (Z-scores)",
        xlabel="Date",
        ylabel="Normalized Value"
    )

# Plot raw 1-minute BTC close and volume series.
def plot_raw_btc_1m(df_btc: pl.DataFrame, ts_col="timestamp", close_col="close", vol_col="volume"):
    # Select and clean required columns.
    btc = (
        df_btc.select([ts_col, close_col, vol_col])
              .drop_nulls(close_col)
              .sort(ts_col)
    )

    # Convert to pandas for plotting.
    btc_pd = btc.to_pandas()

    # Price.
    plt.figure(figsize=(14, 6))
    plt.plot(btc_pd[ts_col], btc_pd[close_col], linewidth=1, alpha=0.8, label="BTC Close Price")
    apply_common_style("BTC 1-Minute Close Price", xlabel="Date", ylabel="Price (USD)")

    # Volume.
    plt.figure(figsize=(14, 6))
    plt.plot(btc_pd[ts_col], btc_pd[vol_col], linewidth=1, alpha=0.8, label="BTC Volume")
    apply_common_style("BTC 1-Minute Volume", xlabel="Date", ylabel="Volume")

# Plot daily BTC volatility measures.
def plot_btc_daily_instability(btc_daily: pl.DataFrame):
    # Sort and convert for plotting.
    df = btc_daily.sort("date").to_pandas()
    plt.figure(figsize=(14, 6))
    plt.plot(df["date"], df["rv"], linewidth=1, alpha=0.8, label="BTC RV")
    plt.plot(df["date"], df["parkinson"], linewidth=1, alpha=0.8, label="BTC Parkinson")
    plt.plot(df["date"], df["absret_daily"], linewidth=1, alpha=0.8, label="BTC absret")
    apply_common_style("Daily Price Instability", xlabel="Date", ylabel="|logret|")

# Plot target series and its lag over time.
def plot_model_target_timeseries(model_train: pl.DataFrame, base_target: str = "rv"):
    # Sort and prepare the timeline.
    df = model_train.sort("date").to_pandas()

    plt.figure(figsize=(14, 6))
    plt.plot(df["date"], df["y"], linewidth=1, alpha=0.8, label="y (target today)")
    lag_col = f"{base_target}_lag1"
    # Plot lagged target if present.
    if lag_col in df.columns:
        plt.plot(df["date"], df[lag_col], linewidth=1, alpha=0.8, label=f"{lag_col}")

    apply_common_style(
        title=f"Daily Target and Lagged Target ({base_target})",
        xlabel="Date",
        ylabel=base_target,
    )


# Plot scatter of target vs each predictor.
def plot_model_feature_scatter(model_train: pl.DataFrame, base_target: str = "rv"):
    # Convert to pandas for plotting.
    df = model_train.to_pandas()

    features = ["D_mad_lag1",
                "D_gap_lag1",
                "D_var_lag1",
                "D_std_lag1",
                "D_iqr_lag1",
                "D_mad_7d",
                "D_gap_7d",
                "D_var_7d",
                "D_std_7d",
                "D_iqr_7d",
                f"{base_target}_lag1"]
    titles = {
        "D_mad_lag1": f"y vs D_mad_lag1 with {base_target}",
        "D_gap_lag1": f"y vs D_gap_lag1 with {base_target}",
        "D_var_lag1": f"y vs D_var_lag1 with {base_target}",
        "D_std_lag1": f"y vs D_std_lag1 with {base_target}",
        "D_iqr_lag1": f"y vs D_iqr_lag1 with {base_target}",
        "D_mad_7d": f"y vs D_mad_7d with {base_target}",
        "D_gap_7d": f"y vs D_gap_7d with {base_target}",
        "D_var_7d": f"y vs D_var_7d with {base_target}",
        "D_std_7d": f"y vs D_std_7d with {base_target}",
        "D_iqr_7d": f"y vs D_iqr_7d with {base_target}",
        f"{base_target}_lag1": f"y vs {base_target}_lag1 (target lag-1)",
    }

    # Draw a scatter for each available feature.
    for feat in features:
        if feat not in df.columns:
            continue

        plt.figure(figsize=(6, 3))
        plt.scatter(df[feat], df["y"], s=10, alpha=0.6, label=f"({feat}, y)")
        apply_common_style(
            title=titles.get(feat, f"y vs {feat}"),
            xlabel=feat,
            ylabel="y (target)",
        )

# Plot correlation heatmap for target and predictors.
def plot_model_correlations(model_train: pl.DataFrame, base_target: str = "rv"):
    cols = ["y", "D_mad_lag1", "D_gap_lag1", "D_var_lag1", "D_std_lag1", "D_iqr_lag1",
            "D_mad_7d", "D_gap_7d", "D_var_7d", "D_std_7d", "D_iqr_7d",
            f"{base_target}_lag1"]
    # Keep only existing columns.
    cols = [c for c in cols if c in model_train.columns]

    df = model_train.select(cols).to_pandas()
    corr = df.corr()

    # Draw heatmap.
    plt.figure(figsize=(5, 4))
    im = plt.imshow(corr.values, interpolation="nearest", aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
    plt.yticks(range(len(cols)), cols)

    plt.title("Correlation Matrix (Target & Predictors)", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_best_models_vs_baseline(
    validation_df: pl.DataFrame,
    results: dict,
    best_models: dict,
    target: str,
    baseline_name="Baseline_AR1",
    use_raw=True
):
    # Get dates and select actual series.
    dates = validation_df["date"].to_list()
    if use_raw:
        y = results[baseline_name]["y_validation_raw"]
        ylabel = f"Volatility (raw {target})"
    else:
        y = results[baseline_name]["y_validation"]
        ylabel = f"log(Volatility ({target}))"

    # Plot actual series and best model predictions.
    plt.figure(figsize=(12, 4))
    plt.plot(dates, y, label=f"Actual ({baseline_name})")

    unique_values = set(best_models.values())
    for item in unique_values:
        yhat = results[item]["predit_validation_raw"]
        plt.plot(dates, yhat, label=item)

    apply_common_style(
        title=f"Validation: Baseline ({baseline_name}) vs Predicted for the target {target}",
        xlabel="Date",
        ylabel=ylabel,
    )


def plot_all_models_vs_actual(
    validation_df: pl.DataFrame,
    results: dict,
    target: str
):
    # Collect dates and actual series.
    dates = validation_df["date"].to_list()
    # actual series
    any_key = next(iter(results))
    y = results[any_key]["y_validation_raw"]
    pred_key = "predit_validation_raw"   # matches fit_predict dict key


    plt.figure(figsize=(12, 4))
    plt.plot(dates, y, label="Actual", linewidth=2.0)

    # all model predictions
    for _, out in results.items():
        plt.plot(dates, out[pred_key], alpha=0.5, linewidth=0.3)


    plt.title(f"Validation: Actual vs predictions (all models) — target={target}", fontsize=10)
    plt.xlabel("Date", fontsize=8)
    plt.ylabel("RV", fontsize=8)
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()

# Plot normal-approximation rejection regions for a test stat.
def plot_null_test_normal(t_stat, alpha=0.05, alternative="greater", title="Hypothesis test (normal approx)"):
    # Build normal curve and critical values.
    x = np.linspace(-4, 4, 1000)
    pdf = (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * x**2)
    from scipy.stats import norm
    z_crit_two_tailed = norm.ppf(1 - alpha/2)
    z_crit_one_tailed = norm.ppf(1 - alpha)
    # critical values
    if alternative == "two-sided":
        crit = z_crit_two_tailed
        reject_left = x <= -crit
        reject_right = x >= crit
    elif alternative == "greater":
        crit = z_crit_one_tailed
        reject_left = np.zeros_like(x, dtype=bool)
        reject_right = x >= crit
    elif alternative == "less":
        crit = z_crit_one_tailed
        reject_left = x <= -crit
        reject_right = np.zeros_like(x, dtype=bool)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    plt.figure(figsize=(8, 4))
    plt.plot(x, pdf, label="pdf")
    plt.fill_between(x, 0, pdf, where=reject_left)
    plt.fill_between(x, 0, pdf, where=reject_right)
    plt.axvline(t_stat, color="green", label="t stats")
    plt.axvline(crit if alternative != "less" else -crit, linestyle="--", color="red", label="Critical Value")
    if alternative == "two-sided":
        plt.axvline(-crit, linestyle="--", color="red", label="Critical Value")

    apply_common_style(
        title=title,
        xlabel="Test statistic (t or z)",
        ylabel="Density",
    )

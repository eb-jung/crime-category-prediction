"""
visualize.py
------------
EDA and results visualizations for SF crime category prediction.
All functions accept an optional `save_path` argument to persist figures to disk.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
plt.rcParams["font.size"] = 11


def _save_or_show(fig: plt.Figure, save_path: str | None) -> None:
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_crime_distribution(df: pd.DataFrame, top_n: int = 15, save_path: str | None = None) -> None:
    """Bar chart of top-N crime categories by incident count."""
    counts = df["Category"].value_counts()
    top = counts.iloc[:top_n]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, top_n))
    ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1])
    ax.set_xlabel("Number of Incidents")
    ax.set_title(f"Top {top_n} Crime Categories (2003–2015)", fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for i, v in enumerate(top.values[::-1]):
        ax.text(v + 800, i, f"{v:,}", va="center", fontsize=9)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_temporal_patterns(df: pd.DataFrame, save_path: str | None = None) -> None:
    """2×2 grid: crimes by hour, day of week, month, and hour×day heatmap."""
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # Hour
    hourly = df.groupby("Hour").size()
    axes[0, 0].plot(hourly.index, hourly.values, "o-", lw=2, ms=6, color="#2E86AB")
    axes[0, 0].fill_between(hourly.index, hourly.values, alpha=0.25, color="#2E86AB")
    axes[0, 0].set_xlabel("Hour of Day")
    axes[0, 0].set_ylabel("Incidents")
    axes[0, 0].set_title("Crimes by Hour of Day", fontweight="bold")
    axes[0, 0].set_xticks(range(0, 24, 2))

    # Day of week
    daily = df["DayOfWeek"].value_counts().reindex(day_order)
    bar_colors = ["#A23B72" if d in ("Friday", "Saturday") else "#F18F01" for d in day_order]
    axes[0, 1].bar(day_order, daily.values, color=bar_colors)
    axes[0, 1].set_ylabel("Incidents")
    axes[0, 1].set_title("Crimes by Day of Week", fontweight="bold")
    axes[0, 1].tick_params(axis="x", rotation=30)

    # Month
    monthly = df.groupby("Month").size()
    axes[1, 0].bar(monthly.index, monthly.values, color="#3D9970")
    axes[1, 0].set_xlabel("Month")
    axes[1, 0].set_ylabel("Incidents")
    axes[1, 0].set_title("Crimes by Month", fontweight="bold")
    axes[1, 0].set_xticks(range(1, 13))
    axes[1, 0].set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], rotation=30)

    # Heatmap: hour × day
    pivot = pd.crosstab(df["Hour"], df["DayOfWeek"])[day_order]
    sns.heatmap(pivot, cmap="YlOrRd", ax=axes[1, 1], cbar_kws={"label": "Incidents"}, linewidths=0.3)
    axes[1, 1].set_title("Crime Density: Hour × Day", fontweight="bold")
    axes[1, 1].set_xlabel("Day of Week")
    axes[1, 1].set_ylabel("Hour of Day")

    fig.suptitle("Temporal Crime Patterns — San Francisco (2003–2015)", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_district_heatmap(df: pd.DataFrame, top_categories: int = 10, save_path: str | None = None) -> None:
    """Heatmap of crime category counts broken down by police district."""
    top_cats = df["Category"].value_counts().iloc[:top_categories].index
    pivot = pd.crosstab(df["PdDistrict"], df["Category"])[top_cats]
    pivot_norm = pivot.div(pivot.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        pivot_norm, cmap="Blues", ax=ax,
        annot=True, fmt=".0%", annot_kws={"size": 8},
        cbar_kws={"label": "Share of district crimes"},
        linewidths=0.4,
    )
    ax.set_title(f"Crime Category Mix by Police District (top {top_categories} categories)", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Police District")
    ax.tick_params(axis="x", rotation=40)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_model_comparison(summary_df: pd.DataFrame, save_path: str | None = None) -> None:
    """Horizontal bar chart comparing models on accuracy and log loss."""
    df = summary_df.copy()
    df["Accuracy_f"] = df["Accuracy"].str.rstrip("%").astype(float)
    df["LogLoss_f"] = df["Log Loss"].astype(float)
    df = df.sort_values("LogLoss_f")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    palette = sns.color_palette("viridis", len(df))

    axes[0].barh(df["Model"], df["Accuracy_f"], color=palette)
    axes[0].set_xlabel("Accuracy (%)")
    axes[0].set_title("Accuracy by Model", fontweight="bold")
    for i, (_, row) in enumerate(df.iterrows()):
        axes[0].text(row["Accuracy_f"] + 0.2, i, row["Accuracy"], va="center", fontsize=9)

    axes[1].barh(df["Model"], df["LogLoss_f"], color=palette)
    axes[1].set_xlabel("Log Loss (lower is better)")
    axes[1].set_title("Log Loss by Model", fontweight="bold")
    for i, (_, row) in enumerate(df.iterrows()):
        axes[1].text(row["LogLoss_f"] + 0.02, i, row["Log Loss"], va="center", fontsize=9)

    fig.suptitle("Model Comparison", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15, save_path: str | None = None) -> None:
    """Horizontal bar chart of top-N feature importances from a tree model."""
    top = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["Feature"][::-1], top["Importance"][::-1], color="steelblue")
    ax.set_xlabel("Feature Importance (mean decrease in impurity)")
    ax.set_title(f"Top {top_n} Predictive Features", fontweight="bold")
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], top_n: int = 8,
                          title: str = "Confusion Matrix", save_path: str | None = None) -> None:
    """Confusion matrix heatmap for the top-N most common categories."""
    fig, ax = plt.subplots(figsize=(10, 8))
    short = [l[:18] for l in labels[:top_n]]
    sns.heatmap(
        cm[:top_n, :top_n], annot=True, fmt="d", cmap="Blues",
        xticklabels=short, yticklabels=short, ax=ax,
        cbar_kws={"label": "Count"},
    )
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.tick_params(axis="x", rotation=40)
    fig.tight_layout()
    _save_or_show(fig, save_path)

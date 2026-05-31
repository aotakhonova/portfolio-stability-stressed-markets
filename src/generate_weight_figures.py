"""
Generate illustrative portfolio-weight figures for Chapter 4.

This script should be placed/run from the root folder of the GitHub project:
portfolio-stability-stressed-markets/

It uses the weight files produced by src/chapter4_backtest.py:
    chapter4_outputs/weights_constrained_mean_variance.csv
    chapter4_outputs/weights_cvar_optimization.csv
    chapter4_outputs/weights_robust_mean_variance.csv

It also uses the Kenneth French data zip in:
    data/49_Industry_Portfolios_CSV.zip

Outputs:
    chapter4_outputs/figure_4_x_selected_portfolio_weights.png
    chapter4_outputs/figure_4_y_weights_by_volatility_quintile.png
"""

import argparse
import re
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_OUTPUT_DIR = Path("chapter4_outputs")
DEFAULT_DATA_ZIP = Path("data/49_Industry_Portfolios_CSV.zip")
DEFAULT_CSV_NAME = "49_Industry_Portfolios.csv"
ROLLING_WINDOW = 60


WEIGHT_FILES = {
    "Constrained Mean-Variance": "weights_constrained_mean_variance.csv",
    "CVaR Optimization": "weights_cvar_optimization.csv",
    "Robust Mean-Variance": "weights_robust_mean_variance.csv",
}


def load_kf_49_monthly(zip_path: Path, csv_name: str = DEFAULT_CSV_NAME) -> pd.DataFrame:
    """Load monthly value-weighted Kenneth French 49 Industry Portfolio returns."""
    with zipfile.ZipFile(zip_path) as zf:
        raw = zf.read(csv_name).decode("latin1")

    lines = raw.splitlines()
    section_header = " Average Value Weighted Returns -- Monthly"
    start_idx = lines.index(section_header)

    header = lines[start_idx + 1].split(",")
    rows = []
    for line in lines[start_idx + 2:]:
        if not line.strip():
            break
        parts = line.split(",")
        if re.fullmatch(r"\d{6}", parts[0].strip()):
            rows.append(parts)
        else:
            break

    df = pd.DataFrame(rows, columns=header)
    df.rename(columns={df.columns[0]: "date"}, inplace=True)

    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["date"] = pd.to_datetime(df["date"], format="%Y%m")
    df = df.set_index("date")
    df = df.replace([-99.99, -999.0, -999], np.nan)
    df = df / 100.0
    return df


def load_weights(output_dir: Path) -> dict[str, pd.DataFrame]:
    """Load portfolio weights produced by the main backtest script."""
    weights = {}
    missing = []

    for method, filename in WEIGHT_FILES.items():
        path = output_dir / filename
        if not path.exists():
            missing.append(str(path))
            continue

        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index.name = "date"
        weights[method] = df

    if missing:
        raise FileNotFoundError(
            "Missing required weight files:\n"
            + "\n".join(missing)
            + "\n\nRun first: python src/chapter4_backtest.py"
        )

    return weights


def choose_date(weights: dict[str, pd.DataFrame], requested_date: str | None) -> pd.Timestamp:
    """Choose selected rebalancing date."""
    common_index = None
    for df in weights.values():
        common_index = df.index if common_index is None else common_index.intersection(df.index)

    if common_index is None or len(common_index) == 0:
        raise ValueError("No common dates found across weight files.")

    if requested_date is not None:
        target = pd.Timestamp(requested_date)
        if target in common_index:
            return target

        closest_pos = np.argmin(np.abs(common_index - target))
        closest = common_index[closest_pos]
        print(f"Requested date {target.date()} not available. Using closest available date: {closest.date()}")
        return closest

    # Default: choose March 2020 if available, otherwise choose the date with the highest
    # average concentration across methods as an illustrative allocation month.
    default = pd.Timestamp("2020-03-01")
    if default in common_index:
        return default

    concentrations = pd.DataFrame({
        method: (df.loc[common_index] ** 2).sum(axis=1)
        for method, df in weights.items()
    })
    return concentrations.mean(axis=1).idxmax()


def plot_selected_weights(weights: dict[str, pd.DataFrame], selected_date: pd.Timestamp, output_path: Path) -> None:
    """Create grouped bar chart of actual portfolio weights across the 49 industries."""
    data = pd.DataFrame({
        method: df.loc[selected_date] * 100.0
        for method, df in weights.items()
    })

    # Sort industries by the constrained mean-variance weights to make concentration visible.
    data = data.sort_values("Constrained Mean-Variance", ascending=False)

    ax = data.plot(kind="bar", figsize=(15, 7), width=0.82)
    ax.set_title(f"Portfolio Weights Across the 49 Industry Portfolios ({selected_date.strftime('%B %Y')})")
    ax.set_xlabel("Industry portfolio")
    ax.set_ylabel("Portfolio weight (%)")
    ax.legend(title="Method")
    ax.grid(axis="y", alpha=0.3)

    plt.xticks(rotation=75, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_weights_by_volatility_quintile(
    returns_df: pd.DataFrame,
    weights: dict[str, pd.DataFrame],
    selected_date: pd.Timestamp,
    output_path: Path,
) -> None:
    """Create bar chart showing how each method allocates across volatility quintiles."""
    estimation_window = returns_df.loc[: selected_date - pd.offsets.MonthBegin(1)].iloc[-ROLLING_WINDOW:]
    if len(estimation_window) < ROLLING_WINDOW:
        raise ValueError("Not enough data before selected date to compute 60-month volatility.")

    annualized_vol = estimation_window.std(ddof=1) * np.sqrt(12.0)

    # Rank assets into five volatility groups. Duplicate='drop' avoids errors if ties occur.
    vol_groups = pd.qcut(
        annualized_vol.rank(method="first"),
        q=5,
        labels=["Q1 lowest vol", "Q2", "Q3", "Q4", "Q5 highest vol"],
    )

    group_weights = {}
    for method, df in weights.items():
        w = df.loc[selected_date]
        group_weights[method] = w.groupby(vol_groups).sum() * 100.0

    data = pd.DataFrame(group_weights)

    ax = data.plot(kind="bar", figsize=(10, 6), width=0.78)
    ax.set_title(f"Portfolio Weights by 60-Month Volatility Quintile ({selected_date.strftime('%B %Y')})")
    ax.set_xlabel("Volatility quintile")
    ax.set_ylabel("Total portfolio weight (%)")
    ax.legend(title="Method")
    ax.grid(axis="y", alpha=0.3)

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate portfolio-weight figures for Chapter 4.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Folder containing backtest output CSV files.")
    parser.add_argument("--data-zip", default=str(DEFAULT_DATA_ZIP), help="Path to Kenneth French 49 Industry Portfolios zip file.")
    parser.add_argument("--date", default=None, help="Optional selected date, e.g. 2020-03-01.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = load_weights(output_dir)
    returns_df = load_kf_49_monthly(Path(args.data_zip))
    selected_date = choose_date(weights, args.date)

    fig1 = output_dir / "figure_4_x_selected_portfolio_weights.png"
    fig2 = output_dir / "figure_4_y_weights_by_volatility_quintile.png"

    plot_selected_weights(weights, selected_date, fig1)
    plot_weights_by_volatility_quintile(returns_df, weights, selected_date, fig2)

    print(f"Selected date: {selected_date.date()}")
    print(f"Saved: {fig1}")
    print(f"Saved: {fig2}")


if __name__ == "__main__":
    main()

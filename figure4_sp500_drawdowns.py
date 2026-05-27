"""
Stylized and Market Stress Figures for Chapters 1 and 2

This script generates the S&P 500 drawdown figure used in the introductory
part of the master's thesis:

Portfolio Stability Under Stressed Market Conditions:
A Comparative Analysis of Mean-Variance, CVaR, and Robust Portfolio Optimization

Generated output:
    figures/figure_4_sp500_drawdowns.png

Note:
    This script downloads S&P 500 data directly from FRED.
    Internet access is required when running the script.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

FRED_SP500_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500"

START_DATE = "2006-01-01"
END_DATE = "2024-12-31"

OUTPUT_DIR = Path("figures")
OUTPUT_FILE = OUTPUT_DIR / "figure_4_sp500_drawdowns.png"


# ============================================================
# DATA LOADING
# ============================================================

def load_sp500_from_fred(url: str = FRED_SP500_URL) -> pd.DataFrame:
    """
    Load S&P 500 index data directly from FRED.

    Parameters
    ----------
    url : str
        Direct FRED CSV download link.

    Returns
    -------
    pd.DataFrame
        DataFrame with Date and SP500 columns.
    """
    df = pd.read_csv(url)

    df.columns = ["Date", "SP500"]

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["SP500"] = pd.to_numeric(df["SP500"], errors="coerce")

    df = df.dropna(subset=["Date", "SP500"]).copy()

    return df


# ============================================================
# DRAWDOWN CALCULATION
# ============================================================

def calculate_drawdown(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Calculate S&P 500 drawdowns over the selected period.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing Date and SP500 columns.
    start_date : str
        Start date for the sample.
    end_date : str
        End date for the sample.

    Returns
    -------
    pd.DataFrame
        DataFrame containing S&P 500 values, running peak, and drawdown.
    """
    plot_df = df[
        (df["Date"] >= start_date)
        & (df["Date"] <= end_date)
    ].copy()

    plot_df["RunningPeak"] = plot_df["SP500"].cummax()
    plot_df["Drawdown"] = (plot_df["SP500"] / plot_df["RunningPeak"] - 1.0) * 100.0

    return plot_df


def nearest_row(plot_df: pd.DataFrame, date_str: str) -> pd.Series:
    """
    Find the row closest to a selected event date.

    Parameters
    ----------
    plot_df : pd.DataFrame
        DataFrame containing Date and Drawdown columns.
    date_str : str
        Event date.

    Returns
    -------
    pd.Series
        Row closest to the selected event date.
    """
    target = pd.to_datetime(date_str)
    idx = (plot_df["Date"] - target).abs().idxmin()
    return plot_df.loc[idx]


# ============================================================
# FIGURE 4
# ============================================================

def plot_sp500_drawdowns(plot_df: pd.DataFrame, output_path: Path) -> None:
    """
    Plot S&P 500 drawdowns and annotate major stress periods.

    Parameters
    ----------
    plot_df : pd.DataFrame
        DataFrame containing Date and Drawdown columns.
    output_path : Path
        Path where the figure will be saved.
    """
    gfc = nearest_row(plot_df, "2009-03-09")
    covid = nearest_row(plot_df, "2020-03-23")
    inflation = nearest_row(plot_df, "2022-10-12")

    plt.figure(figsize=(11, 6))
    plt.plot(plot_df["Date"], plot_df["Drawdown"], linewidth=2)
    plt.axhline(0, linewidth=1)

    plt.xlabel("Year")
    plt.ylabel("Drawdown (%)")
    plt.title("Figure 4. S&P 500 Drawdowns During Selected Recent Stress Episodes")
    plt.grid(True, alpha=0.3)

    plt.annotate(
        "Global Financial Crisis",
        xy=(gfc["Date"], gfc["Drawdown"]),
        xytext=(gfc["Date"], gfc["Drawdown"] + 10),
        arrowprops=dict(arrowstyle="->"),
    )

    plt.annotate(
        "COVID-19 Shock",
        xy=(covid["Date"], covid["Drawdown"]),
        xytext=(covid["Date"], covid["Drawdown"] + 10),
        arrowprops=dict(arrowstyle="->"),
    )

    plt.annotate(
        "Inflation / Rates Shock",
        xy=(inflation["Date"], inflation["Drawdown"]),
        xytext=(inflation["Date"], inflation["Drawdown"] + 10),
        arrowprops=dict(arrowstyle="->"),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# RUN EVERYTHING
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading S&P 500 data from FRED...")
    sp500 = load_sp500_from_fred()

    print("Calculating drawdowns...")
    drawdown_df = calculate_drawdown(
        df=sp500,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    print("Generating Figure 4...")
    plot_sp500_drawdowns(
        plot_df=drawdown_df,
        output_path=OUTPUT_FILE,
    )

    print(f"Saved figure to: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()

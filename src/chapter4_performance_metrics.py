a"""
Chapter 4 Performance Metrics and Drawdown Analysis

This script builds Table 4.2 and Figure 4.2 for the master's thesis:

Portfolio Stability Under Stressed Market Conditions:
A Comparative Analysis of Mean-Variance, CVaR, and Robust Portfolio Optimization

It reads the out-of-sample return series produced by chapter4_backtest.py and calculates:
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Downside Deviation
- Drawdown paths figure

Expected input:
    chapter4_outputs/oos_returns.csv

Generated outputs:
    chapter4_outputs/table_4_2_raw.csv
    chapter4_outputs/table_4_2_formatted.csv
    chapter4_outputs/figure_4_2_drawdowns.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

OUTPUT_DIR = Path("chapter4_outputs")
RETURNS_FILE = OUTPUT_DIR / "oos_returns.csv"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def sharpe_ratio(r: pd.Series, rf: float = 0.0) -> float:
    """
    Calculate the annualized Sharpe ratio using monthly returns.

    Parameters
    ----------
    r : pd.Series
        Monthly portfolio returns.
    rf : float
        Monthly risk-free rate. Default is 0.0.

    Returns
    -------
    float
        Annualized Sharpe ratio.
    """
    excess = r - rf
    vol = excess.std(ddof=1)

    if vol == 0:
        return np.nan

    return (excess.mean() / vol) * np.sqrt(12)


def downside_deviation(r: pd.Series, target: float = 0.0) -> float:
    """
    Calculate annualized downside deviation using monthly returns.

    Parameters
    ----------
    r : pd.Series
        Monthly portfolio returns.
    target : float
        Minimum acceptable monthly return. Default is 0.0.

    Returns
    -------
    float
        Annualized downside deviation.
    """
    downside = np.minimum(r - target, 0)
    return np.sqrt(np.mean(downside ** 2)) * np.sqrt(12)


def sortino_ratio(r: pd.Series, target: float = 0.0) -> float:
    """
    Calculate the annualized Sortino ratio using monthly returns.

    Parameters
    ----------
    r : pd.Series
        Monthly portfolio returns.
    target : float
        Minimum acceptable monthly return. Default is 0.0.

    Returns
    -------
    float
        Annualized Sortino ratio.
    """
    downside = np.minimum(r - target, 0)
    downside_std = np.sqrt(np.mean(downside ** 2))

    if downside_std == 0:
        return np.nan

    return ((r.mean() - target) / downside_std) * np.sqrt(12)


def drawdown_series(r: pd.Series) -> pd.Series:
    """
    Calculate the drawdown path of a portfolio return series.

    Parameters
    ----------
    r : pd.Series
        Monthly portfolio returns.

    Returns
    -------
    pd.Series
        Drawdown series.
    """
    wealth = (1.0 + r).cumprod()
    peak = wealth.cummax()
    return (wealth / peak) - 1.0


def max_drawdown(r: pd.Series) -> float:
    """
    Calculate the maximum drawdown of a portfolio return series.

    Parameters
    ----------
    r : pd.Series
        Monthly portfolio returns.

    Returns
    -------
    float
        Maximum drawdown.
    """
    return drawdown_series(r).min()


# ============================================================
# TABLE 4.2
# ============================================================

def build_table_4_2(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the raw performance and risk statistics table.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Out-of-sample monthly return series for all portfolio methods.

    Returns
    -------
    pd.DataFrame
        Raw Table 4.2 values.
    """
    rows = []

    for method in returns_df.columns:
        r = returns_df[method].dropna()

        rows.append({
            "Method": method,
            "Sharpe Ratio": sharpe_ratio(r),
            "Sortino Ratio": sortino_ratio(r),
            "Maximum Drawdown": max_drawdown(r),
            "Downside Deviation": downside_deviation(r),
        })

    return pd.DataFrame(rows)


def format_table_4_2(table: pd.DataFrame) -> pd.DataFrame:
    """
    Format Table 4.2 for thesis presentation.

    Parameters
    ----------
    table : pd.DataFrame
        Raw Table 4.2 values.

    Returns
    -------
    pd.DataFrame
        Formatted Table 4.2 values.
    """
    out = table.copy()

    out["Sharpe Ratio"] = out["Sharpe Ratio"].map(lambda x: f"{x:.3f}")
    out["Sortino Ratio"] = out["Sortino Ratio"].map(lambda x: f"{x:.3f}")
    out["Maximum Drawdown"] = (out["Maximum Drawdown"] * 100).map(lambda x: f"{x:.2f}%")
    out["Downside Deviation"] = (out["Downside Deviation"] * 100).map(lambda x: f"{x:.2f}%")

    return out


# ============================================================
# FIGURE 4.2
# ============================================================

def plot_figure_4_2(returns_df: pd.DataFrame, output_path: Path) -> None:
    """
    Plot drawdown paths for all portfolio methods.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Out-of-sample monthly return series for all portfolio methods.
    output_path : Path
        File path for saving the figure.
    """
    plt.figure(figsize=(10, 6))

    for method in returns_df.columns:
        dd = drawdown_series(returns_df[method].dropna())
        plt.plot(dd.index, dd * 100, label=method)

    plt.title("Figure 4.2. Drawdown Paths of the Three Portfolio Methods")
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(returns_df: pd.DataFrame, table_4_2: pd.DataFrame) -> None:
    """
    Save Table 4.2 and Figure 4.2 outputs.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Out-of-sample monthly return series.
    table_4_2 : pd.DataFrame
        Raw Table 4.2 values.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    table_4_2.to_csv(OUTPUT_DIR / "table_4_2_raw.csv", index=False)
    format_table_4_2(table_4_2).to_csv(OUTPUT_DIR / "table_4_2_formatted.csv", index=False)

    plot_figure_4_2(
        returns_df=returns_df,
        output_path=OUTPUT_DIR / "figure_4_2_drawdowns.png",
    )


# ============================================================
# RUN EVERYTHING
# ============================================================

def main() -> None:
    if not RETURNS_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {RETURNS_FILE}. "
            "Please run src/chapter4_backtest.py first to generate oos_returns.csv."
        )

    print("Loading out-of-sample returns...")
    returns_df = pd.read_csv(RETURNS_FILE, index_col=0, parse_dates=True)

    print("\nFirst rows of out-of-sample returns:")
    print(returns_df.head())

    print("\nBuilding Table 4.2...")
    table_4_2 = build_table_4_2(returns_df)

    print("\nTable 4.2 (raw):")
    print(table_4_2)

    print("\nTable 4.2 (formatted):")
    print(format_table_4_2(table_4_2))

    save_outputs(returns_df, table_4_2)

    print(f"\nSaved outputs to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

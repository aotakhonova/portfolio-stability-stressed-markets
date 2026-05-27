"""
Chapter 4 Stress-Period Portfolio Performance Analysis

This script builds Table 4.4 and Figure 4.5 for the master's thesis:

Portfolio Stability Under Stressed Market Conditions:
A Comparative Analysis of Mean-Variance, CVaR, and Robust Portfolio Optimization

The script identifies stressed market months using two conditions:
1. Monthly average VIX is above its 75th percentile.
2. The average return across the Kenneth French 49 Industry Portfolios is negative.

It then evaluates the out-of-sample portfolio returns only during those stressed months.

Expected inputs:
    data/49_Industry_Portfolios_CSV.zip
    data/VIXCLS.csv
    chapter4_outputs/oos_returns.csv

Generated outputs:
    chapter4_outputs/table_4_4_raw.csv
    chapter4_outputs/table_4_4_formatted.csv
    chapter4_outputs/figure_4_5_stress_cumulative.png
"""

import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = Path("data")
OUTPUT_DIR = Path("chapter4_outputs")

ZIP_FILE = DATA_DIR / "49_Industry_Portfolios_CSV.zip"
CSV_NAME = "49_Industry_Portfolios.csv"
VIX_FILE = DATA_DIR / "VIXCLS.csv"
RETURNS_FILE = OUTPUT_DIR / "oos_returns.csv"

VIX_QUANTILE = 0.75


# ============================================================
# LOAD KENNETH FRENCH MONTHLY 49 INDUSTRY DATA
# ============================================================

def load_kf_49_monthly(zip_path: Path, csv_name: str) -> pd.DataFrame:
    """
    Load the 'Average Value Weighted Returns -- Monthly' block
    from the Kenneth French 49 Industry Portfolios CSV zip file.

    Parameters
    ----------
    zip_path : Path
        Path to the Kenneth French 49 Industry Portfolios zip file.
    csv_name : str
        CSV file name inside the zip archive.

    Returns
    -------
    pd.DataFrame
        Monthly industry returns in decimal form.
    """
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Could not find {zip_path}. "
            "Please place 49_Industry_Portfolios_CSV.zip inside the data folder."
        )

    with zipfile.ZipFile(zip_path) as zf:
        raw = zf.read(csv_name).decode("latin1")

    lines = raw.splitlines()
    section_header = "  Average Value Weighted Returns -- Monthly"

    try:
        start_idx = lines.index(section_header)
    except ValueError as exc:
        raise ValueError("Could not find the monthly value-weighted section.") from exc

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

    # Missing value codes used in Kenneth French files
    df = df.replace([-99.99, -999.0, -999], np.nan)

    # Convert percentage returns to decimals
    df = df / 100.0

    return df


# ============================================================
# LOAD VIX DATA
# ============================================================

def load_vix_monthly(vix_file: Path) -> pd.Series:
    """
    Load daily VIX data and convert it to monthly average VIX.

    The file is expected to contain:
    - observation_date
    - VIXCLS

    Parameters
    ----------
    vix_file : Path
        Path to the VIX CSV file.

    Returns
    -------
    pd.Series
        Monthly average VIX.
    """
    if not vix_file.exists():
        raise FileNotFoundError(
            f"Could not find {vix_file}. "
            "Please place the VIX file inside the data folder and name it VIXCLS.csv."
        )

    vix = pd.read_csv(vix_file)

    required_columns = {"observation_date", "VIXCLS"}
    missing_columns = required_columns.difference(vix.columns)

    if missing_columns:
        raise ValueError(
            f"VIX file is missing required columns: {missing_columns}. "
            "The file should contain observation_date and VIXCLS."
        )

    vix["observation_date"] = pd.to_datetime(vix["observation_date"])
    vix["VIXCLS"] = pd.to_numeric(vix["VIXCLS"], errors="coerce")

    vix = vix.dropna(subset=["VIXCLS"]).set_index("observation_date")

    vix_monthly = vix["VIXCLS"].resample("MS").mean()
    vix_monthly.index = vix_monthly.index.to_period("M").to_timestamp()
    vix_monthly.name = "VIX"

    return vix_monthly


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def sharpe_ratio(r: pd.Series, rf: float = 0.0) -> float:
    """
    Calculate annualized Sharpe ratio using monthly returns.
    """
    excess = r - rf
    vol = excess.std(ddof=1)

    if vol == 0:
        return np.nan

    return (excess.mean() / vol) * np.sqrt(12)


def max_drawdown(r: pd.Series) -> float:
    """
    Calculate maximum drawdown from a return series.
    """
    wealth = (1.0 + r).cumprod()
    peak = wealth.cummax()
    drawdown = (wealth / peak) - 1.0
    return drawdown.min()


def cumulative_return(r: pd.Series) -> float:
    """
    Calculate cumulative return from a return series.
    """
    return np.prod(1.0 + r) - 1.0


# ============================================================
# STRESS MONTH IDENTIFICATION
# ============================================================

def identify_stress_months(industry_returns: pd.DataFrame, vix_monthly: pd.Series) -> pd.DataFrame:
    """
    Identify stressed market months using VIX and market deterioration conditions.

    A stress month is defined as a month in which:
    - VIX is above its selected quantile threshold.
    - The average return across the 49 industry portfolios is negative.

    Parameters
    ----------
    industry_returns : pd.DataFrame
        Monthly Kenneth French 49 industry returns.
    vix_monthly : pd.Series
        Monthly average VIX.

    Returns
    -------
    pd.DataFrame
        DataFrame containing VIX, market proxy, and stress-month indicator.
    """
    market_proxy = industry_returns.mean(axis=1)
    market_proxy.name = "MarketProxy"
    market_proxy.index = market_proxy.index.to_period("M").to_timestamp()

    stress_df = pd.concat([vix_monthly, market_proxy], axis=1).dropna()
    stress_df.columns = ["VIX", "MarketProxy"]

    vix_threshold = stress_df["VIX"].quantile(VIX_QUANTILE)

    stress_df["StressMonth"] = (
        (stress_df["VIX"] > vix_threshold)
        & (stress_df["MarketProxy"] < 0)
    )

    print(f"VIX {int(VIX_QUANTILE * 100)}th percentile threshold: {vix_threshold:.2f}")
    print(f"Number of stressed months: {int(stress_df['StressMonth'].sum())}")

    return stress_df


# ============================================================
# TABLE 4.4
# ============================================================

def build_table_4_4(stress_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Build Table 4.4 with portfolio performance during stressed months.

    Parameters
    ----------
    stress_returns : pd.DataFrame
        Out-of-sample returns restricted to stressed months.

    Returns
    -------
    pd.DataFrame
        Raw Table 4.4 values.
    """
    rows = []

    for method in stress_returns.columns:
        r = stress_returns[method].dropna()

        rows.append({
            "Method": method,
            "Average Return During Stress": r.mean(),
            "Volatility During Stress": r.std(ddof=1),
            "Sharpe Ratio During Stress": sharpe_ratio(r),
            "Maximum Drawdown During Stress": max_drawdown(r),
            "Cumulative Return During Stress": cumulative_return(r),
        })

    return pd.DataFrame(rows)


def format_table_4_4(table: pd.DataFrame) -> pd.DataFrame:
    """
    Format Table 4.4 for thesis presentation.
    """
    out = table.copy()

    out["Average Return During Stress"] = (
        out["Average Return During Stress"] * 100
    ).map(lambda x: f"{x:.2f}%")

    out["Volatility During Stress"] = (
        out["Volatility During Stress"] * 100
    ).map(lambda x: f"{x:.2f}%")

    out["Sharpe Ratio During Stress"] = (
        out["Sharpe Ratio During Stress"]
    ).map(lambda x: f"{x:.3f}")

    out["Maximum Drawdown During Stress"] = (
        out["Maximum Drawdown During Stress"] * 100
    ).map(lambda x: f"{x:.2f}%")

    out["Cumulative Return During Stress"] = (
        out["Cumulative Return During Stress"] * 100
    ).map(lambda x: f"{x:.2f}%")

    return out


# ============================================================
# FIGURE 4.5
# ============================================================

def plot_figure_4_5(stress_returns: pd.DataFrame, output_path: Path) -> None:
    """
    Plot cumulative portfolio value during stressed market conditions.

    Parameters
    ----------
    stress_returns : pd.DataFrame
        Out-of-sample returns restricted to stressed months.
    output_path : Path
        File path for saving the figure.
    """
    stress_wealth = (1.0 + stress_returns).cumprod()

    plt.figure(figsize=(10, 6))

    for method in stress_wealth.columns:
        plt.plot(stress_wealth.index, stress_wealth[method], label=method)

    plt.title("Figure 4.5. Cumulative Portfolio Value During Stressed Market Conditions")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(table_4_4: pd.DataFrame, stress_returns: pd.DataFrame) -> None:
    """
    Save Table 4.4 and Figure 4.5 outputs.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    table_4_4.to_csv(OUTPUT_DIR / "table_4_4_raw.csv", index=False)
    format_table_4_4(table_4_4).to_csv(
        OUTPUT_DIR / "table_4_4_formatted.csv",
        index=False,
    )

    stress_returns.to_csv(OUTPUT_DIR / "stress_period_returns.csv")

    plot_figure_4_5(
        stress_returns=stress_returns,
        output_path=OUTPUT_DIR / "figure_4_5_stress_cumulative.png",
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

    print("Loading Kenneth French industry data...")
    industry_returns = load_kf_49_monthly(ZIP_FILE, CSV_NAME)

    print("Loading VIX data...")
    vix_monthly = load_vix_monthly(VIX_FILE)

    print("Loading out-of-sample portfolio returns...")
    returns_df = pd.read_csv(RETURNS_FILE, index_col=0, parse_dates=True)
    returns_df.index = returns_df.index.to_period("M").to_timestamp()

    print("Identifying stressed market months...")
    stress_df = identify_stress_months(industry_returns, vix_monthly)

    stress_months = stress_df.index[stress_df["StressMonth"]]
    stress_returns = returns_df.loc[returns_df.index.intersection(stress_months)].copy()

    print(f"Stress-return sample months: {len(stress_returns)}")
    print("\nFirst rows of stress-period returns:")
    print(stress_returns.head())

    print("\nBuilding Table 4.4...")
    table_4_4 = build_table_4_4(stress_returns)

    print("\nTable 4.4 (raw):")
    print(table_4_4)

    print("\nTable 4.4 (formatted):")
    print(format_table_4_4(table_4_4))

    save_outputs(table_4_4, stress_returns)

    print(f"\nSaved outputs to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

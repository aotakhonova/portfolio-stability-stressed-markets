"""
Chapter 4 Portfolio Optimization Backtest

This script runs a rolling out-of-sample backtest comparing:
1. Constrained Mean-Variance Optimization
2. CVaR Optimization
3. Robust Mean-Variance Optimization

Dataset:
Kenneth R. French Data Library - 49 Industry Portfolios, monthly value-weighted returns.

Author: Asalkhon Otakhonova
Thesis: Portfolio Stability Under Stressed Market Conditions: A Comparative Analysis of
Mean-Variance, CVaR, and Robust Portfolio Optimization
"""

import argparse
import re
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize


# ============================================================
# DEFAULT SETTINGS
# ============================================================

DEFAULT_ZIP_FILE = "data/49_Industry_Portfolios_CSV.zip"
DEFAULT_CSV_NAME = "49_Industry_Portfolios.csv"
DEFAULT_OUTPUT_DIR = "chapter4_outputs"

START_OOS = "1990-01-01"
END_OOS = "2026-03-01"
ROLLING_WINDOW = 60
LOWER_BOUND = 0.00
UPPER_BOUND = 0.10
CVAR_ALPHA = 0.95
DELTA = 0.5
TARGET_MULTIPLIER = 0.95
INITIAL_WEALTH = 1.0


# ============================================================
# DATA LOADING
# ============================================================

def load_kf_49_monthly(zip_path: str, csv_name: str) -> pd.DataFrame:
    """
    Load the 'Average Value Weighted Returns -- Monthly' block from the
    Kenneth French 49 Industry Portfolios CSV zip file.
    """
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

    # Missing value codes used in French data files
    df = df.replace([-99.99, -999.0, -999], np.nan)

    # Convert percentage returns into decimal returns
    df = df / 100.0

    return df


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def annualized_return_geometric(r: pd.Series) -> float:
    return (np.prod(1.0 + r.values) ** (12.0 / len(r))) - 1.0


def annualized_volatility(r: pd.Series) -> float:
    return r.std(ddof=1) * np.sqrt(12.0)


def cumulative_return(r: pd.Series) -> float:
    return np.prod(1.0 + r.values) - 1.0


def terminal_wealth(r: pd.Series, initial_wealth: float = 1.0) -> float:
    return initial_wealth * np.prod(1.0 + r.values)


# ============================================================
# OPTIMIZATION ROUTINES
# ============================================================

def solve_mean_variance(mu: np.ndarray, sigma: np.ndarray, target: float, lower: float, upper: float):
    """
    Minimize portfolio variance subject to full investment, target return,
    and portfolio weight bounds.
    """
    n = len(mu)
    sigma = sigma + np.eye(n) * 1e-8  # numerical stabilization

    x0 = np.repeat(1.0 / n, n)
    bounds = [(lower, upper)] * n

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": lambda w, mu=mu, target=target: np.dot(mu, w) - target},
    ]

    objective = lambda w: float(w @ sigma @ w)

    return minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 500, "disp": False},
    )


def solve_cvar(window_returns: np.ndarray, target: float, alpha: float, lower: float, upper: float):
    """
    Minimize empirical CVaR using a linear programming formulation.
    """
    t_periods, n_assets = window_returns.shape
    mu = window_returns.mean(axis=0)

    # Variables: [weights, zeta, auxiliary losses]
    n_variables = n_assets + 1 + t_periods

    c = np.zeros(n_variables)
    c[n_assets] = 1.0
    c[n_assets + 1:] = 1.0 / ((1.0 - alpha) * t_periods)

    a_eq = np.zeros((1, n_variables))
    a_eq[0, :n_assets] = 1.0
    b_eq = np.array([1.0])

    a_ub = []
    b_ub = []

    # Target return: mu'w >= target  ->  -mu'w <= -target
    row = np.zeros(n_variables)
    row[:n_assets] = -mu
    a_ub.append(row)
    b_ub.append(-target)

    # Auxiliary constraints for CVaR losses
    for t in range(t_periods):
        row = np.zeros(n_variables)
        row[:n_assets] = -window_returns[t]
        row[n_assets] = -1.0
        row[n_assets + 1 + t] = -1.0
        a_ub.append(row)
        b_ub.append(0.0)

    bounds = [(lower, upper)] * n_assets + [(None, None)] + [(0.0, None)] * t_periods

    return linprog(
        c=c,
        A_ub=np.array(a_ub),
        b_ub=np.array(b_ub),
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(df: pd.DataFrame):
    """
    Run a rolling out-of-sample backtest for the three portfolio strategies.
    """
    df = df.loc[:END_OOS].copy()
    oos_dates = df.loc[START_OOS:END_OOS].index

    results = {
        "Constrained Mean-Variance": [],
        "CVaR Optimization": [],
        "Robust Mean-Variance": [],
    }

    weights_store = {method: [] for method in results}
    valid_dates = []
    failures = []

    for dt in oos_dates:
        hist = df.loc[: dt - pd.offsets.MonthBegin(1)].iloc[-ROLLING_WINDOW:]

        if len(hist) < ROLLING_WINDOW:
            continue

        if hist.isna().any().any():
            failures.append((dt, "missing data in estimation window"))
            continue

        if df.loc[dt].isna().any():
            failures.append((dt, "missing realized return"))
            continue

        window_returns = hist.values
        mu_hat = hist.mean().values
        sigma_hat = hist.cov().values

        target = TARGET_MULTIPLIER * mu_hat.mean()
        se_mu = hist.std(ddof=1).values / np.sqrt(ROLLING_WINDOW)
        mu_robust = mu_hat - DELTA * se_mu

        mv_res = solve_mean_variance(mu_hat, sigma_hat, target, LOWER_BOUND, UPPER_BOUND)
        cvar_res = solve_cvar(window_returns, target, CVAR_ALPHA, LOWER_BOUND, UPPER_BOUND)
        rb_res = solve_mean_variance(mu_robust, sigma_hat, target, LOWER_BOUND, UPPER_BOUND)

        if not mv_res.success:
            failures.append((dt, f"Mean-Variance failed: {mv_res.message}"))
            continue

        if not cvar_res.success:
            failures.append((dt, f"CVaR failed: {cvar_res.message}"))
            continue

        if not rb_res.success:
            failures.append((dt, f"Robust Mean-Variance failed: {rb_res.message}"))
            continue

        weights = {
            "Constrained Mean-Variance": mv_res.x,
            "CVaR Optimization": cvar_res.x[:df.shape[1]],
            "Robust Mean-Variance": rb_res.x,
        }

        realized_returns = df.loc[dt].values

        for method, w in weights.items():
            results[method].append(float(np.dot(w, realized_returns)))
            weights_store[method].append(w)

        valid_dates.append(dt)

    returns_df = pd.DataFrame(results, index=pd.Index(valid_dates, name="date"))

    weights_dict = {
        method: pd.DataFrame(values, index=pd.Index(valid_dates, name="date"), columns=df.columns)
        for method, values in weights_store.items()
    }

    return returns_df, weights_dict, failures


# ============================================================
# OUTPUTS
# ============================================================

def build_table_4_1(returns_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in returns_df.columns:
        r = returns_df[method].dropna()
        rows.append({
            "Method": method,
            "Average Monthly Return": r.mean(),
            "Annualized Return": annualized_return_geometric(r),
            "Monthly Volatility": r.std(ddof=1),
            "Annualized Volatility": annualized_volatility(r),
            "Cumulative Return": cumulative_return(r),
            "Terminal Wealth": terminal_wealth(r, INITIAL_WEALTH),
        })
    return pd.DataFrame(rows)


def format_table_4_1(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    pct_cols = [
        "Average Monthly Return",
        "Annualized Return",
        "Monthly Volatility",
        "Annualized Volatility",
        "Cumulative Return",
    ]

    for col in pct_cols:
        out[col] = (out[col] * 100).map(lambda x: f"{x:.2f}%")

    out["Terminal Wealth"] = out["Terminal Wealth"].map(lambda x: f"{x:.3f}")
    return out


def plot_cumulative_wealth(returns_df: pd.DataFrame, output_path: Path):
    wealth = (1.0 + returns_df).cumprod() * INITIAL_WEALTH

    plt.figure(figsize=(10, 6))
    for col in wealth.columns:
        plt.plot(wealth.index, wealth[col], label=col)

    plt.title("Figure 4.1. Cumulative Out-of-Sample Portfolio Value")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_outputs(returns_df: pd.DataFrame, weights_dict: dict, table_4_1: pd.DataFrame, output_dir: str):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    returns_df.to_csv(out_dir / "oos_returns.csv")
    table_4_1.to_csv(out_dir / "table_4_1_raw.csv", index=False)
    format_table_4_1(table_4_1).to_csv(out_dir / "table_4_1_formatted.csv", index=False)

    for method, wdf in weights_dict.items():
        safe_name = method.lower().replace(" ", "_").replace("-", "_")
        wdf.to_csv(out_dir / f"weights_{safe_name}.csv")

    plot_cumulative_wealth(returns_df, out_dir / "figure_4_1_cumulative_wealth.png")

    print(f"Saved outputs to: {out_dir.resolve()}")


# ============================================================
# MAIN SCRIPT
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Run Chapter 4 portfolio optimization backtest.")
    parser.add_argument("--zip-file", default=DEFAULT_ZIP_FILE, help="Path to Kenneth French zip file.")
    parser.add_argument("--csv-name", default=DEFAULT_CSV_NAME, help="CSV filename inside the zip archive.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Folder where outputs will be saved.")
    args = parser.parse_args()

    print("Loading data...")
    df = load_kf_49_monthly(args.zip_file, args.csv_name)

    print("Running rolling out-of-sample backtest...")
    returns_df, weights_dict, failures = run_backtest(df)

    print(f"Successful out-of-sample months: {len(returns_df)}")
    print(f"Failed months: {len(failures)}")

    if failures:
        print("First few failures:")
        for failure in failures[:10]:
            print(failure)

    table_4_1 = build_table_4_1(returns_df)

    print("\nTable 4.1 formatted:")
    print(format_table_4_1(table_4_1))

    save_outputs(returns_df, weights_dict, table_4_1, args.output_dir)


if __name__ == "__main__":
    main()

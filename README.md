# Portfolio Stability Under Stressed Market Conditions

This repository contains the Python implementation used for the empirical analysis in Chapter 4 of the master's thesis:

**Portfolio Stability Under Stressed Market Conditions: A Comparative Analysis of Mean-Variance, CVaR, and Robust Portfolio Optimization**

The project compares three portfolio construction approaches using monthly industry portfolio returns:

1. Constrained Mean-Variance Optimization  
2. CVaR Optimization  
3. Robust Mean-Variance Optimization  

The analysis uses a rolling out-of-sample backtest with a 60-month estimation window and monthly portfolio re-optimization.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── place_49_Industry_Portfolios_CSV.zip_here.txt
├── src/
│   └── chapter4_backtest.py
└── chapter4_outputs/
    └── .gitkeep
```

## Data source

The empirical analysis is based on the **49 Industry Portfolios** dataset from the Kenneth R. French Data Library.

The dataset file should be downloaded separately and placed in the `data/` folder as:

```text
data/49_Industry_Portfolios_CSV.zip
```

The script reads the monthly value-weighted return section from the file:

```text
49_Industry_Portfolios.csv
```

## Methodology summary

The script performs a rolling backtest over the out-of-sample period from January 1990 to March 2026. For each month, it uses the previous 60 months of data to estimate expected returns and the covariance matrix. The portfolio is then optimized and evaluated using the realized return of the next month.

The optimization includes the following constraints:

- Portfolio weights sum to one.
- No short-selling is allowed.
- Each industry weight is bounded between 0% and 10%.
- A relaxed target return is applied in each rolling window.

The robust mean-variance strategy adjusts expected returns downward by subtracting a fraction of the standard error of the estimated mean return.

## How to run the code

First, install the required Python packages:

```bash
pip install -r requirements.txt
```

Then place the Kenneth French zip file in the `data/` folder and run:

```bash
python src/chapter4_backtest.py
```

The script will save all results in the `chapter4_outputs/` folder.

## Generated outputs

After running the script, the following files are created:

```text
chapter4_outputs/oos_returns.csv
chapter4_outputs/table_4_1_raw.csv
chapter4_outputs/table_4_1_formatted.csv
chapter4_outputs/weights_constrained_mean_variance.csv
chapter4_outputs/weights_cvar_optimization.csv
chapter4_outputs/weights_robust_mean_variance.csv
chapter4_outputs/figure_4_1_cumulative_wealth.png
```

These outputs are used for the empirical results, tables, and figures in Chapter 4 of the thesis.

## Reproducibility note

The repository does not include the raw Kenneth French dataset. The data should be downloaded directly from the official Kenneth R. French Data Library so that the analysis can be independently reproduced with the same public data source.

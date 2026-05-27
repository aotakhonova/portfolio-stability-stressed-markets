"""
Stylized Figures for Chapters 1 and 2

This script generates the stylized explanatory figures used in the early
chapters of the master's thesis:

Portfolio Stability Under Stressed Market Conditions:
A Comparative Analysis of Mean-Variance, CVaR, and Robust Portfolio Optimization

Generated outputs:
    figures/figure_1_efficient_frontier.png
    figures/figure_2_estimation_error_fragility.png
    figures/figure_3_market_stress_episodes.png
    figures/figure_5_weight_sensitivity.png
    figures/figure_6_var_cvar_tail.png
    figures/figure_7_classical_vs_robust.png

Note:
    These figures are stylized conceptual illustrations. They are not intended
    to report empirical estimation results. Their purpose is to visually explain
    the theoretical mechanisms discussed in Chapters 1 and 2.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle


# ============================================================
# SETTINGS
# ============================================================

OUTPUT_DIR = Path("figures")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def save_current_figure(output_path: Path) -> None:
    """
    Save the current matplotlib figure in high resolution.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def add_box(ax, xy, width, height, text, fontsize=9):
    """
    Add a rounded text box to an axis.
    """
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.03",
        linewidth=1.5,
        fill=False,
    )
    ax.add_patch(box)

    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        wrap=True,
    )


def add_arrow(ax, start, end):
    """
    Add an arrow between two points.
    """
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", linewidth=1.5),
    )


# ============================================================
# FIGURE 1
# ============================================================

def plot_figure_1_efficient_frontier() -> None:
    """
    Figure 1. Stylized mean-variance efficient frontier.

    This figure illustrates the basic trade-off between expected return and
    portfolio risk in the Markowitz mean-variance framework.
    """
    risk = np.linspace(0.05, 0.32, 200)

    # Stylized concave frontier
    expected_return = 0.04 + 0.80 * risk - 1.25 * risk ** 2

    min_var_risk = risk[25]
    min_var_return = expected_return[25]

    higher_risk = risk[95]
    higher_return = expected_return[95]

    plt.figure(figsize=(10, 6))
    plt.plot(risk, expected_return, linewidth=2, label="Efficient Frontier")
    plt.scatter([min_var_risk], [min_var_return], s=60, label="Minimum-Variance Portfolio")
    plt.scatter([higher_risk], [higher_return], s=60, label="Higher Return / Higher Risk")

    plt.annotate(
        "Minimum-Variance Portfolio",
        xy=(min_var_risk, min_var_return),
        xytext=(min_var_risk + 0.02, min_var_return - 0.03),
        arrowprops=dict(arrowstyle="->"),
    )

    plt.annotate(
        "Higher Return / Higher Risk",
        xy=(higher_risk, higher_return),
        xytext=(higher_risk + 0.02, higher_return - 0.015),
        arrowprops=dict(arrowstyle="->"),
    )

    plt.title("Figure 1. Stylized Mean-Variance Efficient Frontier")
    plt.xlabel("Portfolio Risk (Standard Deviation)")
    plt.ylabel("Expected Return")
    plt.grid(True, alpha=0.3)
    plt.legend()

    save_current_figure(OUTPUT_DIR / "figure_1_efficient_frontier.png")


# ============================================================
# FIGURE 2
# ============================================================

def plot_figure_2_estimation_error_fragility() -> None:
    """
    Figure 2. Stylized illustration of how estimation error can lead to
    portfolio fragility.

    The diagram shows the conceptual mechanism through which uncertain inputs
    can be amplified by optimization and result in unstable portfolio weights.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")

    boxes = [
        ((0.3, 1.4), "Uncertain inputs\nexpected returns,\nvolatilities,\ncorrelations"),
        ((2.6, 1.4), "Estimation\nerror"),
        ((4.9, 1.4), "Mean-variance\noptimization"),
        ((7.2, 1.4), "Unstable portfolio\nweights\nconcentration,\nturnover,\nsensitivity"),
        ((9.7, 1.4), "Fragile real-world\nallocation\ncostly to implement,\nunreliable out of\nsample"),
    ]

    for xy, text in boxes:
        add_box(ax, xy, 1.7, 1.2, text, fontsize=8.5)

    arrow_y = 2.0
    arrow_pairs = [
        ((2.0, arrow_y), (2.55, arrow_y)),
        ((4.3, arrow_y), (4.85, arrow_y)),
        ((6.6, arrow_y), (7.15, arrow_y)),
        ((8.9, arrow_y), (9.65, arrow_y)),
    ]

    for start, end in arrow_pairs:
        add_arrow(ax, start, end)

    ax.text(
        6,
        0.45,
        "Small input errors can be amplified into large changes in portfolio allocation.",
        ha="center",
        va="center",
        fontsize=9,
        style="italic",
    )

    plt.title("Figure 2. Stylized Illustration of Estimation Error and Portfolio Fragility")

    save_current_figure(OUTPUT_DIR / "figure_2_estimation_error_fragility.png")


# ============================================================
# FIGURE 3
# ============================================================

def plot_figure_3_market_stress_episodes() -> None:
    """
    Figure 3. Stylized market stress episodes.

    This figure creates a stylized volatility-index path with major stress
    spikes corresponding to the Global Financial Crisis, the COVID-19 shock,
    and the 2022 inflation/rates shock.
    """
    months = np.arange(2006, 2025, 1 / 12)

    base = 14 + 2 * np.sin((months - 2006) * 2 * np.pi / 6)

    gfc_spike = 50 * np.exp(-0.5 * ((months - 2008.75) / 0.12) ** 2)
    euro_spike = 20 * np.exp(-0.5 * ((months - 2011.65) / 0.10) ** 2)
    covid_spike = 62 * np.exp(-0.5 * ((months - 2020.20) / 0.07) ** 2)
    inflation_spike = 21 * np.exp(-0.5 * ((months - 2022.75) / 0.22) ** 2)

    vix_style = base + gfc_spike + euro_spike + covid_spike + inflation_spike

    plt.figure(figsize=(11, 6))
    plt.plot(months, vix_style, linewidth=2)

    plt.title("Figure 3. Stylized Market Stress Episodes")
    plt.xlabel("Year")
    plt.ylabel("Volatility Index Level")
    plt.grid(True, alpha=0.3)

    plt.annotate(
        "Global Financial Crisis",
        xy=(2008.75, vix_style[np.argmin(np.abs(months - 2008.75))]),
        xytext=(2009.0, 62),
        arrowprops=dict(arrowstyle="->"),
    )

    plt.annotate(
        "COVID-19 Shock",
        xy=(2020.20, vix_style[np.argmin(np.abs(months - 2020.20))]),
        xytext=(2020.55, 72),
        arrowprops=dict(arrowstyle="->"),
    )

    plt.annotate(
        "Inflation / Rates Shock",
        xy=(2022.75, vix_style[np.argmin(np.abs(months - 2022.75))]),
        xytext=(2023.0, 36),
        arrowprops=dict(arrowstyle="->"),
    )

    save_current_figure(OUTPUT_DIR / "figure_3_market_stress_episodes.png")


# ============================================================
# FIGURE 5
# ============================================================

def plot_figure_5_weight_sensitivity() -> None:
    """
    Figure 5. Stylized sensitivity of mean-variance portfolio weights to
    small input changes.

    The figure contrasts a more sensitive classical mean-variance response with
    a smoother robust optimization response.
    """
    np.random.seed(42)

    input_change = np.linspace(-5, 5, 80)

    mv_response = (
        1.2
        + 0.55 * input_change ** 2
        + 2.0 * np.sin(1.4 * input_change)
        + np.random.normal(0, 1.0, size=len(input_change))
    )

    robust_response = (
        0.9
        + 0.08 * np.abs(input_change)
        + 0.25 * np.sin(1.4 * input_change)
        + np.random.normal(0, 0.25, size=len(input_change))
    )

    mv_response = np.maximum(mv_response, 0)
    robust_response = np.maximum(robust_response, 0)

    plt.figure(figsize=(10, 6))
    plt.plot(input_change, mv_response, linewidth=2, label="Mean-Variance Optimization")
    plt.plot(input_change, robust_response, linewidth=2, label="Robust Optimization")

    plt.title("Figure 5. Stylized Sensitivity of Portfolio Weights to Small Input Changes")
    plt.xlabel("Change in Expected Returns (%)")
    plt.ylabel("Mean Absolute Change in Portfolio Weights (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    save_current_figure(OUTPUT_DIR / "figure_5_weight_sensitivity.png")


# ============================================================
# FIGURE 6
# ============================================================

def plot_figure_6_var_cvar_tail() -> None:
    """
    Figure 6. Stylized illustration of VaR and CVaR in the loss tail.

    The figure shows a stylized loss distribution, the VaR threshold, and
    the CVaR region beyond the VaR cutoff.
    """
    x = np.linspace(-3.5, 3.5, 600)
    density = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x ** 2)

    var_level = 1.65
    tail_mask = x >= var_level

    cvar_point = x[tail_mask].mean()

    plt.figure(figsize=(10, 6))
    plt.plot(x, density, linewidth=2, label="Loss Distribution")
    plt.fill_between(x[tail_mask], density[tail_mask], alpha=0.3, label="Tail Loss Region")

    plt.axvline(var_level, linestyle="--", linewidth=2, label="VaR Threshold")
    plt.axvline(cvar_point, linestyle=":", linewidth=2, label="CVaR Average Tail Loss")

    plt.annotate(
        "VaR",
        xy=(var_level, 0.12),
        xytext=(var_level - 1.0, 0.20),
        arrowprops=dict(arrowstyle="->"),
    )

    plt.annotate(
        "CVaR\naverage loss beyond VaR",
        xy=(cvar_point, 0.05),
        xytext=(cvar_point + 0.25, 0.18),
        arrowprops=dict(arrowstyle="->"),
    )

    plt.title("Figure 6. Stylized Illustration of VaR and CVaR in the Loss Tail")
    plt.xlabel("Portfolio Loss")
    plt.ylabel("Probability Density")
    plt.grid(True, alpha=0.3)
    plt.legend()

    save_current_figure(OUTPUT_DIR / "figure_6_var_cvar_tail.png")


# ============================================================
# FIGURE 7
# ============================================================

def plot_figure_7_classical_vs_robust() -> None:
    """
    Figure 7. Stylized comparison of classical and robust portfolio optimization
    under parameter uncertainty.

    The diagram compares a classical optimizer, which relies on one point
    estimate, with a robust optimizer, which considers a range of plausible
    parameter values.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # Left side: classical optimization
    add_box(ax, (0.6, 2.8), 2.4, 1.0, "Classical Optimization\nsingle estimated input", fontsize=9)
    ax.scatter([1.8], [2.0], s=80)
    ax.text(1.8, 1.65, "Point estimate", ha="center", fontsize=8)
    add_arrow(ax, (1.8, 2.8), (1.8, 2.12))
    add_arrow(ax, (3.1, 3.3), (4.3, 3.3))
    add_box(ax, (4.4, 2.8), 2.2, 1.0, "Portfolio optimal\nfor estimated scenario", fontsize=9)

    # Right side: robust optimization
    add_box(ax, (0.6, 0.6), 2.4, 1.0, "Robust Optimization\nuncertain inputs", fontsize=9)

    circle = Circle((1.8, 0.0), 0.55, fill=False, linewidth=1.5)
    ax.add_patch(circle)
    ax.scatter([1.8], [0.0], s=60)
    ax.text(1.8, -0.55, "Uncertainty set", ha="center", fontsize=8)

    add_arrow(ax, (1.8, 0.6), (1.8, 0.12))
    add_arrow(ax, (3.1, 1.1), (4.3, 1.1))
    add_box(ax, (4.4, 0.6), 2.2, 1.0, "Portfolio stable\nacross plausible scenarios", fontsize=9)

    # Outcome comparison
    add_arrow(ax, (6.8, 3.3), (8.0, 3.3))
    add_arrow(ax, (6.8, 1.1), (8.0, 1.1))

    add_box(ax, (8.1, 2.8), 2.8, 1.0, "Higher sensitivity\nto estimation error", fontsize=9)
    add_box(ax, (8.1, 0.6), 2.8, 1.0, "Lower sensitivity\nand more stable allocation", fontsize=9)

    ax.text(
        6.0,
        4.55,
        "Classical vs. Robust Portfolio Optimization Under Parameter Uncertainty",
        ha="center",
        va="center",
        fontsize=13,
        weight="bold",
    )

    save_current_figure(OUTPUT_DIR / "figure_7_classical_vs_robust.png")


# ============================================================
# RUN EVERYTHING
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating Figure 1...")
    plot_figure_1_efficient_frontier()

    print("Generating Figure 2...")
    plot_figure_2_estimation_error_fragility()

    print("Generating Figure 3...")
    plot_figure_3_market_stress_episodes()

    print("Generating Figure 5...")
    plot_figure_5_weight_sensitivity()

    print("Generating Figure 6...")
    plot_figure_6_var_cvar_tail()

    print("Generating Figure 7...")
    plot_figure_7_classical_vs_robust()

    print(f"All stylized figures saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

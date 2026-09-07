import os
import json
import vl_convert as vlc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from meridian.model.model import Meridian
from meridian.analysis.optimizer import BudgetOptimizer


def run_budget_optimization(mmm: Meridian, budget: float | None = None, 
    spend_constraint_lower: float = 0.5,  # each channel can go down to 50% of historical spend
    spend_constraint_upper: float = 2.0,  # each channel can go up to 200% of historical spend
    output_dir: str = "../outputs/charts",
    table_dir: str = "../outputs/tables",
) -> None:
    
    # BudgetOptimizer wraps the fitted Meridian model
    optimizer = BudgetOptimizer(mmm)

    # if no budget is specified, use the total historical spend as constraint
    results = optimizer.optimize(
        use_posterior=True,
        fixed_budget=True, # we are maximizing revenue under a fixed total budget
        budget=budget, # use historical total spend
        spend_constraint_lower=spend_constraint_lower, 
        spend_constraint_upper=spend_constraint_upper,
        use_kpi=True,  # optimize in NET_REVENUE space
        confidence_level=0.9, # 90% credible intervals on results
    )

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    # generate plots with proper channel labels and comparison table
    comparison = plot_optimization_results(results, mmm, output_dir)  # ← sostituisce le 4 _save_altair

    # save comparison table as csv
    csv_path = os.path.join(table_dir, "optimization_results.csv")
    comparison.to_csv(csv_path)
    print(f"Saved: {csv_path}")

    return results

def plot_optimization_results(results, mmm: Meridian, output_dir: str = "../outputs/charts") -> None:
    os.makedirs(output_dir, exist_ok=True)

    # extract optimized and historical spend/outcome per channel (median only)
    opt_df    = results.optimized_data.to_dataframe().reset_index()
    nonopt_df = results.nonoptimized_data.to_dataframe().reset_index()

    opt_med    = opt_df[opt_df["metric"] == "median"].set_index("channel")
    nonopt_med = nonopt_df[nonopt_df["metric"] == "median"].set_index("channel")

    channels = list(opt_med.index)
    x        = np.arange(len(channels))
    width    = 0.35

    # Plot spend comparison (historical vs optimized)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars_hist = ax.bar(x - width/2, nonopt_med["spend"],  width, label="Historical", color="#4C72B0")
    bars_opt  = ax.bar(x + width/2, opt_med["spend"],     width, label="Optimized",  color="#55A868")

    # value labels on bars
    for bar in bars_hist:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + nonopt_med["spend"].max()*0.01,
                f'{bar.get_height():,.0f}', ha='center', va='bottom', fontsize=8)
    for bar in bars_opt:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + nonopt_med["spend"].max()*0.01,
                f'{bar.get_height():,.0f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(channels, rotation=30, ha='right')
    ax.set_ylabel("Spend ($)")
    ax.set_title("Budget Allocation — Historical vs Optimized")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/optimization_budget_allocation.jpg", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir}/optimization_budget_allocation.jpg")

    # Plot spend delta (optimized - historical)
    spend_delta = opt_med["spend"] - nonopt_med["spend"]
    colors = ["#55A868" if v >= 0 else "#C44E52" for v in spend_delta]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(channels, spend_delta, color=colors)
    for bar, val in zip(bars, spend_delta):
        ax.text(bar.get_x() + bar.get_width()/2,
                val + spend_delta.abs().max()*0.01 if val >= 0 else val - spend_delta.abs().max()*0.03,
                f'{val:+,.0f}', ha='center', va='bottom', fontsize=8)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticklabels(channels, rotation=30, ha='right')
    ax.set_ylabel("Spend Delta ($)")
    ax.set_title("Spend Delta — Optimized vs Historical")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/optimization_spend_delta.jpg", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir}/optimization_spend_delta.jpg")

    # Plot incremental outcome delta (optimized - historical)
    outcome_delta = opt_med["incremental_outcome"] - nonopt_med["incremental_outcome"]
    colors        = ["#55A868" if v >= 0 else "#C44E52" for v in outcome_delta]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(channels, outcome_delta, color=colors)
    for bar, val in zip(bars, outcome_delta):
        ax.text(bar.get_x() + bar.get_width()/2,
                val + outcome_delta.abs().max()*0.01 if val >= 0 else val - outcome_delta.abs().max()*0.03,
                f'{val:+,.0f}', ha='center', va='bottom', fontsize=8)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticklabels(channels, rotation=30, ha='right')
    ax.set_ylabel("Incremental Revenue Delta ($)")
    ax.set_title("Incremental Outcome Delta — Optimized vs Historical")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/optimization_outcome_delta.jpg", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir}/optimization_outcome_delta.jpg")

    # Table of the full comparison historical vs optimized
    comparison = pd.DataFrame({
        "spend_historical": nonopt_med["spend"],
        "spend_optimized": opt_med["spend"],
        "spend_delta": spend_delta,
        "spend_delta_pct": (spend_delta / nonopt_med["spend"] * 100).round(1),
        "revenue_historical": nonopt_med["incremental_outcome"],
        "revenue_optimized": opt_med["incremental_outcome"],
        "revenue_delta": outcome_delta,
        "roi_historical": nonopt_med["roi"],
        "roi_optimized": opt_med["roi"],
        "mroi_optimized": opt_med["mroi"],
    })

    print("\nFull comparison — Historical vs Optimized:")
    print(comparison.to_string())

    return comparison


def save_optimization_results(results, table_dir: str = "../outputs/tables") -> None:
    os.makedirs(table_dir, exist_ok=True)

    # save optimized and non-optimized allocation tables as csv
    opt_df     = results.optimized_data.to_dataframe().reset_index()
    nonopt_df  = results.nonoptimized_data.to_dataframe().reset_index()

    opt_path    = os.path.join(table_dir, "optimization_optimized.csv")
    nonopt_path = os.path.join(table_dir, "optimization_nonoptimized.csv")

    opt_df.to_csv(opt_path, index=False)
    nonopt_df.to_csv(nonopt_path, index=False)

    print(f"Optimization results saved to {table_dir}")


def load_optimization_results(table_dir: str = "../outputs/tables") -> tuple[pd.DataFrame, pd.DataFrame]:
    opt_path    = os.path.join(table_dir, "optimization_optimized.csv")
    nonopt_path = os.path.join(table_dir, "optimization_nonoptimized.csv")

    opt_df    = pd.read_csv(opt_path)
    nonopt_df = pd.read_csv(nonopt_path)

    print(f"Optimization results loaded from {table_dir}")
    return opt_df, nonopt_df


def print_optimization_summary(opt_df: pd.DataFrame, nonopt_df: pd.DataFrame) -> None:
    opt_med = opt_df[opt_df["metric"] == "median"].set_index("channel")
    nonopt_med = nonopt_df[nonopt_df["metric"] == "median"].set_index("channel")

    # total revenue and ROI before and after optimization
    total_spend = nonopt_med["spend"].sum()  # same for both, budget is fixed
    total_revenue_hist = nonopt_med["incremental_outcome"].sum()
    total_revenue_opt = opt_med["incremental_outcome"].sum()
    revenue_gain = total_revenue_opt - total_revenue_hist
    revenue_gain_pct = revenue_gain / total_revenue_hist * 100

    roi_hist = total_revenue_hist / total_spend
    roi_opt = total_revenue_opt / total_spend

    print(f"Total budget (fixed) : ${total_spend:>12,.0f}")
    print(f"Revenue before       : ${total_revenue_hist:>12,.0f} ROI: {roi_hist:.2f}")
    print(f"Revenue after        : ${total_revenue_opt:>12,.0f} ROI: {roi_opt:.2f}")
    print(f"Gain                 : ${revenue_gain:>12,.0f} (+{revenue_gain_pct:.1f}%)")

    # per-channel spend changes
    print("\nBudget reallocation:")
    for ch in opt_med.index:
        spend_hist = nonopt_med.loc[ch, "spend"]
        spend_opt = opt_med.loc[ch, "spend"]
        delta = spend_opt - spend_hist
        direction = "Increase" if delta > 0 else "Decrease"
        print(f"{ch:<20} {direction} ${delta:>+10,.0f} (from ${spend_hist:>10,.0f} to ${spend_opt:>10,.0f})")
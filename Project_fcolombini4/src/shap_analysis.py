import os, math
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
from meridian.model.model import Meridian
from meridian.analysis import analyzer



def build_shap_input(mmm: Meridian) ->  tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    # Analyzer instance to query the fitted model
    az = analyzer.Analyzer(mmm)

    #Calculate the incremental outcome tensor for all geos (states), times, and channels (the presents)
    inc_tensor = az.incremental_outcome(  #incremental_outcome returns the revenue contribution of each channel per timestep
        use_posterior=True,
        aggregate_geos=False,  # aggregate_geos=False and aggregate_times=False keep all dimensions intact
        aggregate_times=False,
        use_kpi=True,
    ).numpy()

    # Collapse chains to obtain the most representative contribution per geo,
    contributions = np.median(inc_tensor, axis=(0, 1))

    # Compute 95% confidence intervals for the contributions
    contributions_lower = np.percentile(inc_tensor, 2.5,  axis=(0, 1))
    contributions_upper = np.percentile(inc_tensor, 97.5, axis=(0, 1))


    # as before the expected_outcome returns the total predicted revenue per timestep
    exp_tensor = az.expected_outcome(
        use_posterior=True,
        aggregate_geos=False,
        aggregate_times=False,
        use_kpi=True,
    ).numpy()

    # Here we compute the baseline residual: total predicted - sum(channel contributions)
    baseline = np.median(exp_tensor, axis=(0, 1))

    # extract the names
    channels = list(mmm.input_data.media_channel.values)
    geos     = list(mmm.input_data.geo.values)

    return contributions, contributions_lower, contributions_upper, baseline, channels, geos



def get_geo_contributions(contributions: np.ndarray, contributions_lower: np.ndarray, contributions_upper: np.ndarray, baseline: np.ndarray, geos: list[str], geo: str,) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Select only the geo index to extract the right slice for the given geo
    geo_idx = geos.index(geo)
    return contributions[geo_idx], contributions_lower[geo_idx], contributions_upper[geo_idx], baseline[geo_idx]


def compute_mean_absolute_contribution(geo_contributions: np.ndarray, channels: list[str],) -> dict[str, float]:
    # Compute mean absolute contribution per channel over all timesteps
    # This is the direct analogue of mean (SHAP value) used in SHAP bar plots
    mean_abs = np.abs(geo_contributions).mean(axis=0)
    return dict(zip(channels, mean_abs.tolist()))


def print_credible_intervals(geo_contributions: np.ndarray, geo_contributions_lower: np.ndarray, geo_contributions_upper: np.ndarray, channels: list[str], geo: str,
) -> pd.DataFrame:
    # compute mean contribution and credible interval bounds across all timesteps
    median = geo_contributions.mean(axis=0)
    lower  = geo_contributions_lower.mean(axis=0)
    upper  = geo_contributions_upper.mean(axis=0)

    # build a readable summary table per channel
    df = pd.DataFrame({
        "channel": channels,
        "median": median,
        "ci_2.5%": lower,
        "ci_97.5%": upper,
        "ci_width": upper - lower,  # wider = more uncertainty around that channel's contribution
    }).set_index("channel")

    print(f"\nBayesian Credible Intervals (95% CI) — {geo}")
    print(df.to_string())
    return df



################ PLOTS ################

def plot_shap_importance(geo_contributions: np.ndarray, channels: list[str], geo: str,) -> None:
    output_dir = "../outputs/charts"
    # Compute mean absolute contribution per channel over all timesteps — analogue of mean (SHAP value)
    importance = np.abs(geo_contributions).mean(axis=0)

    # Sort channels by importance for readability (least → most important, bottom → top)
    sorted_idx = np.argsort(importance)
    sorted_imp = importance[sorted_idx]
    sorted_ch  = [channels[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_ch)))
    bars = ax.barh(sorted_ch, sorted_imp, color=colors)

    # Add value labels at the end of each bar
    for bar, val in zip(bars, sorted_imp):
        ax.text(
            bar.get_width() + sorted_imp.max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,.0f}",
            va="center", fontsize=9,
        )

    ax.set_xlabel("Mean Absolute SHAP Value")
    ax.set_title(f"SHAP Feature Importance - {geo}")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/shap_importance_{geo.lower()}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP importance plot saved for {geo}")


def plot_shap_summary(geo_contributions: np.ndarray, channels: list[str],geo: str,) -> None:
    output_dir = "../outputs/charts"
    # Here each dot is a timestep: X = contribution value, Y = channel (jittered), color = magnitude
    fig, ax = plt.subplots(figsize=(9, 5))
    n_ch = len(channels)

    for i, ch in enumerate(channels):
        vals = geo_contributions[:, i]  # contributions for this channel

        # Vertical jitter to avoid overplotting across timesteps
        jitter = np.random.default_rng(seed=i).uniform(-0.3, 0.3, size=len(vals))

        # Normalize contribution values to [0,1] for colour mapping
        v_norm = (vals - vals.min()) / (vals.ptp() + 1e-9)
        colors = cm.RdBu_r(v_norm)

        ax.scatter(
            vals, np.full_like(vals, i) + jitter,
            c=colors, s=12, alpha=0.6, linewidths=0,
        )

    ax.set_yticks(range(n_ch))
    ax.set_yticklabels(channels)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP Value (Revenue contribution per timestep)")
    ax.set_title(f"SHAP Summary Plot — {geo}")
    ax.spines[["top", "right"]].set_visible(False)

    # Colorbar go from low (blue) to high (red) contribution magnitude
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", pad=0.01)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])
    cbar.set_label("Contribution magnitude")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/shap_summary_{geo.lower()}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary plot saved for {geo}")


def plot_shap_dependence(mmm: Meridian, geo_contributions: np.ndarray, channels: list[str], geo: str,) -> None:
    output_dir = "../outputs/charts"
    # Select the geo index to extract the right media input data
    geos    = list(mmm.input_data.geo.values)
    geo_idx = geos.index(geo)
    media_values = mmm.input_data.media.values[geo_idx]

    os.makedirs(output_dir, exist_ok=True)

    for i, ch in enumerate(channels):
        x_vals = media_values[:, i]  # media input (clicks/impressions) for this channel
        y_vals = geo_contributions[:, i]  # revenue contribution for this channel

        # Colour by contribution magnitude (low = blue, high = red)
        v_norm = (y_vals - y_vals.min()) / (y_vals.ptp() + 1e-9)
        colors = cm.RdBu_r(v_norm)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(x_vals, y_vals, c=colors, s=18, alpha=0.7, linewidths=0)

        ax.set_xlabel(f"{ch} — Media Input (clicks/impressions)")
        ax.set_ylabel(f"{ch} — SHAP Value (Revenue Contribution)")
        ax.set_title(f"SHAP Dependence — {ch} — {geo}")
        ax.spines[["top", "right"]].set_visible(False)

        sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.01)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["Low contrib.", "High contrib."])

        plt.tight_layout()
        safe_ch = ch.lower().replace(" ", "_")
        plt.savefig(
            f"{output_dir}/shap_dependence_{safe_ch}_{geo.lower()}.png",
            dpi=300, bbox_inches="tight"
        )
        plt.close()

    print(f"SHAP dependence plots saved for {geo}")



def combine_geo_plots(geo: str) -> tuple[Path, list[Path]]:
    output_dir=Path("../outputs/charts")
    output_file = output_dir / f"SHAP_combined_results_{geo}.png"
    plot_files = sorted([p for p in output_dir.glob(f"*{geo}*.png") if p.name != output_file.name])

    if not plot_files:
        print(f"No plots found for geo: {geo}")
        return None, None

    
    # Calculate grid dimensions
    num_plots = len(plot_files)
    cols = math.ceil(math.sqrt(num_plots))
    rows = math.ceil(num_plots / cols)
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))

    # Normalize axes to a 1D list of Axes
    if isinstance(axes, np.ndarray):
        axes = axes.ravel()
    else:
        axes = np.array([axes])
    
    # Load and display each plot
    for idx, plot_file in enumerate(plot_files):
        img = mpimg.imread(plot_file)
        axes[idx].imshow(img)
        axes[idx].set_title(plot_file.stem, fontsize=10)
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(num_plots, len(axes)):
        axes[idx].axis('off')

    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

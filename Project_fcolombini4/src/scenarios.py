from meridian.analysis.analyzer import Analyzer, DataTensors
import numpy as np
import tensorflow as tf
import pandas as pd
from meridian.model.model import Meridian

def build_scaled_data(mmm, multipliers: dict):
    # Here we have to create the scaled format, to apply the same trained function to the new data
    # extract the channel order used internally by Meridian
    channel_names = mmm.input_data.media_channel.values

    # check for the excact match of channel names, raise error if any invalid keys are found
    invalid_keys = set(multipliers.keys()) - set(channel_names)
    if invalid_keys:
        raise ValueError(f"Channels not found in model: {invalid_keys}. Valid channels: {list(channel_names)}")

    # multiply each channel by its corresponding multiplier, and channels that are not specified keep multiplier 1.0
    multiplier_array = np.array([multipliers.get(ch, 1.0) for ch in channel_names], dtype=np.float32)

    # original tensors shape for media and spend
    original_media = mmm.input_data.media.values.astype(np.float32)
    original_spend = mmm.input_data.media_spend.values.astype(np.float32)

    # scale per channel, same multiplier across all geos and times
    scaled_media = original_media * multiplier_array[np.newaxis, np.newaxis, :]
    scaled_spend = original_spend * multiplier_array[np.newaxis, np.newaxis, :]

    # create new dimensions for the scaled media and spend tensors, to match the expected input shape of the model
    scaled_media_tensor = tf.convert_to_tensor(scaled_media, dtype=tf.float32)
    scaled_spend_tensor = tf.convert_to_tensor(scaled_spend, dtype=tf.float32)

    # create an object of DataTensors with the scaled media and spend tensors
    new_data = DataTensors(media=scaled_media_tensor, media_spend=scaled_spend_tensor)

    # save resolved multipliers, useful later for logging what each scenario actually did
    resolved_multipliers = dict(zip(channel_names, multiplier_array.tolist()))
    return new_data, resolved_multipliers




def simulate_scenario(mmm, multipliers: dict, scenario_name: str) -> dict:
    az = Analyzer(mmm)

    # historical distribution of total revenue, to compare with the scenario distribution
    baseline_tensor = az.incremental_outcome().numpy()

    # sum total revenue from the single channel contributions baseline
    baseline_total = baseline_tensor.sum(axis=-1)

    # full distribution of total revenue for the baseline, to compute confidence intervals
    baseline_samples = baseline_total.reshape(-1)

    # same logic here, but with different data. This scale the historical data according to the given multipliers
    scenario_data, resolved_multipliers = build_scaled_data(mmm, multipliers)
    scenario_tensor = az.incremental_outcome(new_data=scenario_data).numpy()
    scenario_total = scenario_tensor.sum(axis=-1)
    scenario_samples = scenario_total.reshape(-1)

    # evaluate the metrics needed for the scenario report, both baseline and scenario distributions
    baseline_median = np.percentile(baseline_samples, 50)
    baseline_lower = np.percentile(baseline_samples, 2.5)
    baseline_upper = np.percentile(baseline_samples, 97.5)

    scenario_median = np.percentile(scenario_samples, 50)
    scenario_lower = np.percentile(scenario_samples, 2.5)
    scenario_upper = np.percentile(scenario_samples, 97.5)

    # compute the median difference and percentage change between the scenario and baseline
    delta_abs = scenario_median - baseline_median
    delta_pct = delta_abs / baseline_median * 100

    
    # fixed result columns, same for every scenario
    row = {
        "scenario_name": scenario_name,
        "baseline_median": baseline_median,
        "baseline_ci_low": baseline_lower,
        "baseline_ci_high": baseline_upper,
        "scenario_median": scenario_median,
        "scenario_ci_low": scenario_lower,
        "scenario_ci_high": scenario_upper,
        "delta_abs": delta_abs,
        "delta_pct": delta_pct,
    }

    # one extra column per channel, so this works regardless of how many channels exist
    for channel, mult in resolved_multipliers.items():
        row[f"mult_{channel}"] = mult

    return pd.DataFrame([row])


def get_channel_metrics(mmm: Meridian, 
                         roi_table: pd.DataFrame, 
                         mroi_table: pd.DataFrame,
                         mroi_threshold: float = 1.0) -> pd.DataFrame:
    
    # total historical spend per channel, summed across geo and time
    channels = mmm.input_data.media_channel.values
    spend = mmm.input_data.media_spend.values.sum(axis=(0, 1))

    metrics = pd.DataFrame({
        "spend_total": spend,
        "roi_median": roi_table["ROI Median"],
        "mroi_median": mroi_table["mROI Median"],
        "is_saturated": mroi_table["mROI Median"] < mroi_threshold,
    }, index=channels)
    metrics.index.name = "channel"

    return metrics



def strategy_equal_allocation(mmm, metrics: pd.DataFrame, pct: float = 0.20, df_cleaned: pd.DataFrame = None) -> pd.DataFrame:
    # First of all we must calculate raw spend per each channel (summed across geos and time)
    spend = metrics["spend_total"].values
    total_spend = spend.sum()
    n_channels = len(spend)
    channel_names = metrics.index.tolist()

    # +20% of total budget distributed equally across all channels
    extra_budget = total_spend * 0.20
    extra_per_channel = extra_budget / n_channels

    new_spend = spend + extra_per_channel
    blended_mult = new_spend / spend

    # build the multipliers dict expected by simulate_scenario
    multipliers = dict(zip(channel_names, blended_mult.tolist()))

    # run the simulation trough the function written before
    result_df = simulate_scenario(mmm, multipliers, scenario_name="equal_allocation")

    # add derived KPIs trought the original dataframe, to compute incremental purchases and incremental ROI
    avg_price = ((df_cleaned['ALL_PURCHASES_ORIGINAL_PRICE'] - df_cleaned['ALL_PURCHASES_GROSS_DISCOUNT']).sum()
    / df_cleaned['ALL_PURCHASES'].sum())
    result_df["incremental_purchases"] = result_df["delta_abs"] / avg_price
    result_df["incremental_roi"] = result_df["delta_abs"] / extra_budget

    return result_df


def strategy_mroi_optimized(mmm, metrics: pd.DataFrame, pct: float = 0.20, df_cleaned: pd.DataFrame = None) -> pd.DataFrame:
    spend         = metrics["spend_total"].values
    mroi          = metrics["mroi_median"].values
    channel_names = metrics.index.tolist()

    total_spend  = spend.sum()
    extra_budget = total_spend * pct  # same fixed budget as scenario 1

    # allocate the entire extra budget to the single best channel by mROI
    best_channel_idx = np.argmax(mroi)
    best_channel     = channel_names[best_channel_idx]
    print(f"Best channel by mROI: {best_channel} (mROI = {mroi[best_channel_idx]:.3f})")

    new_spend                    = spend.copy().astype(float)
    new_spend[best_channel_idx] += extra_budget
    blended_mult                 = new_spend / spend

    multipliers = dict(zip(channel_names, blended_mult.tolist()))
    result_df   = simulate_scenario(mmm, multipliers, scenario_name="mroi_optimized")

    avg_price = ((df_cleaned['ALL_PURCHASES_ORIGINAL_PRICE'] - df_cleaned['ALL_PURCHASES_GROSS_DISCOUNT']).sum()
                 / df_cleaned['ALL_PURCHASES'].sum())
    result_df["incremental_purchases"] = result_df["delta_abs"] / avg_price
    result_df["incremental_roi"]       = result_df["delta_abs"] / extra_budget

    return result_df


def strategy_saturation_aware(mmm, metrics: pd.DataFrame, pct: float = 0.20, df_cleaned: pd.DataFrame = None) -> pd.DataFrame:
    spend         = metrics["spend_total"].values
    mroi          = metrics["mroi_median"].values
    channel_names = metrics.index.tolist()

    total_spend  = spend.sum()
    extra_budget = total_spend * pct  # same fixed budget as the other scenarios

    saturated_mask = mroi < 1.0
    efficient_mask = ~saturated_mask

    if efficient_mask.sum() == 0:
        raise ValueError("All channels are saturated — cannot run saturation_aware scenario.")

    # -30% on saturated channels
    new_spend                   = spend.copy().astype(float)
    new_spend[saturated_mask]  *= 0.70
    freed_budget                = (spend[saturated_mask] - new_spend[saturated_mask]).sum()

    # freed budget + extra budget both allocated to efficient channels weighted by mROI
    total_to_redistribute     = freed_budget + extra_budget
    efficient_weights         = mroi[efficient_mask] / mroi[efficient_mask].sum()
    new_spend[efficient_mask] += total_to_redistribute * efficient_weights

    blended_mult = new_spend / spend
    multipliers  = dict(zip(channel_names, blended_mult.tolist()))
    result_df    = simulate_scenario(mmm, multipliers, scenario_name="saturation_aware")

    avg_price = ((df_cleaned['ALL_PURCHASES_ORIGINAL_PRICE'] - df_cleaned['ALL_PURCHASES_GROSS_DISCOUNT']).sum()
                 / df_cleaned['ALL_PURCHASES'].sum())
    result_df["incremental_purchases"] = result_df["delta_abs"] / avg_price
    # incremental ROI computed on the extra budget only (freed budget is internal reallocation)
    result_df["incremental_roi"]       = result_df["delta_abs"] / extra_budget

    return result_df
from matplotlib import ticker
import pandas as pd
import numpy as np
import vl_convert as vlc
import json
import matplotlib.pyplot as plt
import os
from meridian.data.load import CoordToColumns, DataFrameDataLoader
from meridian.model.model import Meridian
from meridian.model import spec
from meridian.model import prior_distribution
from meridian.model import model
from meridian.analysis import analyzer, visualizer
import tensorflow_probability as tfp



def prepare_mmm_dataset(df: pd.DataFrame, media_info: dict) -> pd.DataFrame:
    # Here we drop metadata and unused targets to clean the dataset
    cols_to_drop = [
        'MMM_TIMESERIES_ID', 
        'ORGANISATION_ID', 
        'ORGANISATION_VERTICAL',
        'ORGANISATION_SUBVERTICAL', 
        'ORGANISATION_MARKETING_SOURCES',
        'ORGANISATION_PRIMARY_TERRITORY_NAME', 
        'CURRENCY_CODE',
        'FIRST_PURCHASES', 
        'FIRST_PURCHASES_UNITS',  
        'ALL_PURCHASES',
        'ALL_PURCHASES_UNITS',
        'FIRST_PURCHASES_ORIGINAL_PRICE', 
        'FIRST_PURCHASES_GROSS_DISCOUNT',
    ]
    
    df = df.drop(columns=cols_to_drop)
    
    # Aggregate at daily level by summing across territories
    df = df.sort_values('DATE_DAY').reset_index(drop=True)
    df['GEO'] = df['TERRITORY_NAME']
    df['DATE_DAY'] = pd.to_datetime(df['DATE_DAY'])

    # net revenue is the target variable for MMM as it captures the true economic value of purchases after discounts
    df['NET_REVENUE'] = df['ALL_PURCHASES_ORIGINAL_PRICE'] - df['ALL_PURCHASES_GROSS_DISCOUNT']
    
    # Discount rate varies by geo and date so we add a valid control for geo-level model
    df['DISCOUNT_RATE'] = (
        df['ALL_PURCHASES_GROSS_DISCOUNT'] / 
        df['ALL_PURCHASES_ORIGINAL_PRICE']
    ).fillna(0)

    # Media variables and control variables are identified based on media_info configuration
    # This was created with the function identify_media_channels() that analyzes column names and metadata to classify media and control variables
    # Paid media spend columns are the primary inputs for MMM
    media_cols = [col for col in media_info['spend_cols'] if col in df.columns]
    # Control variables
    # Non-paid clicks capturing organic and direct demand effects
    control_cols = [col for col in media_info['clicks_cols'] if col in df.columns]

    # Fill missing values with 0 for both media and control variables (handling missing values)
    # In this way, missing values = channel inactive or no traffic recorded that day
    df[media_cols + control_cols] = df[media_cols + control_cols].fillna(0)
    
    #### Time variables
    # The trend is the progressive index capturing long-term business growth independent of marketing
    df['TREND'] = range(len(df))
    
    # Day of week with 1=Monday, 7=Sunday that captures weekly purchase patterns
    df['DAY_OF_WEEK'] = df['DATE_DAY'].dt.dayofweek + 1
    
    # Month captures seasonal purchase patterns
    df['MONTH'] = df['DATE_DAY'].dt.month
    
    return df


def temporal_train_test_split(df: pd.DataFrame):
    train_size = 0.75 #3/4 of the data is used for training, 1/4 for validation
    # Here we sort by date by chronological order
    df = df.sort_values('DATE_DAY').reset_index(drop=True)
    
    # Then compute split index
    split_idx = int(len(df) * train_size)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    # And print the results
    print(f"Train set: {train_df['DATE_DAY'].min()} to {train_df['DATE_DAY'].max()} ({len(train_df)} rows)")
    print(f"Validation set: {test_df['DATE_DAY'].min()} to {test_df['DATE_DAY'].max()} ({len(test_df)} rows)")

    # Validation checks on train set for meridian - NEEDED by the literature
    # Without these checks the model cannot capture seasonality, long-term trends and cross-channel interactions effectively, 
    # This can lead to a poor performance of the model and unreliable ROI estimates
    # Also couldn't run the model if the target variable or time control variables are missing
    train_days = (train_df['DATE_DAY'].max() - train_df['DATE_DAY'].min()).days
    train_weeks = train_days / 7
    spend_cols = [col for col in train_df.columns if col.endswith('_SPEND')]
    active_channels = (train_df[spend_cols].sum() > 0).sum()
    
    # Minimum 2 years of data (104 weeks) recommended for MMM to capture seasonality and long-term trends
    if train_weeks < 104:
        print(f"!!! Train set covers only {train_weeks:.0f} weeks - minimum 104 recommended")
    else:
        print(f"OK! Train set covers {train_weeks:.0f} weeks")
    
    # Minimum 3 active media channels recommended for MMM to capture cross-channel interactions and attribution effects
    if active_channels < 3:
        print(f"!!! Train set has only {active_channels} active media channels - minimum 3 recommended")
    else:
        print(f"OK! Train set has {active_channels} active media channels")

    # Check the variable existence, choosen in the previous function prepare_mmm_dataset() as target and control variables for the modeling phase. 
    if 'NET_REVENUE' not in train_df.columns:
        print("!!! Target variable NET_REVENUE not found in train set")
    else:
        print("OK! Target variable NET_REVENUE present")
    
    # Check control variables
    time_controls = ['TREND', 'DAY_OF_WEEK', 'MONTH'] # we check that the previous code run correctly and created these time control variables in the train set
    missing_controls = [col for col in time_controls if col not in train_df.columns]
    if missing_controls:
        print(f"!!! Missing control variables: {missing_controls}")
    else:
        print("OK! Time control variables present")
    
    return train_df, test_df



def align_geo_dates(df: pd.DataFrame) -> pd.DataFrame:
    # unique dates and geos in the dataset
    all_dates = df['DATE_DAY'].unique()
    all_geos = df['GEO'].unique()

    # Create the cartesian product of all geos and all dates to ensure every geo has a row for every date
    full_index = pd.MultiIndex.from_product(
        [all_geos, all_dates], names=['GEO', 'DATE_DAY']
    )
    df_full = pd.DataFrame(index=full_index).reset_index()

    # Then merge with original data, missing rows become NaN
    df_aligned = df_full.merge(df, on=['GEO', 'DATE_DAY'], how='left')

    # Fill time-only controls (same value for all geos on the same date)
    time_only_cols = ['TREND', 'DAY_OF_WEEK', 'MONTH']
    for col in time_only_cols:
        if col in df_aligned.columns:
            date_vals = df.drop_duplicates('DATE_DAY').set_index('DATE_DAY')[col]
            df_aligned[col] = df_aligned['DATE_DAY'].map(date_vals)

    # NET_REVENUE is geo-specific so we can fill missing with 0 (because there are no sales that day in that geo)
    if 'NET_REVENUE' in df_aligned.columns:
        df_aligned['NET_REVENUE'] = df_aligned['NET_REVENUE'].fillna(0)

    # Fill all remaining numeric columns with 0 (inactive channel or missing day)
    num_cols = df_aligned.select_dtypes(include='number').columns.tolist()
    df_aligned[num_cols] = df_aligned[num_cols].fillna(0)

    # Then remove dates where any geo has NET_REVENUE is equal to 0
    zero_dates = df_aligned[df_aligned['NET_REVENUE'] == 0]['DATE_DAY'].unique()
    df_aligned = df_aligned[~df_aligned['DATE_DAY'].isin(zero_dates)].reset_index(drop=True) # avoid results = infinity in the ROI calculation, 
    # If we divide by 0, ROI = revenue/0 = infinity. So we remove rows with NET_REVENUE = 0 to avoid this issue.

    print(f"Geos: {list(all_geos)}")
    for geo in all_geos:
        n = (df_aligned['GEO'] == geo).sum()
        print(f"  {geo}: {n} rows")

    return df_aligned



def compute_empirical_roi(df: pd.DataFrame) -> float:
    # as before, we identify the spend columns based on the suffix '_SPEND'
    spend_cols = [col for col in df.columns if col.endswith('_SPEND')]
    total_spend = df[spend_cols].sum().sum() #total spend across all channels and geos
    total_revenue = df['NET_REVENUE'].sum() # total revenue across all geos
    empirical_roi = total_revenue / total_spend #calculate the empirical ROI as total revenue divided by total spend

    # print the results for each geo to see if there are significant differences in ROI across geos
    print(f"Total spend: {total_spend:.2f}")
    print(f"Total revenue: {total_revenue:.2f}")
    print(f"Empirical ROI (global): {empirical_roi:.4f}")
    for geo in df['GEO'].unique():
        geo_df = df[df['GEO'] == geo]
        geo_roi = geo_df['NET_REVENUE'].sum() / geo_df[spend_cols].sum().sum() # weighted mean ROI across all channels in that geo
        print(f"  {geo} ROI: {geo_roi:.4f}")
    return empirical_roi


def build_meridian_input_data(df: pd.DataFrame, media_info: dict):
    # As done in the preprocessing part, identify active spend columns and corresponding channels
    # We must keep only channels that have BOTH spend and clicks columns in the dataframe
    paired = [
        (col, col.replace('_SPEND', ''))
        for col in media_info['spend_cols']
        if col in df.columns and f"{col.replace('_SPEND', '')}_CLICKS" in df.columns
    ]
    spend_cols  = [p[0] for p in paired]
    channels    = [p[1] for p in paired]
    clicks_cols = [f'{ch}_CLICKS' for ch in channels]
    control_columns = ['DISCOUNT_RATE']
    

    # CoordToColumns maps each dataframe column to its Meridian coordinate role
    coord_to_columns = CoordToColumns(
        time='DATE_DAY', #date column
        geo='GEO', #geography column of the 2 territories
        kpi='NET_REVENUE', #target variable
        controls=control_columns, # time control variables are excluded because Meridian already capture internal knot
        media=clicks_cols, #exposure metric (clicks)
        media_spend=spend_cols, # paid spend per channel
    ) 

    # DataFrameDataLoader converts the pandas DataFrame into a Meridian InputData object
    loader = DataFrameDataLoader(
        df=df,
        coord_to_columns=coord_to_columns,
        kpi_type='revenue', #KPI is a  monetary revenue, not a count of purchase like ALL_PURCHASES
        media_to_channel=dict(zip(clicks_cols, channels)), #maps each clicks column to its channel name
        media_spend_to_channel=dict(zip(spend_cols, channels)), # maps each spend column to its channel name
    )
    return loader.load()



def build_holdout_id(df: pd.DataFrame, train_ratio: float = 0.75) -> np.ndarray:
    # Meridian requires all time periods to be present in the input data, so we cannot simply pass a train-only DataFrame 
    # Instead, we use an holdout_id mechanism where we provide a boolean array that tells Meridian which time periods to exclude from the
    # posterior likelihood during training, while still using their media data for adstock.
    geos  = sorted(df["GEO"].unique())
    times = sorted(df["DATE_DAY"].unique())

    n_geos  = len(geos)
    n_times = len(times)
    split_idx = int(n_times * train_ratio)

    # Build a boolean mask, in this way we ensure consistency across markets
    holdout_id = np.zeros((n_geos, n_times), dtype=bool)
    holdout_id[:, split_idx:] = True  # last 25% marked as holdout for all geos

    return holdout_id



def build_and_train_meridian(input_data, empirical_roi: float, holdout_id: np.ndarray | None = None) -> Meridian:
    # First of all we have to define the ROI prior using a LogNormal since ROI is strictly positive
    # We decided to change from the default LogNormal(0, 0.9) to a more informative prior based on the empirical ROI calculated from the data, which is around 4.28 (revenue/spend)
    # The empirical roi is obtained by summing all the net revenue and all the spend across the training period and then dividing total revenue by total spend. (see the notebook cell where we calculate the empirical ROI)
    roi_mu = float(np.log(empirical_roi))  # centers the LogNormal prior around the empirical ROI
    roi_sigma = float(0.5)  # tighter than default 0.9, because we have a prior estimate and we can reduce uncertainty around it


    prior = prior_distribution.PriorDistribution(
        roi_m=tfp.distributions.LogNormal(roi_mu, roi_sigma, name='roi_m') #log normal distribution
    )
    
    # Configuration of adstock and Hill curve settings to capture media carryover and saturation effects
    model_specifications = spec.ModelSpec(
        prior=prior,
        max_lag=8, #considers up to 8 days of carryover effect for each media channel
        hill_before_adstock=False, #applies saturation after adstock (standard approach)
        media_effects_dist='log_normal',
        paid_media_prior_type='roi',  # replaces media_prior_type into ROI
        knots=50, #number of internal knots for the spline basis functions that capture time dynamics (trend and seasonality
        # this not capture daily movements. Instead use 50 knots to capture more complex time patterns, such as weekly seasonality and long-term trends, without overfitting to daily noise
        holdout_id=holdout_id, # This ensure the completeness of the training period and the consistency of the holdout period across geos, avoiding issues with missing dates in the test set that can lead to NaN metrics
    ) 
    
    # Initialize Meridian with input data and model spec
    mmm = Meridian(input_data=input_data, model_spec=model_specifications)
    
    # Sample from prior — verifies prior assumptions before training
    mmm.sample_prior(500)
    
    # Sample from posterior via MCMC (monte carlo markov chain)
    mmm.sample_posterior(
        n_chains=4, #parallel sampling chains for convergence diagnostics
        n_adapt=1000, #adaptation steps for tuning the sampler
        n_burnin=1000, #burn-in samples discarded from final results
        n_keep=2000, #posterior samples retained for inference
        seed=42, #reproducibility index
    )
    
    return mmm


def evaluate_meridian_model(mmm: Meridian) -> None:
    mmm_analyzer = analyzer.Analyzer(mmm)

    # Compute predictive accuracy metrics on the full training period
    accuracy = mmm_analyzer.predictive_accuracy()

    print("Model performance metrics:")
    print(accuracy.to_dataframe())




def save_model(mmm: Meridian) -> None:
    path = "../outputs/model/mmm_model.binpb"
    # Instead of running everytime the model, we can save the trained model to disk and load it later for evaluation 
    # This is useful for large models that take a long time to train, like this one (30min)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save_mmm(mmm, path)
    print(f"Model saved to {path}")



def load_model(path: str = "../outputs/model/mmm_model.binpb") -> Meridian:
    # Load the trained model from disk, as a consequence of the previous function 
    mmm = model.load_mmm(path)
    print(f"Model loaded from {path}")
    return mmm


### PLOTS ###

def plots_MMM(mmm: Meridian) -> None:
    output_dir = '../outputs/charts'
    os.makedirs(output_dir, exist_ok=True)

    # vl_convert must use the system fonts to render the charts, so we register the Windows font directory to ensure it can find common fonts like Arial and Roboto used in the charts.
    vlc.register_font_directory("C:/Windows/Fonts")

    # Model Fit using meridian's visualizer
    model_fit = visualizer.ModelFit(mmm)
    fig_fit = model_fit.plot_model_fit()

    # Substitute the fonts because the library can't find it
    spec = json.loads(fig_fit.to_json())
    spec = json.loads(json.dumps(spec).replace('"Roboto"', '"Arial"'))
    spec['title'] = {
        'text': 'Model Fit: Expected vs Actual Revenue',
        'fontSize': 16,
        'fontWeight': 'bold',
        'color': '#5F6368'
    }

    # Save the model fit plot as PNG using vl_convert, otherwise they can only print html and json outputs
    png_bytes = vlc.vegalite_to_png(spec, scale=3)
    with open(os.path.join(output_dir, 'Meridian_model_fit.png'), "wb") as f:
        f.write(png_bytes) # save the PNG file to the output directory
    print(f"Saved: {output_dir}/Meridian_model_fit.png")

    # Response Curves
    # use the function to obtain the data, so that we can rebuild the plot with matplotlib with a layout 2x2
    media_effects = visualizer.MediaEffects(mmm)
    ds = media_effects.response_curves_data()
    channels = ds.channel.values

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Response Curves by Channel', fontsize=16, fontweight='bold')

    # recreate the full plot to have control on the output format
    for ax, channel in zip(axes.flatten(), channels):
        spend = ds['spend'].sel(channel=channel).values
        mean  = ds['incremental_outcome'].sel(channel=channel, metric='mean').values
        ci_lo = ds['incremental_outcome'].sel(channel=channel, metric='ci_lo').values
        ci_hi = ds['incremental_outcome'].sel(channel=channel, metric='ci_hi').values

        ax.plot(spend, mean, color='steelblue', linewidth=2)
        ax.fill_between(spend, ci_lo, ci_hi, alpha=0.2, color='steelblue', label='90% CI')
        ax.set_title(channel, fontsize=12, fontweight='bold')
        ax.set_xlabel('Spend', fontsize=10)
        ax.set_ylabel('Incremental Revenue', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style='plain', axis='y')
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}')) 

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Meridian_response_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/Meridian_response_curves.png")
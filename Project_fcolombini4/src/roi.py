import pandas as pd
import numpy as np
from meridian.model.model import Meridian
from meridian.analysis import analyzer
import numpy as np
import pandas as pd
from typing import Literal

def build_return_on_investment_table(mmm: Meridian, kind: Literal["roi", "mroi"] = "roi", prefix: str | None = None,
    **marginal_kwargs) -> pd.DataFrame:

    # Create analyzer from the fitted model
    az = analyzer.Analyzer(mmm)
    # And get channel names from model input data
    channels = mmm.input_data.media_channel.values

    # Choose which posterior tensor to compute
    if kind == "roi":
        # Compute posterior ROI tensor, shape
        samples_tensor = az.roi().numpy()
        default_prefix = "ROI"
    elif kind == "mroi":
        # Compute posterior MARGINAL ROI tensor, shape
        samples_tensor = az.marginal_roi(**marginal_kwargs).numpy()
        default_prefix = "mROI"
    else:
        raise ValueError("kind must be 'roi' or 'mroi'")

    # Collapse chains and draws into a single dimension: (n_samples, n_channels)
    samples = samples_tensor.reshape(-1, samples_tensor.shape[-1])

    # Compute posterior statistics across samples
    median = np.percentile(samples, 50, axis=0)
    lower = np.percentile(samples,  2.5, axis=0)
    upper = np.percentile(samples, 97.5, axis=0)

    # Determine column name prefix
    if prefix is None:
        prefix = default_prefix

    # Build DataFrame with results
    table = pd.DataFrame(
        {
            f"{prefix} Median": median,
            f"{prefix} 2.5%": lower,
            f"{prefix} 97.5%": upper,
        },
        index=channels,
    )
    table.index.name = "Channel"

    return table



def investment_evaluation_channel(roi_table: pd.DataFrame, mroi_table: pd.DataFrame) -> pd.DataFrame:

    # Combine median columns from both tables
    efficiency = pd.DataFrame(
        {
            "ROI Median":  roi_table["ROI Median"],
            "mROI Median": mroi_table["mROI Median"],
        },
        index=roi_table.index,
    )

    # Classify channels based on mROI threshold
    efficiency["Status"] = efficiency["mROI Median"].apply(
        lambda x: "Under-invested" if x > 1 else "Over-invested"
    )

    return efficiency
import os
from matplotlib import ticker
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy import stats
import calendar
from pathlib import Path

# Load the dataset
def load_datasets(dataset_path: str) -> pd.DataFrame:
    read_data = pd.read_csv(dataset_path) #read a csv file
    return read_data


# EDA of the dataset
def dataset_overview(df: pd.DataFrame) -> None:
    # 1. Print dataset shapes
    print("Dataset shape:", df.shape)

    # 2. Display missing values
    print("\nMissing values in dataset:")
    print(df.isnull().sum())

    # 3. Display basic statistics
    print("\nBasic statistics for dataset:")
    print(df.describe())

    print("Unique organisations:", df['ORGANISATION_ID'].nunique())
    print("Unique territories:", df['TERRITORY_NAME'].nunique())
    print("\nRows per organisation:")
    print(df['ORGANISATION_ID'].value_counts().head(10))
    print("\nDate range:")
    print(df['DATE_DAY'].min(), "→", df['DATE_DAY'].max())


def find_best_organisation(df: pd.DataFrame) -> str:
    results = []
    missing_threshold = 0.80
    spend_cols = [col for col in df.columns if col.endswith('_SPEND')]
    
    for org_id in df['ORGANISATION_ID'].unique():
        org_df = df[(df['ORGANISATION_ID'] == org_id) & (df['TERRITORY_NAME'] != 'All Territories')]
        
        # Date range coverage
        date_range = (pd.to_datetime(org_df['DATE_DAY'].max()) - pd.to_datetime(org_df['DATE_DAY'].min())).days
        
        # Count channels below missing threshold
        missing_pct = org_df[spend_cols].isnull().mean()
        active_channels = (missing_pct < missing_threshold).sum()
        
        results.append({
            'organisation_id': org_id,
            'date_range_days': date_range,
            'active_channels': active_channels,
            'rows': len(org_df)
        })
    
    results_df = pd.DataFrame(results).sort_values(
        by=['active_channels', 'date_range_days'], ascending=False
    )
    
    print(results_df.head(10))
    return results_df.iloc[0]['organisation_id']


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    spend_cols = [col for col in df.columns if col.endswith('_SPEND')]
    MISSING_TRESHOLD = 0.70
    # Drop channels above threshold
    missing_pct_by_geo = df.groupby('TERRITORY_NAME')[spend_cols].apply(
        lambda x: x.isnull().mean()
    )
    # Drop if ANY geo exceeds threshold
    cols_to_drop = missing_pct_by_geo.columns[
        missing_pct_by_geo.max() > MISSING_TRESHOLD
    ].tolist()

    # Drop also corresponding _CLICKS and _IMPRESSIONS columns
    for col in cols_to_drop:
        channel = col.replace('_SPEND', '')
        cols_to_drop += [c for c in df.columns if c.startswith(channel) and c != col]
    
    df = df.drop(columns=cols_to_drop)
    print(f"Dropped channels: {[col.replace('_SPEND', '') for col in cols_to_drop if col.endswith('_SPEND')]}")
    
    # Fill remaining missing with 0
    remaining_spend_cols = [col for col in df.columns if col.endswith('_SPEND')]
    df[remaining_spend_cols] = df[remaining_spend_cols].fillna(0)
    print(f"Filled with 0: {[col.replace('_SPEND', '') for col in remaining_spend_cols]}")
    
    return df


def select_firm(df: pd.DataFrame, NAME: str) -> pd.DataFrame:
    # given the org id, we select the rows corresponding to that org and all territories except "All Territories"
    org_top = df[(df['ORGANISATION_ID'] == NAME) & (df['TERRITORY_NAME'] != 'All Territories')].copy()
    print("Territories:", org_top['TERRITORY_NAME'].unique())
    print("Date range:", org_top['DATE_DAY'].min(), "to", org_top['DATE_DAY'].max())
    print("Rows:", len(org_top))
    print("Rows per territory:")
    print(org_top['TERRITORY_NAME'].value_counts())

    return org_top


def identify_media_channels(df: pd.DataFrame) -> dict:
    # we select each column useful for the media channels identification
    spend_cols = [col for col in df.columns if col.endswith('_SPEND')]
    clicks_cols = [col for col in df.columns if col.endswith('_CLICKS')]
    impressions_cols = [col for col in df.columns if col.endswith('_IMPRESSIONS')]
    
    # Extract unique channel names from spend columns
    channels = [col.replace('_SPEND', '') for col in spend_cols]
    
    return {
        'channels': channels,
        'spend_cols': spend_cols,
        'clicks_cols': clicks_cols,
        'impressions_cols': impressions_cols
    }


##################### TABLES ########################

def save_table(df: pd.DataFrame, NAME: str = None) -> None:
    output_dir= "../outputs/tables"
    filepath = os.path.join(output_dir, f"{NAME}.csv")
    df.to_csv(filepath)
    print(f"Table saved to {filepath}")




###################### PLOTS ########################


def plot_all_purchases_original_price(df: pd.DataFrame, NAME: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    # Group by date and sum net revenue, then plot
    df_grouped = df.groupby('DATE_DAY')['ALL_PURCHASES_ORIGINAL_PRICE'].sum().reset_index()
    ax.plot(df_grouped['DATE_DAY'], df_grouped['ALL_PURCHASES_ORIGINAL_PRICE'], alpha=0.7, color='navy')
    ax.xaxis.set_major_locator(plt.MaxNLocator(20))
    plt.xticks(rotation=30, ha='right')
    ax.set_title(f'{NAME} - Daily All Purchases (Original Price)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Total value of merchandise before discount')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    plt.tight_layout()
    plt.savefig(f'../outputs/charts/{NAME}_daily_all_purchases_original_price.jpg')
    plt.show()



def plot_channel_spend(
    df: pd.DataFrame,
    NAME: str,
    by_geo: bool = True,
    geo_col: str = 'TERRITORY_NAME',
) -> pd.DataFrame:
    spend_cols = [col for col in df.columns if col.endswith('_SPEND')]
    output_dir = Path(__file__).resolve().parents[1] / 'outputs' / 'tables'
    output_dir.mkdir(parents=True, exist_ok=True)

    if not by_geo or geo_col not in df.columns:
        spend_totals = df[spend_cols].fillna(0).sum().sort_values(ascending=False)
        spend_table = spend_totals.reset_index()
        spend_table.columns = ['Channel', 'Total Spend']
        spend_table['Channel'] = spend_table['Channel'].str.replace('_SPEND', '', regex=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(spend_table['Channel'], spend_table['Total Spend'], color='navy', alpha=0.8)

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{height:,.0f}',
                ha='center',
                va='bottom',
                fontsize=9,
            )

        ax.set_title(f'{NAME} - Actual Spend by Channel')
        ax.set_xlabel('Channel')
        ax.set_ylabel('Total Spend')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig(f'../outputs/charts/{NAME}_actual_spend_by_channel.jpg')
        plt.show()

        csv_path = output_dir / f'{NAME}_actual_spend_by_channel.csv'
        spend_table.to_csv(csv_path, index=False)
        print("saved csv:", str(csv_path))
        
        return spend_table

    spend_by_geo = (
        df.groupby(geo_col)[spend_cols]
        .sum(min_count=1)
        .fillna(0)
    )
    spend_by_geo.columns = [col.replace('_SPEND', '') for col in spend_by_geo.columns]
    spend_by_geo = spend_by_geo.sort_values(by=spend_by_geo.columns.tolist(), ascending=False)

    fig, ax = plt.subplots(figsize=(12, 7))
    spend_by_geo.plot(kind='bar', stacked=True, ax=ax, colormap='Blues')

    ax.set_title(f'{NAME} - Actual Spend by Channel and Geo')
    ax.set_xlabel('Geo')
    ax.set_ylabel('Total Spend')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    plt.xticks(rotation=30, ha='right')
    plt.legend(title='Channel', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'../outputs/charts/{NAME}_actual_spend_by_channel_geo.jpg', bbox_inches='tight')
    plt.show()

    csv_path = output_dir / f'{NAME}_actual_spend_by_channel_geo.csv'
    spend_by_geo.to_csv(csv_path)
    print("saved csv:", str(csv_path))

    return spend_by_geo

def plot_missing_heatmap(df: pd.DataFrame, NAME: str) -> None:
    #we select only the columns related to the media channels to analyze the missing values distribution
    spend_cols = [col for col in df.columns if col.endswith('_SPEND')]
    missing_pct = df[spend_cols].isnull().mean().sort_values(ascending=False) * 100
    print("Missing percentage per channel:")
    print(missing_pct.round(1))

    # Then we create a binary missing matrix (1 = missing, 0 = present)
    missing_matrix = df.set_index('DATE_DAY')[spend_cols].isnull().astype(int).sort_index()
    
    fig, ax = plt.subplots(figsize=(15, 6))
    sns.heatmap(missing_matrix.T, cmap=['navy', 'lightgrey'], 
                cbar=False, ax=ax, yticklabels=[col.replace('_SPEND', '') for col in spend_cols]) #clean the NAME
    
    ax.set_title(f'{NAME} - Missing Values Distribution Over Time (navy = present, grey = missing)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Channel')
    plt.tight_layout()
    plt.savefig(f'../outputs/charts/{NAME}_missing_heatmap.jpg')
    plt.show()



def plot_media_spend_trends(df: pd.DataFrame, NAME: str) -> None:
    # we select the spending columns like done before
    spend_cols = [col for col in df.columns if col.endswith('_SPEND')]
    
    #we group everything by date
    df_group = df.groupby('DATE_DAY')[spend_cols].sum()
    df_group['TOTAL_SPEND'] = df_group.sum(axis=1)
    
    # And plot the final graph with all the channels together, to see the trends over time
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df_group.index, df_group['TOTAL_SPEND'], color='navy', alpha=0.7)
    
    ax.set_title(f'{NAME} - Media Spend Trends by Channel')
    ax.set_xlabel('Date')
    ax.set_ylabel('Spend')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.xaxis.set_major_locator(plt.MaxNLocator(20))
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(f'../outputs/charts/{NAME}_media_spend_trends.jpg')
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, NAME: str) -> None:
    #same columns as before
    spend_cols = [col for col in df.columns if col.endswith('_SPEND')]
    #add also the target cols to see the correlation with the media spend
    target_cols = ['FIRST_PURCHASES', 'ALL_PURCHASES', 
                   'FIRST_PURCHASES_ORIGINAL_PRICE', 'ALL_PURCHASES_ORIGINAL_PRICE']
    
    # clear corr matrix
    corr_matrix = df[spend_cols + target_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap = sns.blend_palette(["white", "lightblue", "navy"], as_cmap=True),
                center=0, ax=ax, linewidths=0.5)
    
    ax.set_title(f'{NAME} - Correlation Heatmap — Media Spend vs Target Variables')
    plt.tight_layout()
    plt.savefig(f'../outputs/charts/{NAME}_correlation_heatmap.jpg')
    plt.show()



def plot_partial_correlation_heatmap(df: pd.DataFrame, NAME: str) -> None:
    # same as before
    spend_cols = [col for col in df.columns if col.endswith('_SPEND')]
    target_cols = ['FIRST_PURCHASES', 'ALL_PURCHASES', 
                   'FIRST_PURCHASES_ORIGINAL_PRICE', 'ALL_PURCHASES_ORIGINAL_PRICE']
    
    all_cols = spend_cols + target_cols
    all_cols_data = df[all_cols].fillna(0)

    # Compute partial correlations via residuals
    partial_corr = pd.DataFrame(np.nan, index=all_cols, columns=all_cols)
    
    for col_a in all_cols:
        for col_b in all_cols:
            if col_a == col_b:
                partial_corr.loc[col_a, col_b] = 1.0
            else:
                # Control variables: all columns except col_a and col_b
                controls = [con for con in all_cols if con != col_a and con != col_b]
                X = all_cols_data[controls].values
                
                # Residuals of col_a and col_b after removing control effects
                res_a = all_cols_data[col_a].values - LinearRegression().fit(X, all_cols_data[col_a]).predict(X)
                res_b = all_cols_data[col_b].values - LinearRegression().fit(X, all_cols_data[col_b]).predict(X)
                
                partial_corr.loc[col_a, col_b] = np.corrcoef(res_a, res_b)[0, 1]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(partial_corr.astype(float), annot=True, fmt='.2f',
                cmap = sns.blend_palette(["white", "lightblue", "navy"], as_cmap=True),
                center=0, ax=ax, linewidths=0.5, vmin=-1, vmax=1)
    
    ax.set_title(f'{NAME} - Partial Correlation Heatmap — Media Spend vs Target Variables')
    plt.tight_layout()
    plt.savefig(f'../outputs/charts/{NAME}_partial_correlation_heatmap.jpg')
    plt.show()



def plot_spend_concentration(df: pd.DataFrame, NAME: str) -> None:
    # as before
    spend_cols = [col for col in df.columns if col.endswith('_SPEND')]
    
    # Total spend per channel
    total_spend = df[spend_cols].fillna(0).sum()
    total_spend_pct = (total_spend / total_spend.sum() * 100).sort_values(ascending=False) #percentage of total spend per channel

    fig, ax1 = plt.subplots(figsize=(8, 6))
    bars = ax1.bar(
        [col.replace('_SPEND', '') for col in total_spend_pct.index],
        total_spend_pct.values,
        color='navy', alpha=0.7
    )
    
    # Add percentage labels on top of each bar
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)
    
    # set the second axis for absolute spend values
    scale = total_spend.sum() / 1_000_000 / 100.0  # millions per percentage point

    ax2 = ax1.twinx()
    y0, y1 = ax1.get_ylim()
    ax2.set_ylim(y0 * scale, y1 * scale)
    ax2.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.1f}M'))
    ax2.set_ylabel('Total Spend (Millions)')
    
    ax1.set_title(f'{NAME} - Spend Concentration by Channel')
    ax1.set_xlabel('Channel')
    ax1.set_ylabel('% of Total Spend')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    plt.setp(ax1.get_xticklabels(), rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(f'../outputs/charts/{NAME}_spend_concentration.jpg')
    plt.show()


def plot_seasonality(df: pd.DataFrame, NAME: str) -> None:
    df = df.copy()
    df['DATE_DAY'] = pd.to_datetime(df['DATE_DAY']) #idenify the proper columns and dates for each row
    df['day_of_week'] = df['DATE_DAY'].dt.day_name()
    df['month'] = df['DATE_DAY'].dt.month_name()

    # Order for day of week and month
    # full weekday names Monday..Sunday
    dow_order = list(calendar.day_name)
    # full month names January..December (calendar.month_name[0] is '')
    month_order = [calendar.month_name[i] for i in range(1, 13)]

    dow_avg = df.groupby('day_of_week')['ALL_PURCHASES'].mean().reindex(dow_order) #select seasonality across days of the week and months
    month_avg = df.groupby('month')['ALL_PURCHASES'].mean().reindex(month_order)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # first plot for day of week
    ax1.bar(dow_avg.index, dow_avg.values, color='navy', alpha=0.7)

    groups = [group['ALL_PURCHASES'].values for _, group in df.groupby('day_of_week')]
    f_stat, p_value = stats.f_oneway(*groups) # we use 2 outputs because stats return 2 values
   
   #Anova test to check if there are significant differences in purchases across days of the week
    if p_value < 0.05:
        anova_result = "ANOVA: significant differences across days (p<0.05)" 
    else: anova_result ="ANOVA: no significant differences across days"

    ax1.set_title(f'Average Purchases by Day of Week\n{anova_result}', fontsize=10)
    ax1.set_xlabel('Day of Week')
    ax1.set_ylabel('Avg Purchases')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.set_xticks(range(len(dow_order)))
    ax1.set_xticklabels(dow_order, rotation=30, ha='right')

    # second plot for the month
    ax2.bar(month_avg.index, month_avg.values, color='navy', alpha=0.7)
    ax2.set_title('Average Purchases by Month')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Avg Purchases')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.set_xticks(range(len(month_order)))
    ax2.set_xticklabels(month_order, rotation=30, ha='right')

    fig.suptitle(f'{NAME} - Seasonality Analysis')
    plt.tight_layout()
    plt.savefig(f'../outputs/charts/{NAME}_seasonality.jpg')
    plt.show()



def plot_outliers(df: pd.DataFrame, NAME: str) -> None:
    # as before
    spend_cols = [col for col in df.columns if col.endswith('_SPEND')]
    data = df[spend_cols].fillna(0) # avoid nan values for the boxplot
    
    # navy palette
    colors = sns.color_palette("Blues", len(spend_cols))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    # check the outliers for each channel with a boxplot, to see if there are some extreme values that could affect the model
    bp = ax.boxplot(data.values, patch_artist=True, labels=[col.replace('_SPEND', '') for col in spend_cols])
    
    for x, color in zip(bp['boxes'], colors):
        x.set_facecolor(color)
        x.set_alpha(0.7)
    
    ax.set_title(f'{NAME} - Outliers Detection by Channel')
    ax.set_xlabel('Channel')
    ax.set_ylabel('Spend')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(f'../outputs/charts/{NAME}_outliers.jpg')
    plt.show()
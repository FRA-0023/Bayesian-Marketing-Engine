
import numpy as np
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes.utils import calibration_and_holdout_data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns



# Docker parameters for PostgreSQL connection (In a real worl project should be stored in a .env file inserted in .gitignore)
DB_PARAMS = {
    'dbname': 'test',
    'user': 'admin',
    'password': 'admin',
    'host': 'localhost',
    'port': 5432
}


# Following the ER diagram, the parent tables must be loaded before child tables 
# This is made to respect Foreign Key constraints creating a logical order of loading the data
# Level 1 (no dependencies) -> Level 2 -> Level 3 -> Engagement tables
LOAD_ORDER = [
    # Level 1 - No foreign key dependencies
    'CATEGORY.csv',
    'CHANNEL.csv',
    'STORE.csv',
    'CUSTOMER.csv',
    'PROMOTION.csv',
    # Level 2 - Depend on Level 1
    'PRODUCT.csv',
    'ORDER.csv',
    # Level 3 - Depend on Level 2
    'ORDER_ITEM.csv',
    'PRICE_CHANGE.csv',
    # Engagement tables - Depend on CUSTOMER and STORE
    'SESSION.csv',
    'SUPPORT_TICKET.csv',
    'CAMPAIGN_TOUCH.csv',
    'EXTERNAL_FACTOR.csv',
    'MARKETING_CHANNEL.csv',
]

# Creation of the script in SQL to create the database schema
DDL_QUERY = """
-- Level 1: No foreign key dependencies
-- Creation of the dimensional tables that tend to be fixed in the time
CREATE TABLE IF NOT EXISTS CATEGORY (
    category_id VARCHAR PRIMARY KEY,
    category_name VARCHAR
);

CREATE TABLE IF NOT EXISTS CHANNEL (
    channel_id VARCHAR PRIMARY KEY,
    channel_name VARCHAR
);

CREATE TABLE IF NOT EXISTS STORE (
    store_id VARCHAR PRIMARY KEY,
    store_city VARCHAR,
    store_state VARCHAR,
    store_type VARCHAR
);

CREATE TABLE IF NOT EXISTS CUSTOMER (
    customer_id VARCHAR PRIMARY KEY,
    signup_date DATE,
    city VARCHAR,
    state VARCHAR,
    segment VARCHAR,
    status VARCHAR
);

CREATE TABLE IF NOT EXISTS PROMOTION (
    promo_id VARCHAR PRIMARY KEY,
    promo_type VARCHAR,
    start_date DATE,
    end_date DATE,
    discount_value FLOAT
);




-- Level 2: Depend on Level 1
-- Product Table
CREATE TABLE IF NOT EXISTS PRODUCT (
    product_id VARCHAR PRIMARY KEY,
    category_id VARCHAR REFERENCES CATEGORY(category_id),
    brand VARCHAR,
    sku_name VARCHAR
);

-- Order Table
CREATE TABLE IF NOT EXISTS "ORDER" (
    order_id VARCHAR PRIMARY KEY,
    customer_id VARCHAR REFERENCES CUSTOMER(customer_id),
    order_date DATE,
    store_id VARCHAR REFERENCES STORE(store_id),
    channel_id VARCHAR REFERENCES CHANNEL(channel_id),
    payment_type VARCHAR,
    year INT,
    week INT
);




-- Level 3: Depend on Level 2
-- Specific order table
CREATE TABLE IF NOT EXISTS ORDER_ITEM (
    order_item_id VARCHAR PRIMARY KEY,
    order_id VARCHAR REFERENCES "ORDER"(order_id),
    product_id VARCHAR REFERENCES PRODUCT(product_id),
    promo_id VARCHAR REFERENCES PROMOTION(promo_id),
    quantity INT,
    item_discount FLOAT
);

-- Historical price changes for each product
CREATE TABLE IF NOT EXISTS PRICE_CHANGE (
    product_id VARCHAR REFERENCES PRODUCT(product_id),
    year INT,
    week INT,
    unit_price FLOAT,
    PRIMARY KEY (product_id, year, week)
);




-- Engagement tables: depend on CUSTOMER or STORE
CREATE TABLE IF NOT EXISTS SESSION (
    session_id           VARCHAR PRIMARY KEY,
    customer_id          VARCHAR REFERENCES CUSTOMER(customer_id),
    session_start        TIMESTAMP,
    session_duration_sec INT,
    pages_viewed         INT,
    device               VARCHAR,
    referrer             VARCHAR
);

CREATE TABLE IF NOT EXISTS SUPPORT_TICKET (
    ticket_id          VARCHAR PRIMARY KEY,
    customer_id        VARCHAR REFERENCES CUSTOMER(customer_id),
    created_date       DATE,
    issue_type         VARCHAR,
    status             VARCHAR,
    resolution_time_hr INT,
    csat_score         INT
);

CREATE TABLE IF NOT EXISTS CAMPAIGN_TOUCH (
    touch_id      VARCHAR PRIMARY KEY,
    customer_id   VARCHAR REFERENCES CUSTOMER(customer_id),
    touch_date    DATE,
    channel       VARCHAR,
    campaign_name VARCHAR,
    outcome       VARCHAR
);

-- Depends on STORE
CREATE TABLE IF NOT EXISTS EXTERNAL_FACTOR (
    factor_id   VARCHAR PRIMARY KEY,
    store_id    VARCHAR REFERENCES STORE(store_id),
    factor_date DATE,
    is_holiday  BOOLEAN,
    temp_c      FLOAT,
    rainfall_mm FLOAT,
    trend_index FLOAT,
    cpi_index   FLOAT,
    year INT,
    week INT
);

CREATE TABLE IF NOT EXISTS MARKETING_CHANNEL (
    channel_id   VARCHAR PRIMARY KEY,
    channel_name VARCHAR,
    channel_type VARCHAR,
    cost_tier    VARCHAR
);
"""

# Function to connect to the PostgreSQL database and execute the DDL script to create the schema
def setup_database():
    try:
        with psycopg2.connect(**DB_PARAMS) as infrastructure:
            print("Successfully connected to the database PostgreSQL (in Docker).")
            
            with infrastructure.cursor() as execution:
                print("Executing the DDL script...")
                execution.execute(DDL_QUERY) # execution of the SQL script to create the database schema
                infrastructure.commit() # Saving of the changes in the database
                print("Operation completed: dimensional schema created successfully.")
                
    except Exception as error:
        print(f"Error during the operation: {error}")


# Dynamic ingestion function to insert data from a DataFrame into the specified table, using the provided columns
def ingest_table(conn, df: pd.DataFrame, table_name: str, columns: list):

    csv_columns = df.columns.tolist() # Check that CSV columns match the expected DDL columns
    missing = [col for col in columns if col not in csv_columns]
    if missing:
        print(f"[SKIP] {table_name}: missing columns {missing}")
        return

    cur = conn.cursor() #connection to the database
    col_str = ', '.join(columns) # transform the list of columns into a string separated by commas for the SQL query
    placeholders = ', '.join(['%s'] * len(columns)) # match the number of insertion with the number of columns
    sql = f"INSERT INTO {table_name} ({col_str}) VALUES ({placeholders}) ON CONFLICT DO NOTHING" # SQL query to insert data into the table, 
    # this avoid potential conflicts by ignoring them
    
    for _, row in df.iterrows():
        cur.execute(sql, tuple(row[col] for col in columns)) #execution of the SQL query for each row of the DataFrame, passing the values as a tuple
        # in this way psycopg2 can substitute the values of %s with the real values
    
    conn.commit() # Saving of the changes in the database
    print(f"{table_name}: {len(df)} rows inserted.")


def ingest_all():
    conn = psycopg2.connect(**DB_PARAMS) #connection with postgreSQL database
    
    for filename in LOAD_ORDER: # loop through the list of files in the specified order
        table_name = filename.replace('.csv', '') # transform the filename into the table name by removing .csv
        
        if table_name == 'ORDER':
            table_name = '"ORDER"' # ORDER is a reserved keyword in SQL, so it needs to be quoted to be used as a table name
        
        df = pd.read_csv(f'Data/{filename}') # read with pandas and parse dates in the correct format
        # Dynamically convert date columns and try parsing any string column as a date to solve potential date issues
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col], format='%d-%m-%Y')
                except:
                    pass
        df = df.where(pd.notnull(df), None) # Replace NaN with None for proper insertion into PostgreSQL
        ingest_table(conn, df, table_name, df.columns.tolist()) # use the ingestion function created above
            
    conn.close() # close the connection with the database
    print("Ingestion complete.")


def load_main_dataset(conn) -> pd.DataFrame:
    # We select only the relevant columns for the analysis and we join the tables to create a single dataset that contains all the information needed for the analysis
    query = """
    SELECT 
        o.order_id,
        o.customer_id,
        o.order_date,
        o.year,
        o.week,
        c.signup_date,
        c.city,
        c.state,
        c.segment,
        oi.quantity,
        oi.item_discount,
        p.product_id,
        p.brand,
        cat.category_name,
        pc.unit_price
    FROM "ORDER" o
    JOIN CUSTOMER c ON o.customer_id = c.customer_id
    JOIN ORDER_ITEM oi ON o.order_id = oi.order_id
    JOIN PRODUCT p ON oi.product_id = p.product_id
    JOIN CATEGORY cat ON p.category_id = cat.category_id
    JOIN PRICE_CHANGE pc ON p.product_id = pc.product_id 
        AND o.year = pc.year 
        AND o.week = pc.week
    """
    # We connect the table of the products with the specific value in the customer ID, 
    # then we connect the order to the order_item 
    # tham it that is connected to the product and the price change for that specific date (year + week) 
    # to have the price of the product at the moment of the order
    df = pd.read_sql(query, conn)
    print(f"Main dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def data_quality(df: pd.DataFrame) -> None:

    # Missing Values handling
    print("\nMissing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'missing_count': missing,
        'missing_%': missing_pct
    })
    if missing_df['missing_count'].sum() == 0:
        print("No missing values found in any column")
    else:
        print(missing_df[missing_df['missing_count'] > 0])


    #Descriptive Statistics
    print("\nDesriptive Statistics:")
    # Fix date columns
    for col in ['order_date', 'signup_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    # Identify column types automatically
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Exclude IDs from the EDA because they are not meaningful to describe
    exclude_cols = ['order_id', 'customer_id', 'product_id', 'category_id']
    analytic_numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

    print(f"Categorical columns : {categorical_cols}")
    print(f"Date columns : {date_cols}")
    print(f"Numeric (analytic) : {analytic_numeric_cols}")

    # Temporal range from year/week
    print(f"Temporal range:")
    print(f"Year: min: {df['year'].min()}, max: {df['year'].max()}")
    print(f"Week: min: {df['week'].min()}, max: {df['week'].max()}")

    analytic_numeric_cols = [col for col in analytic_numeric_cols if col not in ['year', 'week']]
    print(df[analytic_numeric_cols].describe())



def data_preparation(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    # Currency Conversion (INR -> USD)
    EXCHANGE_RATE = 83.0
    df['unit_price_usd'] = df['unit_price'] / EXCHANGE_RATE
    df['item_discount_usd'] = df['item_discount'] / EXCHANGE_RATE
    
    # Calcolate Gross Revenue for each line item (quantity * unit price)
    df['gross_revenue_usd'] = df['quantity'] * df['unit_price_usd']

    df['net_revenue_usd'] = df['gross_revenue_usd'] - df['item_discount_usd']
    
    # Aggregation by order_id to create the Orders dataset
    df_orders = df.groupby('order_id').agg({
        'customer_id': 'first',
        'order_date': 'first',
        'signup_date': 'first', 
        'year': 'first',
        'week': 'first',
        'gross_revenue_usd': 'sum',
        'item_discount_usd': 'max'
    }).reset_index()
    
    # Calcolate Net Revenue for each order
    df_orders['net_revenue_usd'] = df_orders['gross_revenue_usd'] - df_orders['item_discount_usd']
    df_orders['net_revenue_usd'] = df_orders['net_revenue_usd'].clip(lower=0)
    
    print(f"Dataset Items: {df.shape[0]} for each item.")
    print(f"Dataset Orders: {df_orders.shape[0]} unique orders.")
    
    return df, df_orders


def plot_spending_and_frequency(df_orders: pd.DataFrame) -> None:
    
    # Single client metrics calculation: total spending and purchase frequency
    customer_metrics = df_orders.groupby('customer_id').agg(
        total_spending=('net_revenue_usd', 'sum'),
        purchase_frequency=('order_id', 'count')
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # Customer Spending Distribution
    axes[0].hist(customer_metrics['total_spending'], bins=50, color='#ADD8E6', edgecolor='black')
    axes[0].set_title('1. Customer Spending Distribution', fontsize=14, pad=15)
    axes[0].set_xlabel('Total Spending per Customer (USD)', fontsize=12)
    axes[0].set_ylabel('Number of Customers', fontsize=12)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    axes[0].yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    # Purchase Frequency Distribution
    max_freq = int(customer_metrics['purchase_frequency'].max())
    axes[1].hist(customer_metrics['purchase_frequency'], bins=range(1, max_freq + 2), color='#000080', edgecolor='black', align='left')
    axes[1].set_title('2. Purchase Frequency Distribution', fontsize=14, pad=15)
    axes[1].set_xlabel('Number of Purchases (Transactions)', fontsize=12)
    axes[1].set_ylabel('Number of Customers', fontsize=12)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    axes[1].yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    plt.tight_layout()
    plt.savefig('plots/spending_and_frequency.jpg', dpi=300)
    plt.show()

def plot_spending_and_frequency_boxplot(df_orders: pd.DataFrame) -> None:
    
    # Single client metrics calculation: total spending and purchase frequency
    customer_metrics = df_orders.groupby('customer_id').agg(
        total_spending=('net_revenue_usd', 'sum'),
        purchase_frequency=('order_id', 'count')
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # 1. Customer Spending Distribution (Boxplot)
    axes[0].boxplot(customer_metrics['total_spending'], vert=True, patch_artist=True, 
                    boxprops=dict(facecolor='#ADD8E6', color='black'),
                    medianprops=dict(color='black', linewidth=1.5))
    axes[0].set_title('1. Customer Spending Distribution', fontsize=14, pad=15)
    axes[0].set_ylabel('Total Spending per Customer (USD)', fontsize=12)
    axes[0].set_xticks([]) # Rimuoviamo la spunta '1' sull'asse X perché non necessaria
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    axes[0].yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    # 2. Purchase Frequency Distribution (Boxplot)
    axes[1].boxplot(customer_metrics['purchase_frequency'], vert=True, patch_artist=True, 
                    boxprops=dict(facecolor='#000080', color='black'),
                    medianprops=dict(color='black', linewidth=1.5))
    axes[1].set_title('2. Purchase Frequency Distribution', fontsize=14, pad=15)
    axes[1].set_ylabel('Number of Purchases (Transactions)', fontsize=12)
    axes[1].set_xticks([])
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    axes[1].yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    
    plt.tight_layout()
    plt.savefig('plots/spending_and_frequency_boxplot.jpg', dpi=300)
    plt.show()


def plot_category(df_items: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 1, figsize=(8, 6))
    # Revenue by Product Category
    cat_revenue = df_items.groupby('category_name')['net_revenue_usd'].sum().sort_values(ascending=True)
    axes.barh(cat_revenue.index, cat_revenue.values, color='#ADD8E6', edgecolor='black')
    axes.set_title('3. Revenue by Category', fontsize=14, pad=15)
    axes.set_xlabel('Total Net Revenue (USD)', fontsize=12)
    axes.grid(axis='x', linestyle='--', alpha=0.7)
    axes.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    plt.tight_layout()
    plt.savefig('plots/category.jpg', dpi=300)
    plt.show()




def plot_demographics(df_items: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    # Customer Demographics (Segment)
    # We need to consider only unique customers to avoid biasing the distribution of segments and states with multiple orders from the same customer
    unique_customers = df_items.groupby('customer_id').first()
    segment_counts = unique_customers['segment'].value_counts()
    
    # Pie chart of segments distribution with percentage labels and custom colors
    axes[0].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%', 
                colors=['#000080', '#3b6ea0',  '#7fbfe6','#ADD8E6'], startangle=90)
    axes[0].set_title('4a. Customer Demographics (Segment)', fontsize=14, pad=15)

    # Customer Demographics (State)
    state_counts = unique_customers['state'].value_counts().head(10)
    axes[1].bar(state_counts.index, state_counts.values, color='#000080', edgecolor='black')
    axes[1].set_title('4b. Top 10 States by Customer Count', fontsize=14, pad=15)
    axes[1].set_ylabel('Number of Customers', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    axes[1].xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    plt.tight_layout()
    plt.savefig('plots/demographics.jpg', dpi=300)
    plt.show()


def plot_concentration(df_orders: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 1, figsize=(8, 6))

    # Revenue Concentration (Lorenz Curve)
    # We sum the total revenue per customer and sort it in descending order to calculate the cumulative distribution of revenue across customers
    customer_revenue = df_orders.groupby('customer_id')['net_revenue_usd'].sum().sort_values(ascending=False)
    
    # We calculate the cumulative percentage of revenue and the cumulative percentage of customers
    cum_rev_pct = customer_revenue.cumsum() / customer_revenue.sum() * 100
    cum_cust_pct = np.arange(1, len(customer_revenue) + 1) / len(customer_revenue) * 100

    axes.plot(cum_cust_pct, cum_rev_pct, color="#000080", linewidth=2, label='Actual Concentration')
    axes.plot([0, 100], [0, 100], color='black', linestyle='--', alpha=0.5, label='Equal Distribution')
    
    axes.set_title('5. Revenue Concentration', fontsize=14, pad=15)
    axes.set_xlabel('Percentage of Customers (%)', fontsize=12)
    axes.set_ylabel('Cumulative Percentage of Revenue (%)', fontsize=12)
    axes.legend()
    axes.grid(axis='both', linestyle='--', alpha=0.7)
    axes.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    plt.tight_layout()
    plt.savefig('plots/concentration.jpg', dpi=300)    
    plt.show()


def plot_trends(df_orders: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 1, figsize=(8, 6))

    # Time-based Purchasing Trends
    # We convert the dates to datetime and we create a new column with the year and month to group the revenue by month
    df_temp = df_orders.copy()
    df_temp['order_date'] = pd.to_datetime(df_temp['order_date'])
    df_temp['year_month'] = df_temp['order_date'].dt.strftime('%Y-%m')
    # So we group the revenues by month to see the trend of the revenue over time, we sum the revenue for each month to have a single point for each month in the plot
    trend_data = df_temp.groupby('year_month')['net_revenue_usd'].sum()

    axes.plot(trend_data.index, trend_data.values, color="#000080", marker='o', linewidth=2)
    axes.set_title('6. Time-based Purchasing Trends', fontsize=14, pad=15)
    axes.set_xlabel('Month', fontsize=12)
    axes.set_ylabel('Total Net Revenue (USD)', fontsize=12)
    axes.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    # Set a limited number of x-ticks to avoid overcrowding, we can set a tick every 3 months
    axes.set_xticks(trend_data.index[::3])
    axes.tick_params(axis='x', rotation=45)
    axes.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('plots/trends.jpg', dpi=300)    
    plt.show()

def assign_score(series, n_bins, ascending=True):
    labels = list(range(1, n_bins + 1)) if ascending else list(range(n_bins, 0, -1)) # build the laves if ascending or descending order
    try:
        return pd.qcut(series, q=n_bins, labels=labels, duplicates='drop') #it's preferred a quantile-based binning (equal size)
        # also it drops duplicates to avoid issues with bin edges that are not unique, which can happen with small datasets or skewed distributions like this case
    except ValueError:
        # If qcut fails fall back to equal-width bins.
        return pd.cut(series, bins=n_bins, labels=labels, duplicates='drop')


def calculate_rfm(df_orders: pd.DataFrame) -> pd.DataFrame:

    # Point 1 - Snapshot date: day after last recorded transaction
    df_orders['order_date'] = pd.to_datetime(df_orders['order_date'])
    snapshot_date = df_orders['order_date'].max() + pd.Timedelta(days=1)
    print(f"Snapshot date: {snapshot_date.date()}")

    # Point 2 - Aggregate raw RFM metrics at customer level
    rfm = df_orders.groupby('customer_id').agg({
        'order_date': lambda x: (snapshot_date - x.max()).days,
        'order_id': 'count',
        'net_revenue_usd': 'sum'
    }).reset_index()

    rfm.rename(columns={
        'order_date': 'Recency',
        'order_id': 'Frequency',
        'net_revenue_usd': 'Monetary'
    }, inplace=True)

    print(rfm[['Recency', 'Frequency', 'Monetary']].describe().round(2))

    # Point 3 - Adaptive scoring with 5 bins and handling of potential duplicates in bin edges
    N_BINS = 5
    rfm['R_Score'] = assign_score(rfm['Recency'],   n_bins=N_BINS, ascending=False)
    rfm['F_Score'] = assign_score(rfm['Frequency'], n_bins=N_BINS, ascending=True)
    rfm['M_Score'] = assign_score(rfm['Monetary'],  n_bins=N_BINS, ascending=True)

    for col in ['R_Score', 'F_Score', 'M_Score']:
        rfm[col] = rfm[col].astype(int)

    # RFM_Segment is a concatenation of the 3 scores, in this way we can extract easily the values of each score for each segment
    rfm['RFM_Segment'] = (
        rfm['R_Score'].astype(str) +
        rfm['F_Score'].astype(str) +
        rfm['M_Score'].astype(str))
    
    # RFM_Score as single summary indicator (1-5 scale)
    rfm['RFM_Score'] = ((rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']) / 3).round(2)

    print(f"\nUnique RFM Segments: {rfm['RFM_Segment'].nunique()} out of 125 possible")

    # Point 4 - Segment Verification and Pruning
    MIN_SIZE = 100 # under 100 observations segments are not statistically representative
    segment_counts = rfm['RFM_Segment'].value_counts()
    small_segments = segment_counts[segment_counts < MIN_SIZE].index
    large_segments = segment_counts[segment_counts >= MIN_SIZE].index

    customers_to_reassign = rfm['RFM_Segment'].isin(small_segments).sum()
    print(f"Segments below {MIN_SIZE} customers: {len(small_segments)}")

    def decode_segment(seg): #separate the R, F and M scores from the segment string to calculate the distance between segments
        return int(seg[0]), int(seg[1]), int(seg[2])

    def find_nearest_segment(small_seg, large_segments):
        """
        Find the nearest segment in `large_segments` to `small_seg` using Euclidean distance
        on decoded (R, F, M) coordinates.

        How does the function do that:
        - Decode small_seg to (r1, f1, m1).
        - Prefer searching only among large segments that share the same R value (r2 == r1).
        This reduces search space and enforces R-matching when possible.
        - If no large segment shares the R value, search all large_segments as a fallback.
        - Compute Euclidean distance between decoded vectors and return the large segment with the smallest distance.
        """
        r1, f1, m1 = decode_segment(small_seg)

        # Restrict to large segments that match the R component to prioritize R-aligned matches.
        same_r = [s for s in large_segments if decode_segment(s)[0] == r1]

        # Use matching pool if available, otherwise fall back to full set.
        search_pool = same_r if same_r else list(large_segments)

        min_dist, nearest = float('inf'), None
        for large_seg in search_pool:
            # Decode candidate large segment
            r2, f2, m2 = decode_segment(large_seg)

            # Euclidean distance in (R,F,M) space
            dist = np.sqrt((r1 - r2) ** 2 + (f1 - f2) ** 2 + (m1 - m2) ** 2)

            # Keep the closest candidate
            if dist < min_dist:
                min_dist, nearest = dist, large_seg

        return nearest

    # Build mapping: for each small segment, find its nearest large segment.
    segment_mapping = {}
    for small_seg in small_segments:
        segment_mapping[small_seg] = find_nearest_segment(small_seg, large_segments)


    # Apply mapping
    rfm['RFM_Segment'] = rfm['RFM_Segment'].replace(segment_mapping)

    print(f"Segments before pruning: {len(large_segments) + len(small_segments)}")
    print(f"Segments after pruning: {rfm['RFM_Segment'].nunique()}")
    print(f"Customers reassigned: {customers_to_reassign}")    
    print(f"\nRFM table successfully generated for {rfm.shape[0]} unique customers.")

    # Output required:
    # 1. Customer-level RFM table
    print("\nCustomer-Level RFM Table (sample)")
    print(rfm[['customer_id', 'Recency', 'Frequency', 'Monetary', 
            'R_Score', 'F_Score', 'M_Score', 'RFM_Segment', 'RFM_Score']].head(5).to_string(index=False))
    
    # 2. Distribution of RFM scores
    print("\nRFM Score Distribution")
    print(rfm['RFM_Score'].describe().round(2))
    print(f"\nCustomers per RFM Score level:")
    print(rfm['RFM_Score'].value_counts().sort_index())

    # Top 10 customers by RFM score
    print("\nTop 10 Customers by RFM Score")
    top10 = (rfm[['customer_id', 'Recency', 'Frequency', 'Monetary', 
                'RFM_Segment', 'RFM_Score']]
            .sort_values(by=['RFM_Score', 'RFM_Segment'], ascending=[False, False])
            .head(10)
            .to_string(index=False))
    print(top10)
    return rfm


def calculate_clv(rfm: pd.DataFrame, df_orders: pd.DataFrame) -> pd.DataFrame:

    # Churn Rate
    CHURN_THRESHOLD = 90  # days
    total_customers = len(rfm)
    churned_customers = (rfm['Recency'] > CHURN_THRESHOLD).sum()
    churn_rate = churned_customers / total_customers # calculate churn rate

    print()
    print(f"Total customers: {total_customers:,}")
    print(f"Churned customers: {churned_customers:,} (Recency > {CHURN_THRESHOLD} days)")
    print(f"Churn Rate: {churn_rate:.4f} ({churn_rate*100:.1f}%)")

    # Join signup_date via customer_id
    signup = df_orders[['customer_id', 'signup_date']].drop_duplicates('customer_id')
    clv = rfm.merge(signup, on='customer_id', how='left')

    # APV: Average Purchase Value per customer
    clv['APV'] = clv['Monetary'] / clv['Frequency']

    # PF: Purchase Frequency normalized by active days
    snapshot_date = pd.to_datetime(df_orders['order_date']).max() + pd.Timedelta(days=1)
    clv['signup_date'] = pd.to_datetime(clv['signup_date'])
    clv['active_months'] = (snapshot_date - clv['signup_date']).dt.days / 30 
    #frequency normalized by the number of months since signup to account for different customer lifetimes
    clv['PF'] = clv['Frequency'] / clv['active_months']

    print()
    # CLV per customer
    clv['CLV'] = (clv['APV'] * clv['PF']) / churn_rate
    clv.rename(columns={
        'APV': 'APV ($)',
        'PF': 'PF (purchases/month)',
        'CLV': 'CLV ($/month)'
    }, inplace=True)
    
    print(f"\nAll Customers CLV")
    print(clv[['APV ($)', 'PF (purchases/month)', 'CLV ($/month)']].describe().round(3))

    # Active customers only (not churned)
    clv_active = clv[clv['Recency'] <= 90].copy()

    print(f"\nActive Customers CLV")
    print(f"Active customers: {len(clv_active):,} ({len(clv_active)/len(clv)*100:.1f}% of total)")
    print(clv_active[['APV ($)', 'PF (purchases/month)', 'CLV ($/month)']].describe().round(3))

    print("\nTop 20 Customers by CLV")
    top20 = (clv[['customer_id', 'Recency', 'Frequency', 'Monetary',
                'APV ($)', 'PF (purchases/month)', 'CLV ($/month)']]
            .sort_values('CLV ($/month)', ascending=False)
            .head(20)
            .to_string(index=False))
    print(top20)

    return clv

def plot_clv_distribution(clv: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.hist(clv['CLV ($/month)'], bins=40, color="#000080", edgecolor='black')
    ax.set_title('CLV Distribution', fontsize=14, pad=15)
    ax.set_xlabel('CLV ($/month)', fontsize=12)
    ax.set_ylabel('Number of Customers', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    
    plt.tight_layout()
    plt.savefig('plots/clv_distribution.jpg', dpi=300)
    plt.show()


def predict_clv(df_orders: pd.DataFrame) -> pd.DataFrame:

    df_orders['order_date'] = pd.to_datetime(df_orders['order_date'])
    snapshot_date = df_orders['order_date'].max() + pd.Timedelta(days=1) #select the last day +1 of the dataset as it done before
    calibration_end = pd.Timestamp('2024-12-31') #split the dataset in 2 periods (train 2 years + test 1 year)

    # Transform the dataframe into the format required by the lifetimes library
    bgf_summary = summary_data_from_transaction_data(
        df_orders,
        customer_id_col='customer_id',
        datetime_col='order_date',
        monetary_value_col='gross_revenue_usd',
        observation_period_end=snapshot_date,
        freq='W' #weekly frequency because the month time period is not supported by the libraries (pandas + lifetimes)
    )

    print(f"\nBG/NBD Summary")
    print(bgf_summary.describe().round(3)) # show the summary of the data after the transformation

    # Filter for repeat purchasers for Gamma-Gamma (requires frequency > 0) and purchases without the discount
    # In fact as said before, some discounts are <0 because are calculated on the unit price, not the full order.
    gg_summary = bgf_summary[
        (bgf_summary['frequency'] > 0) & 
        (bgf_summary['monetary_value'] > 0)] # customers with at least 1 repeated purchase

    # Fit BetaGeo model (for frequency/recency)
    bgf = BetaGeoFitter(penalizer_coef=0.01) #it's the model BG/NDB 
    bgf.fit(
        bgf_summary['frequency'],
        bgf_summary['recency'],
        bgf_summary['T']
    )
    print(f"\nBetaGeo Model") # show the summary of the model fit, including parameter estimates and statistical significance
    print(bgf.summary)

    # Fit GammaGamma model (for monetary value)
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(
        gg_summary['frequency'],
        gg_summary['monetary_value']
    )
    print(f"\nGammaGamma Model") # same as before but with gamma  model
    print(ggf.summary)

    # Predict CLV over the 52 weeks of the year
    bgf_summary['predicted_clv'] = ggf.customer_lifetime_value( #combine the 2 models to predict the CLTV
        bgf,
        bgf_summary['frequency'],
        bgf_summary['recency'],
        bgf_summary['T'],
        bgf_summary['monetary_value'],
        time=52,
        freq='W',
        discount_rate=0.01 #applied to reflect the time value of money, used in finance evaluations
    )


    # Validation on holdout period (our last year of data)
    summary_cal_holdout = calibration_and_holdout_data(
        df_orders,
        customer_id_col='customer_id',
        datetime_col='order_date',
        monetary_value_col='gross_revenue_usd',
        calibration_period_end=calibration_end, #divide by calibration and holdout with the dates we have chosen before
        observation_period_end=snapshot_date,
        freq='W'
    )

    # Predict frequency during holdout period
    summary_cal_holdout['predicted_frequency'] = bgf.predict(
        t=summary_cal_holdout['duration_holdout'],
        frequency=summary_cal_holdout['frequency_cal'],
        recency=summary_cal_holdout['recency_cal'],
        T=summary_cal_holdout['T_cal']
    )

    actual = summary_cal_holdout['frequency_holdout']
    predicted = summary_cal_holdout['predicted_frequency']

    # Remove NaN values before computing metrics
    mask = ~(predicted.isna() | actual.isna())
    actual_clean = actual[mask]
    predicted_clean = predicted[mask]
    
    print(f"Customers excluded due to NaN: {(~mask).sum()}")

    # Evaluation metrics for frequency prediction
    rmse = np.sqrt(mean_squared_error(actual_clean, predicted_clean))
    mae = mean_absolute_error(actual_clean, predicted_clean)
    r2 = r2_score(actual_clean, predicted_clean)

    print(f"\nModel Evaluation")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"R²: {r2:.3f}")

    print(f"\nPredicted CLV Summary")
    print(bgf_summary['predicted_clv'].describe().round(3))

    print(f"\nTop 20 Customers by Predicted CLV")
    top20_predicted = (bgf_summary[['frequency', 'recency', 'T', 
                                     'monetary_value', 'predicted_clv']]
                      .sort_values('predicted_clv', ascending=False)
                      .head(20)
                      .round(2)
                      .to_string())
    print(top20_predicted)

    return bgf_summary

def plot_predicted_clv(bgf_summary: pd.DataFrame) -> None:

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # Left: full distribution
    axes[0].hist(bgf_summary['predicted_clv'].dropna(), 
                 bins=40, color='#ADD8E6', edgecolor='black')
    axes[0].set_title('Predicted CLV Distribution (All Customers)')
    axes[0].set_xlabel('Predicted CLV ($)')
    axes[0].set_ylabel('Number of Customers')
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # Right: zoomed (exclude top 1% outliers for readability)
    p99 = bgf_summary['predicted_clv'].quantile(0.99)
    axes[1].hist(bgf_summary[bgf_summary['predicted_clv'] <= p99]['predicted_clv'].dropna(),
                 bins=40, color="#ADD8E6", edgecolor='black')
    axes[1].set_title('Predicted CLV Distribution (Excluding Top 1%)')
    axes[1].set_xlabel('Predicted CLV ($)')
    axes[1].set_ylabel('Number of Customers')
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('plots/predicted_clv_distribution.jpg', dpi=300)
    plt.show()


def segment_customers(clv: pd.DataFrame, df_items: pd.DataFrame) -> pd.DataFrame:

    # assign segments based on CLV
    df_segments = clv[['customer_id', 'Recency', 'Frequency', 'Monetary',
                        'APV ($)', 'PF (purchases/month)', 'CLV ($/month)']].copy()

    df_segments['Segment'] = pd.qcut(
        df_segments['CLV ($/month)'],
        q=[0, 0.20, 0.50, 0.80, 1.0],
        labels=['Bronze', 'Silver', 'Gold', 'Platinum']
    ) #divide each segment based on the quantities required

    order = ['Platinum','Gold','Silver','Bronze']
    df_segments['Segment'] = pd.Categorical(df_segments['Segment'], categories=order, ordered=True)
    df_segments = df_segments.sort_values('Segment')  # Platinum first

    # Segment Summary Table with average metrics per segment
    summary = df_segments.groupby('Segment').agg(
        Customers=('customer_id', 'count'),
        Avg_Recency=('Recency', 'mean'),
        Avg_Frequency=('Frequency', 'mean'),
        Avg_Monetary=('Monetary', 'mean'),
        Avg_APV=('APV ($)', 'mean'),
        Avg_PF=('PF (purchases/month)', 'mean'),
        Avg_CLV=('CLV ($/month)', 'mean')
    ).round(2) # calculate the average of each metric for each segment to have a summary table of the segments

    print("\nSegment summary table")
    print(summary.to_string())

    # Join with df_items for demographics and product preferences
    items_customer = df_items[['customer_id', 'segment', 'city',
                                'state', 'category_name', 'brand']].copy()
    items_customer.rename(columns={'segment': 'customer_type'}, inplace=True)
    df_segments = df_segments.merge(items_customer, on='customer_id', how='left')

    # Demographics: segment distribution per CLV segment
    print("\nCustomer demographics by segment")
    demo = df_segments.groupby(['Segment', 'customer_type']).size().unstack(fill_value=0)
    print(demo.to_string())

    # Product preferences: top category per CLV segment as a similar plot created before, but this time with the segment as categories
    print("\nProduct preferences by segment")
    prefs = (df_segments.groupby(['Segment', 'category_name'])
             .size()
             .reset_index(name='item_occurrences')
             .sort_values(['Segment', 'item_occurrences'], ascending=[True, False])
             .groupby('Segment')
             .head(3))
    print(prefs.to_string(index=False))

    return df_segments

def plot_segments(df_segments: pd.DataFrame) -> None:

    segment_order = ['Bronze', 'Silver', 'Gold', 'Platinum']
    colors = ['#CD7F32', '#C0C0C0', '#FFD700', '#E5E4E2']

    #  CLV + Customer Type
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Average CLV by Segment
    clv_means = df_segments.groupby('Segment')['CLV ($/month)'].mean().reindex(segment_order)
    axes[0].bar(clv_means.index, clv_means.values, color=colors, edgecolor='black')
    axes[0].set_title('Average CLV by Segment', fontsize=13, pad=10)
    axes[0].set_xlabel('Segment')
    axes[0].set_ylabel('Avg CLV ($/month)')
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(clv_means.values):
        axes[0].text(i, v + 0.1, f'${v:.2f}', ha='center', fontsize=9)

    # Plot 2: Customer Type Distribution by Segment
    demo = (df_segments.groupby(['Segment', 'customer_type'])
            .size()
            .unstack(fill_value=0)
            .reindex(segment_order))
    demo_pct = demo.div(demo.sum(axis=1), axis=0) * 100
    demo_pct.plot(kind='bar', stacked=True, ax=axes[1],
                  color=['#000080', '#3b6ea0',  '#7fbfe6','#ADD8E6'], edgecolor='black')
    axes[1].set_title('Customer Type Distribution by Segment', fontsize=13, pad=10)
    axes[1].set_xlabel('Segment')
    axes[1].set_ylabel('% of Customers')
    axes[1].legend(title='Customer Type', bbox_to_anchor=(1.0, 1.0))
    axes[1].tick_params(axis='x', rotation=0)

    plt.tight_layout()
    plt.savefig('plots/segment_clv_demographics.jpg', dpi=300, bbox_inches='tight')
    plt.show()


def heatmap_segments(df_segments: pd.DataFrame) -> None:
    # Product Category Heatmap
    segment_order = ['Bronze', 'Silver', 'Gold', 'Platinum']
    fig, ax = plt.subplots(figsize=(8, 7))

    cat_segment = (df_segments.groupby(['Segment', 'category_name'])
                .size()
                .unstack(fill_value=0)
                .reindex(segment_order))
    cat_pct = cat_segment.div(cat_segment.sum(axis=1), axis=0) * 100


    sns.heatmap(cat_pct, ax=ax, 
                cmap = sns.blend_palette(["white", "lightblue", "navy"], as_cmap=True),
                annot=True, fmt='.1f', linewidths=0.5,
                annot_kws={'fontsize': 8}, cbar_kws={'shrink': 0.8})
    ax.set_title('Product Category Mix by Segment (%)', fontsize=13, pad=10)
    ax.set_xlabel('')
    ax.set_ylabel('Segment')

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)
    plt.savefig('plots/segment_heatmap.jpg', dpi=300, bbox_inches='tight')
    plt.show()



if __name__ == "__main__":
    # PART A - Data Understanding and Preparation

    #setup_database()
    #ingest_all()
    print("Database setup and data ingestion completed successfully.")

    conn = psycopg2.connect(**DB_PARAMS) # We open the connection at the beginning of the analysis
    df_main = load_main_dataset(conn) # Load the query to identify the specific dataset for the analysis
    df_items, df_orders = data_preparation(df_main) # Perform data preparation steps to create the final datasets for the analysis, including currency conversion, revenue calculation, and aggregation at order level

    data_quality(df_items) # Perform data quality checks on the prepared dataset
    data_quality(df_orders) # Perform data quality checks on the grouped dataset

    # PART B - Exploratory Data Analysis (EDA)
    plot_spending_and_frequency(df_orders) # Plot the distribution of customer spending and purchase frequency to understand customer behavior patterns
    plot_spending_and_frequency_boxplot(df_orders) # Plot the distribution of customer spending and purchase frequency using boxplots to visualize the spread and identify potential outliers
    plot_category(df_items)
    plot_demographics(df_items)
    plot_concentration(df_orders)
    plot_trends(df_orders)

    # PART C - RFM Analysis
    rfm_table = calculate_rfm(df_orders) # Calculate RFM scores for each customer

    # PART D - Customer Lifetime Value Analysis
    clv_table = calculate_clv(rfm_table, df_orders)
    plot_clv_distribution(clv_table) # Plot the distribution of CLV across customers to understand the range and identify high-value customers

    # PART E - Predict CLV using BG/NBD
    bgf_summary = predict_clv(df_orders)
    print(df_orders.columns.tolist())
    plot_predicted_clv(bgf_summary) # Plot the distribution of predicted CLV from the BG/NBD model to visualize the expected customer value over time

    # PART F - Customer Segmentation 
    df_segments = segment_customers(clv_table, df_items)
    plot_segments(df_segments) # Plot the characteristics of each customer segment to understand their demographics and preferences
    heatmap_segments(df_segments) # Plot a heatmap of product category preferences by segment to identify which segments prefer which categories

    conn.close() # We close the connection at the end of the analysis
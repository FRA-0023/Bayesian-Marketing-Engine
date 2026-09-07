import os
from typing import Tuple, List, Dict
import pandas as pd
import json
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from pyspark.sql.functions import col, sum as spark_sum, when
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    DoubleType
)

from pyspark.ml.feature import (
    Imputer,
    StandardScaler,
    StringIndexer,
    OneHotEncoder,
    VectorAssembler
)

from pyspark.ml.functions import vector_to_array
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression


# =========================================================
# STEP 0 - SPARK SESSION INITIALIZATION, SCHEMA DISCOVERY, AND FREEZING SCHEMA
# =========================================================

def create_spark_session(app_name: str = "Marketing_Analytics_Pipeline") -> SparkSession:
    """
    Initialize and configure the SparkSession.
    Using local[*] to leverage all available cores on the driver machine during development. 
    In a real production cluster this would be replaced with a cluster manager URL (like Kubernetes).
    """
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")  # Uses all local cores, simulating a distributed environment
        .config("spark.sql.shuffle.partitions", "8")  # Reduced from default 200 for local dev workloads
        .config("spark.driver.memory", "4g")  # Explicit driver memory cap to avoid uncontrolled growth
        .getOrCreate()
    )
    return spark


def discover_and_freeze_schema(spark: SparkSession,csv_path: str,schema_output_path: str) -> None:
    """
    Infer the schema once from the raw CSV before the loading using native Spark inference,
    then persist it as JSON to be reused on every future pipeline run without paying the inference cost again.
    RUN ONLY ONCE, OFFLINE, BEFORE THE PIPELINE IS DEPLOYED.

    Parameters:
    spark : SparkSession (Active Spark session)
    csv_path : str (Path to the raw source CSV file)
    schema_output_path : str (Path where the resulting schema JSON contract will be saved)
    """
    # inferSchema=True triggers a full data scan. This is done because this function runs exactly once, offline, not inside
    # the recurring production pipeline. This avoid manual implementation and potential errors in schema definition, 
    # while still allowing us to freeze the schema for future runs.
    inferred_df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(csv_path)
    )

    schema_dict = inferred_df.schema.jsonValue()

    with open(schema_output_path, "w") as f:
        json.dump(schema_dict, f, indent=2)

    print(f"Schema contract frozen and saved to: {schema_output_path}")
    inferred_df.printSchema()  # Visual check: review types before trusting the JSON


def load_schema_from_json(schema_path: str) -> StructType:
    """
    Load the schema from JSON.
    This is the function actually called inside the recurring pipeline,
    replacing any runtime inference.
    """
    with open(schema_path, "r") as f:
        schema_json = json.load(f)

    return StructType.fromJson(schema_json)


# =========================================================
# STEP 1 - LOAD DATASETS
# =========================================================

def load_datasets(spark: SparkSession, dataset_path: str, train_ratio: float = 0.8, seed: int = 420, schema: StructType = None) -> Tuple[DataFrame, DataFrame]:
    """
    Load the raw dataset with an explicit schema and split it into
    training and test Spark DataFrames using a distributed random split.

    Parameters:
    spark : SparkSession (The active Spark session initialized above)
    dataset_path : str (File path (local or distributed FS) for the source dataset)
    train_ratio : float (Proportion of data allocated to the training set)
    seed : int (Random seed for reproducibility across the cluster)
    schema: StructType (Explicit schema to avoid runtime inference overhead)

    Returns a Tuple[DataFrame, DataFrame] (like HW1 results):
        train_df : Training Spark DataFrame
        test_df  : Test Spark DataFrame
    """

    raw_df = (
        spark.read
        .option("header", "true")
        .schema(schema)  # Explicit schema, no inference overhead
        .csv(dataset_path)
    )

    # Distributed split of train and test sets, ensuring reproducibility with a fixed seed 
    train_df, test_df = raw_df.randomSplit([train_ratio, 1 - train_ratio], seed=seed)

    print(f"Loaded dataset from {dataset_path}:")
    train_df.show(5, truncate=False)

    return train_df, test_df


# =========================================================
# STEP 2 - DATASET OVERVIEW
# =========================================================


def dataset_overview(train_df: DataFrame, test_df: DataFrame) -> None:
    """
    Display datasets information using distributed Spark

    Parameters:
    train_df : DataFrame -> Training Spark DataFrame.
    test_df : DataFrame -> Test Spark DataFrame.

    Returns: None -> Just showing information
    """

    # 1. Dataset shape
    # Column count is free (schema metadata, no job triggered).
    # Row count triggers one distributed action per DataFrame.
    print("Training dataset shape:", (train_df.count(), len(train_df.columns)))
    print("Test dataset shape:", (test_df.count(), len(test_df.columns)))

    # 2. Missing values evaluation is computed in a single aggregation pass, building one expression per column, but executing them all together inside a single select(). 
    # In this way we obtain a single Spark job for the whole dataset, instead of one job per column and we let's take advantage of lazy evaluation
    def missing_values_report(df: DataFrame, label: str) -> None:
        missing_exprs = [
            spark_sum(when(col(c).isNull(), 1).otherwise(0)).alias(c)
            for c in df.columns
        ]
        # Result is a single row: safe to collect to the driver for formatting.
        result_pd = df.select(missing_exprs).toPandas().T
        result_pd.columns = ["missing_count"]
        print(f"\nMissing values in {label} dataset:")
        print(result_pd)

    missing_values_report(train_df, "training")
    missing_values_report(test_df, "test")


    # 3. Basic statistics to compute count, mean, stddev, min, max plus quartiles in one move
    # Here we use toPandas() to collect the summary statistics to the driver for display. 
    # This is safe because the summary is a small DataFrame (one row per statistic, one column per feature).
    print("\nBasic statistics for training dataset:")
    print(train_df.summary().toPandas().set_index("summary").T)

    print("\nBasic statistics for test dataset:")
    print(test_df.summary().toPandas().set_index("summary").T)


# =========================================================
# STEP 3 - DEFINE FEATURES AND TARGET
# =========================================================

def split_features_target(train_df: DataFrame,test_df: DataFrame,target_column: str) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """
    Separate features and target variable, Spark-native style.
    This function returns column-pruned views: a features-only DataFrame (all columns except the target) and a single-column label DataFrame, 
    both still logically tied to the same underlying row set via Spark's lazy DAG.
    In this way we avoid any unnecessary data movement and minimize the possible costs.

    Parameters:
    train_df : DataFrame -> Training Spark DataFrame.
    test_df : DataFrame -> Test Spark DataFrame.
    target_column : str -> Name of the target column.

    Returns:
    Tuple[DataFrame, DataFrame, DataFrame, DataFrame]
        X_train : Training features (all columns except target)
        y_train : Training label (single-column DataFrame)
        X_test  : Test features (all columns except target)
        y_test  : Test label (single-column DataFrame)
    """

    # .drop() on a Spark DataFrame is a lazy transformation: it only
    # updates the logical plan's column projection, no data movement occurs.
    X_train = train_df.drop(target_column)
    y_train = train_df.select(target_column)

    X_test = test_df.drop(target_column)
    y_test = test_df.select(target_column)

    print("Train and test features and target split completed")
    return X_train, y_train, X_test, y_test


# =========================================================
# STEP 4 - IDENTIFY COLUMN TYPES
# =========================================================

def identify_column_types(X_train: DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identify categorical and numerical columns using Spark's schema metadata.

    This operation reads the schema that we created before, not the data, in this way no job is triggered and no partitions are scanned. 

    Parameters:
    X_train : DataFrame -> Training feature Spark DataFrame

    Returns:
    Tuple[List[str], List[str]]
        categorical_cols : List of categorical (string) columns
        numeric_cols     : List of numerical (integer/double) columns
    """

    categorical_cols = [
        name for name, dtype in X_train.dtypes
        if dtype == "string"
    ]

    numeric_cols = [
        name for name, dtype in X_train.dtypes
        if dtype in ("int", "double", "bigint", "float")
    ]

    print("Identified categorical columns:", categorical_cols)
    print("Identified numerical columns:", numeric_cols)
    return categorical_cols, numeric_cols


# =========================================================
# STEP 5 - BUILD PREPROCESSOR
# =========================================================


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str], df: DataFrame) -> Tuple[Pipeline, dict]:
    """
    Create the MLlib preprocessing with the following preprocessing phases:
    - Missing value handling -> Imputation (median for numeric, precomputed mode for categorical)
    - Feature scaling -> Standardization of numerical features
    - One-hot encoding -> Sparse binarization of categorical variables
    - Final assembly -> All processed features combined into one vector

    Parameters:
    numeric_cols : List[str] -> Numerical feature column names
    categorical_cols : List[str] -> Categorical feature column names
    df : DataFrame -> Spark DataFrame used to compute categorical modes for imputation

    Returns:
    Pipeline -> Unfitted MLlib Pipeline by design 
    dict -> Categorical modes for imputation
    """
    
    stages = []
    # Numeric branch: median imputation -> vector assembly -> scaling
    imputed_numeric_cols = [f"{c}_imputed" for c in numeric_cols]

    numeric_imputers = Imputer(
        strategy="median",
        inputCols=numeric_cols,
        outputCols=imputed_numeric_cols
    )
    stages.append(numeric_imputers)

    numeric_assembler = VectorAssembler(
        inputCols=imputed_numeric_cols,
        outputCol="numeric_features_raw"
    )
    stages.append(numeric_assembler)

    # StandardScaler operates on the assembled vector, not on individual columns, this is why assembly happens BEFORE scaling in Spark
    numeric_scalers = StandardScaler(
        inputCol="numeric_features_raw",
        outputCol="numeric_features_scaled",
        withMean=True,
        withStd=True
    )
    stages.append(numeric_scalers)



    # Categorical branch: StringIndexer -> OneHotEncoder (sparse)
    def compute_categorical_modes(df: DataFrame, categorical_cols: List[str]) -> dict:
        """
        Using spark, we can't use the fillna() method with mode, so we need to compute it for each categorical column manually.
        With this fucntion we compute the most frequent value (mode) for each categorical column using a single distributed groupBy per column.
        This is computed once, on the training set and the resulting fixed values are later reused to fill both train and test sets consistently.

        Parameters:
        df : DataFrame -> Spark DataFrame (the training set only, to avoid leakage).
        categorical_cols : List[str] -> Categorical column names

        Returns:
        dict -> Mapping {column_name: most_frequent_value}
        """
        modes = {}
        for t in categorical_cols:
            # Distributed aggregation: groups values, counts occurrences, keeps only the top row. 
            top_row = (
                df.filter(col(t).isNotNull())
                .groupBy(t)
                .count()
                .orderBy(col("count").desc())
                .first()
            )
            modes[t] = top_row[t] if top_row else None
        return modes

    categorical_modes = compute_categorical_modes(df, categorical_cols)

    indexed_cols = [f"{t}_indexed" for t in categorical_cols]
    encoded_cols = [f"{t}_encoded" for t in categorical_cols]

    string_indexer = StringIndexer(
        inputCols=categorical_cols,
        outputCols=indexed_cols,
        handleInvalid="keep"  # unseen categories at inference time get their own index instead of failing
    )
    stages.append(string_indexer)

    one_hot_encoders = OneHotEncoder(
        inputCols=indexed_cols,
        outputCols=encoded_cols
    )
    stages.append(one_hot_encoders)

    # Final assembly: combine scaled numeric vector + all encoded vectors
    final_assembler = VectorAssembler(
        inputCols=["numeric_features_scaled"] + encoded_cols,
        outputCol="features"  # MLlib's conventional column name for estimators
    )
    stages.append(final_assembler)

    preprocessor = Pipeline(stages=stages)
    print("Preprocessing pipeline created")

    return preprocessor, categorical_modes


# =========================================================
# STEP 6 - BUILD MODEL PIPELINE
# =========================================================

def compute_class_weights(df: DataFrame, label_col: str) -> DataFrame:
    """
    In Sklearn the class_weight='balanced' automatically balances the classes.
    In Spark there is no automatic option, so we have to calculate the weight of each class by hand and add it as a new column.
    """

    # Count how many rows belong to each class (0 and 1).
    # It produces just 2 rows as a result = cheap operation.
    class_counts = df.groupBy(label_col).count().collect()

    # Then compute the total number of rows in the dataset (sum all 0 + all 1)
    total = sum(row["count"] for row in class_counts)
    # And check how many distinct classes we have (should be 2 for churn)
    n_classes = len(class_counts)

    # weight = total_rows / (number_of_classes * rows_in_this_class)
    # This gives a HIGHER weight to the class with FEWER rows, so the model pays more attention to it during training.
    weights = {
        row[label_col]: total / (n_classes * row["count"])
        for row in class_counts
    }

    # We need to add this weight to every row of the DataFrame, depending on which class that row belongs to.
    weight_expr = None
    for class_value, weight in weights.items():
        if weight_expr is None:
            weight_expr = when(col(label_col) == class_value, weight)
        else:
            weight_expr = weight_expr.when(col(label_col) == class_value, weight)

    # add the new column "class_weight" to the DataFrame using the expression built above, and return the updated DataFrame.
    return df.withColumn("class_weight", weight_expr)


def build_model(preprocessor: Pipeline, label_col: str) -> Dict[str, Pipeline]:
    """
    Defines the models and the pipelines, just the steps, not the training.
    """

    # Random Forest model definition
    random_forest = RandomForestClassifier(
        labelCol=label_col, # select the column as target 
        featuresCol="features", #assembled feature
        weightCol="class_weight", # class balancing column
        seed=420  # same idea as random_state in sklearn, just a different name
    )

    # Logistic Regression model definition
    logistic_regression = LogisticRegression(
        labelCol=label_col, #same as above
        featuresCol="features",
        weightCol="class_weight",
        maxIter=1000 # 1000 iterations to ensure convergence
    )

    # Add the classifierat the end of that previous list, so preprocessing + model become a single pipeline.
    full_pipeline = {
        "Random Forest": Pipeline(stages=preprocessor.getStages() + [random_forest]),
        "Logistic Regression": Pipeline(stages=preprocessor.getStages() + [logistic_regression])
    }

    return full_pipeline


# =========================================================
# STEP 7 - TRAIN MODEL
# =========================================================

def train_model(model: Pipeline, train_df: DataFrame) -> PipelineModel:
    """
    Train machine learning model.

    Parameters:
    model : Pipeline (not yet fitted)

    train_df : DataFrame
        Training Spark DataFrame, containing the original feature columns,the label column, and the class_weight column already computed

    Returns:
    PipelineModel -> Trained model complete
    """

    # fit the model
    fitted_model = model.fit(train_df)
    print(f"Model {model.getStages()[-1].__class__.__name__} trained successfully")

    return fitted_model


# =========================================================
# STEP 8 - MAKE PREDICTIONS
# =========================================================

def make_predictions(model: PipelineModel, X_test: DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Generate predictions and probabilities.

    Parameters:
    model : PipelineModel -> Fitted pipeline 
    X_test : DataFrame -> Original, untransformed columns

    Returns:
    Tuple[pd.Series, pd.Series]
        y_pred : Predicted labels
        y_prob : Prediction probabilities (positive class only)

    This function converts the entire test set predictions to pandas Series (AS REQUESTED FROM THE INSTRUCTOR). 
    That means that it could cause an "Out Of Memory" crash.
    If the dataset ever grows significantly, this function should be redesigned to avoid extreme costs (on scale production).
    """

    # Run the transformation to produce rawPrediction, probability, prediction columns all at once
    predictions_df = model.transform(X_test)

    # Extract the positive-class probability from the "probability" vector column
    predictions_df = predictions_df.withColumn(
        "probability_array", vector_to_array(col("probability"))
    ).withColumn(
        "probability_positive_class", col("probability_array")[1]
    )

    # Collect only the two relevant columns to the driver as pandas.
    result_pd = predictions_df.select(
        "prediction", "probability_positive_class"
    ).toPandas()

    y_pred = result_pd["prediction"].rename("prediction")
    y_prob = result_pd["probability_positive_class"].rename("probability")
    print(f"Predictions generated successfully using {model.stages[-1].__class__.__name__}")

    return y_pred, y_prob


# =========================================================
# STEP 9 - EVALUATE MODEL
# =========================================================

def evaluate_model(test_df: DataFrame, y_pred: pd.Series, y_prob: pd.Series, label_col: str = "churn") -> None:
    """
    Evaluate classification performance.

    Parameters:
    test_df : DataFrame
        Spark test DataFrame, used here only to extract the true labels.
    y_pred : pd.Series
        Predicted labels, from make_predictions.
    y_prob : pd.Series
        Predicted probabilities, from make_predictions.
    label_col : str
        Name of the true label column.
    
    Returns: None
    """

    y_test = test_df.select(label_col).toPandas()[label_col]

    # 1. Compute evaluation metrics
    # we use the functions of sklearn to compute the requested metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    #here the specific metrics, requested from the instructions, are computed

    # 2. Print model performance
    print("Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    #a simple print of the metrics is made, with 4 decimal digits for better readability

    # BONUS POINTS: ROC Curve
    # To plot the ROC curve, we need to calculate the true positive rate (TPR) and false positive rate (FPR) and put them together in a df
    # ROC Curve = TPR (x axis) vs FPR (y axis)
    curve = pd.DataFrame({'y_true': y_test, 'y_prob': y_prob}).sort_values('y_prob', ascending=False)
    # a dataframe oredered by the predicted probabilities is created, to plot the ROC curve

    curve['TP'] = curve['y_true'].cumsum() #sum of the true positives
    curve['FP'] = (1 - curve['y_true']).cumsum() #sum of the false positives

    POSITIVE = curve['y_true'].sum() #sum of the total positives (TP + FN)
    NEGATIVE = len(curve) - POSITIVE #sum of the total negatives (FP + TN)

    TPR = curve['TP'] / POSITIVE  # calculated as cumulative true positives divided by the total positives (TPR = TP / (TP + FN))
    # TPR is also called Recall and could be used to plot the second requested curve (Precision-Recall curve)
    FPR = curve['FP'] / NEGATIVE  # calculated as cumulative false positives divided by the total negatives (FPR = FP / (FP + TN))

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(FPR, TPR, color='coral', lw=2)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.title('ROC Curve', fontsize=18, fontweight='bold')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.tight_layout()
    plt.show()

    Precision = curve['TP'] / (curve['TP'] + curve['FP'])  # Precision = TP / (TP + FP)

    # Plot Precision-Recall curve
    # shows the trade-off between precision and recall across different classification score thresholds
    plt.figure(figsize=(8, 6))
    plt.plot(TPR, Precision, color='pink', lw=2, label='Precision')
    plt.title('Precision-Recall Curve', fontsize=18, fontweight='bold')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.tight_layout()
    plt.show()


# =========================================================
# STEP 10 - CONFUSION MATRIX
# =========================================================

def display_confusion_matrix(test_df: DataFrame, y_pred: pd.Series, label_col: str = "churn") -> None:
    """
    Display confusion matrix.

    Parameters:
    test_df : DataFrame -> Spark test DataFrame, used here only to extract the true labels.
    y_pred : pd.Series -> Predicted labels, from make_predictions.
    label_col : str -> Name of the true label column.

    Returns: None
    """

    # Extract true labels as pandas here, same pattern used in evaluate_model.
    y_test = test_df.select(label_col).toPandas()[label_col]

    # 1. Compute confusion matrix
    confusion_matrix_created = confusion_matrix(y_test, y_pred)

    confusion_matrix_created_df = pd.DataFrame(
        confusion_matrix_created,
        index=['Real: 0', 'Real: 1'],
        columns=['Predicted: 0', 'Predicted: 1']
    )

    # 2. Display results
    print(f"\nConfusion Matrix: \n{confusion_matrix_created_df}")


# =========================================================
# STEP 11 - FEATURE IMPORTANCE
# =========================================================

def feature_importance_analysis(model: PipelineModel,df: DataFrame,numeric_cols: List[str],categorical_cols: List[str]
) -> pd.DataFrame:
    """
    Analyze feature importance.

    Parameters:
    model : PipelineModel -> The trained pipeline 
    df : DataFrame -> Spark DataFrame with the original feature columns
    numeric_cols : List[str] -> List of numeric column names
    categorical_cols : List[str] -> List of categorical column names

    Returns:
    pd.DataFrame -> The DataFrame containing feature importances.
    """

    transformed_df = model.transform(df)
    schema = transformed_df.schema

    #exact order we had before 
    numeric_feature_names = list(numeric_cols)

    # OneHotEncoder expands each single column into MULTIPLE vector positions, so names must be read from
    # metadata to stay aligned with the actual number of positions in "features"
    categorical_feature_names = []
    for c in categorical_cols:
        encoded_col = f"{c}_encoded"
        attrs = schema[encoded_col].metadata["ml_attr"]["attrs"]
        col_attrs = []
        for attr_type in attrs:
            col_attrs.extend(attrs[attr_type])
        col_attrs_sorted = sorted(col_attrs, key=lambda a: a["idx"])
        categorical_feature_names.extend([f"{c}_{a['name']}" for a in col_attrs_sorted])

    # Final order must match final_assembler: numeric_features_scaled + encoded_cols
    feature_names = numeric_feature_names + categorical_feature_names

    # Extract feature importance values
    model_classifier = model.stages[-1]

    if hasattr(model_classifier, "featureImportances"):
        # For Tree-based models, like Random Forest, featureImportances is a SparseVector, so we mustconvert to a plain array first
        importance_of_the_features = model_classifier.featureImportances.toArray()

    elif hasattr(model_classifier, "coefficients"):
        # For Linear models, like Logistic Regression, the coefficients are already a 1D Vector for binary classification
        importance_of_the_features = [abs(val) for val in model_classifier.coefficients.toArray()]

    else:
        importance_of_the_features = [0.0] * len(feature_names) # everything else = 0

    # Normalize to comparable percentages, same rationale as the original sklearn version.
    total_importance = sum(importance_of_the_features)
    if total_importance > 0:
        importance_of_the_features = [(val / total_importance) for val in importance_of_the_features]


    # Create importance DataFrame
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importance_of_the_features})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    # Plot top features
    top_n = 5
    top_features = importance_df.head(top_n)
    total_importance = importance_df["Importance"].sum()
    top_5_weight = importance_df["Importance"].head(top_n).sum() / total_importance * 100

    plt.figure(figsize=(8, 6))
    plt.barh(top_features["Feature"][::-1], top_features["Importance"][::-1], color="navy")
    plt.xlabel("Importance")
    plt.suptitle("Top Features for importance", fontsize=18, fontweight="bold")
    plt.title(f"Top {top_n} features account for {top_5_weight:.2f}% of total importance", fontsize=12)
    plt.gcf().subplots_adjust(left=0.25)
    plt.show()

    return importance_df


# =========================================================
# STEP 12 - SAVE PREDICTIONS
# =========================================================

def save_predictions(test_df: DataFrame, y_pred: pd.Series, y_prob: pd.Series, output_file: str) -> None:
    """
    Save prediction results to CSV file

    Parameters:
    test_df : DataFrame -> Original Spark test dataset (untransformed columns).
    y_pred : pd.Series -> Predicted labels, from make_predictions.
    y_prob : pd.Series -> Prediction probabilities, from make_predictions.
    output_file : str -> Output CSV filename.

    Returns: None

    """

    # Create output DataFrame
    # Convert the Spark test set to pandas
    output_dataframe = test_df.toPandas()

    # 2. Add predictions
    output_dataframe['predicted_class'] = y_pred
    output_dataframe['predicted_probability'] = y_prob

    # 3. Save to CSV file
    output_dataframe.to_csv(output_file, index=False)


DATA_PATH = "data/raw/"
CSV_FILE = ["dataset1_HW1.csv", "dataset2_HW1.csv"]
DS1_PATH = DATA_PATH + CSV_FILE[0]
DS2_PATH = DATA_PATH + CSV_FILE[1]
SCHEMA1_PATH = "schemas/dataset1_churn_schema.json"
SCHEMA2_PATH = "schemas/dataset2_churn_schema.json"
DATA_OUTPUT_PATH = "data/processed/"

if __name__ == "__main__":

    spark = create_spark_session()

    datasets_info = {
        "Dataset 1": {
            "csv_path": DS1_PATH,
            "schema_path": SCHEMA1_PATH
        },
        "Dataset 2": {
            "csv_path": DS2_PATH,
            "schema_path": SCHEMA2_PATH
        }
    }

    for ds_name, ds_info in datasets_info.items():
        print(f"\nANALYSIS for {ds_name.upper()}")

        ds_path = ds_info["csv_path"]
        schema_path = ds_info["schema_path"]

        # One-time schema discovery per dataset, same pattern as before
        if not os.path.exists(schema_path):
            discover_and_freeze_schema(
                spark=spark,
                csv_path=ds_path,
                schema_output_path=schema_path
            )

        schema = load_schema_from_json(schema_path)
        train_df, test_df = load_datasets(spark, ds_path, schema=schema)

        print(f"\n{ds_name} Overview:")
        dataset_overview(train_df, test_df)

        # PREPROCESSING
        x_train, y_train, X_test, y_test = split_features_target(train_df, test_df, target_column="churn")
        categorical_cols, numeric_cols = identify_column_types(x_train)
        preprocessor, categorical_modes = build_preprocessor(numeric_cols, categorical_cols, train_df)

        train_df_imputed = train_df.fillna(categorical_modes)
        test_df_imputed = test_df.fillna(categorical_modes)

        train_df_weighted = compute_class_weights(train_df_imputed, "churn")

        # MODELLING
        models_dict = build_model(preprocessor, "churn")
        y_test_pd = test_df_imputed.select("churn").toPandas()["churn"]

        best_roc_auc = 0
        best_model_name = ""
        best_trained_model = None
        best_predictions_test = None
        best_probabilities_test = None

        for model_name, pipeline in models_dict.items():
            print(f"Training and evaluating {model_name}...")
            trained_model = train_model(pipeline, train_df_weighted)
            predictions_test, probabilities_test = make_predictions(trained_model, test_df_imputed)

            # Evaluate the ROC AUC score for the current model
            current_auc = roc_auc_score(y_test_pd, probabilities_test)
            print(f"-> ROC AUC {model_name}: {current_auc:.4f}")

            # Save the first model, then update only if the next one is better
            if current_auc > best_roc_auc:
                best_roc_auc = current_auc
                best_model_name = model_name
                best_trained_model = trained_model
                best_predictions_test = predictions_test
                best_probabilities_test = probabilities_test

        print(f"\n[Best Model for AUC {ds_name.upper()}]: {best_model_name} with ROC AUC of {best_roc_auc:.4f}\n")

        # RESULTS
        evaluate_model(test_df_imputed, best_predictions_test, best_probabilities_test, "churn")
        display_confusion_matrix(test_df_imputed, best_predictions_test, "churn")
        feature_importance_analysis(best_trained_model, test_df_imputed, numeric_cols, categorical_cols)

        # Predictions on the train dataset, to save them later in csv, with the best model
        predictions_train, probabilities_train = make_predictions(best_trained_model, train_df_imputed)

        # Save the data in different csv files
        ds_prefix = ds_name.lower().replace(" ", "")
        save_predictions(train_df_imputed, predictions_train, probabilities_train, f"{DATA_OUTPUT_PATH}churn_predictions_{ds_prefix}_train.csv")
        save_predictions(test_df_imputed, best_predictions_test, best_probabilities_test, f"{DATA_OUTPUT_PATH}churn_predictions_{ds_prefix}_test.csv")

        # Full dataset with all the features and the predictions.
        full_df_imputed = train_df_imputed.unionByName(test_df_imputed)
        predictions_full, probabilities_full = make_predictions(best_trained_model, full_df_imputed)
        save_predictions(full_df_imputed, predictions_full, probabilities_full, f"{DATA_OUTPUT_PATH}churn_predictions_{ds_prefix}.csv")

        print(f"\nResults for {ds_name} saved in CSV files.")

    spark.stop()  # close spark session once, after both datasets are processed
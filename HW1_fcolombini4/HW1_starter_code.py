from ast import Return
from typing import Tuple, List
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from sklearn.linear_model import *
from sklearn.ensemble import *

# =========================================================
# STEP 1 - LOAD DATASETS
# =========================================================

def load_datasets(dataset_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training and test datasets.

    Parameters
    ----------
    dataset_path : str
        File path for training dataset.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        train_df : Training dataset
        test_df  : Test dataset
    """

    # TODO:
    # 1. Read dataset CSV file
    read_data = pd.read_csv(dataset_path) #read a csv file
    # 2. Define training and test DataFrames
    train_df = read_data.sample(frac=0.8, random_state=23)  # 80% for training
    test_df = read_data.drop(train_df.index) # Remaining 20% for testing

    # 3. Return both DataFrames
    return train_df, test_df

# =========================================================
# STEP 2 - DATASET OVERVIEW
# =========================================================

def dataset_overview(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Display basic dataset information.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataset.

    test_df : pd.DataFrame
        Test dataset.

    Returns
    -------
    None
    """

    # TODO:
    # 1. Print dataset shapes
    print("Training dataset shape:", train_df.shape)
    print("Test dataset shape:", test_df.shape)

    # 2. Display missing values
    print("\nMissing values in training dataset:")
    print(train_df.isnull().sum())

    print("\nMissing values in test dataset:")
    print(test_df.isnull().sum())

    # 3. Display basic statistics
    print("\nBasic statistics for training dataset:")
    print(train_df.describe())

    print("\nBasic statistics for test dataset:")
    print(test_df.describe())

# =========================================================
# STEP 3 - DEFINE FEATURES AND TARGET
# =========================================================

def split_features_target(train_df: pd.DataFrame, test_df: pd.DataFrame, target_column: str) -> Tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series
]:
    """
    Separate features and target variable.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataset.

    test_df : pd.DataFrame
        Test dataset.

    target_column : str
        Name of target column.

    Returns
    -------
    Tuple[
        pd.DataFrame,
        pd.Series,
        pd.DataFrame,
        pd.Series
    ]
        X_train : Training features
        y_train : Training labels
        X_test  : Test features
        y_test  : Test labels
    """

    # TODO:
    # 1. Split features and labels
    X_train = train_df.drop(columns=[target_column]) #we drop the target column and select all the other variables
    y_train = train_df[target_column] # we select only the target column

    #same as before but for the test dataset
    X_test = test_df.drop(columns=[target_column]) 
    y_test = test_df[target_column]

    return X_train, y_train, X_test, y_test

# =========================================================
# STEP 4 - IDENTIFY COLUMN TYPES
# =========================================================

def identify_column_types(X_train: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identify categorical and numerical columns.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature dataset.

    Returns
    -------
    Tuple[List[str], List[str]]
        categorical_cols : List of categorical columns
        numeric_cols     : List of numerical columns
    """

    # TODO:
    # 1. Identify categorical columns
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist() #select only categorical ones

    # 2. Identify numerical columns
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist() #select only numerical ones

    # 3. Return both lists
    return categorical_cols, numeric_cols

# =========================================================
# STEP 5 - BUILD PREPROCESSOR
# =========================================================

def build_preprocessor(numeric_cols: List[str],  categorical_cols: List[str]) -> ColumnTransformer:
    """
    Create preprocessing pipeline.

    Parameters
    ----------
    numeric_cols : List[str]
        Numerical feature column names.

    categorical_cols : List[str]
        Categorical feature column names.

    Returns
    -------
    ColumnTransformer
        Preprocessing pipeline.

    Expected preprocessing:
    - Missing value handling -> Imputation
    - Feature scaling -> Standardization of numerical features
    - One-hot encoding -> Binarization of categorical variables
    """

    # TODO:
    # 1. Create numerical pipeline
    numeric_pipeline = Pipeline(steps=[
        ('missing_handling', SimpleImputer(strategy='median')), #if there are missing values replace them with the median of the column
        ('standardization', StandardScaler()) #standardize the numerical features to have mean 0 and variance 1
    ])

    # 2. Create categorical pipeline
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), # same as before but we replace the missing values with the most frequent value of the column
        ('binarization', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) #we avoid to add the sparse matrix output of the one hot encoder to the pipeline, 
        #this is made to keep it as a pandas dataframe and make it easier to analyze the results in an intermediate passage
    ])
    
    # 3. Combine using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('numerical', numeric_pipeline, numeric_cols), #apply the pipeline on the numerical columns
            ('categorical', categorical_pipeline, categorical_cols)  #apply the pipeline on the categorical columns
        ],
        remainder='passthrough', #let pass the columns that are not specified in the transformers (if any)
        verbose_feature_names_out=False  # Avoid prefix (numerical_ and categorical_) in output column names
    )


    return preprocessor

# =========================================================
# STEP 6 - BUILD MODEL PIPELINE
# =========================================================

def build_model(preprocessor: ColumnTransformer) -> Pipeline:
    """
    Create machine learning pipeline.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Data preprocessing pipeline.

    Returns
    -------
    Pipeline
        Full machine learning pipeline.
    """

    # TODO:
    # 1. Create Random Forest model
    random_forest = RandomForestClassifier(random_state=101, class_weight='balanced') #random forest classifier
    logistic_regression = LogisticRegression(random_state=101, class_weight='balanced', max_iter=1000) #logistic regression classifier
    #we stop the iterations at 1000 to avoid the possibility that the model doesn't converge, because the default value of 100 is too low for this dataset

    # 2. Combine preprocessing + model
    full_pipeline = {
        'Random Forest': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', random_forest)
        ]),
        
        'Logistic Regression': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', logistic_regression)
        ])}
    # Create different models to compare their performance (BONUS POINT SECTION)
    
    # 3. Return pipeline
    return full_pipeline

# =========================================================
# STEP 7 - TRAIN MODEL
# =========================================================

def train_model(model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    Train machine learning model.

    Parameters
    ----------
    model : Pipeline
        Machine learning pipeline.

    X_train : pd.DataFrame
        Training features.

    y_train : pd.Series
        Training labels.

    Returns
    -------
    Pipeline
        Trained model.
    """

    # TODO:
    # 1. Fit model
    model.fit(X_train, y_train) #fit the model (created with the full_pipeline) on the training data
    
    # 2. Return trained model
    return model

# =========================================================
# STEP 8 - MAKE PREDICTIONS
# =========================================================

def make_predictions(model: Pipeline, X_test: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Generate predictions and probabilities.

    Parameters
    ----------
    model : Pipeline
        Trained machine learning model.

    X_test : pd.DataFrame
        Test features.

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        y_pred : Predicted labels
        y_prob : Prediction probabilities
    """

    # TODO:
    # 1. Predict class labels
    y_predicted_array = model.predict(X_test) #create the array of the predicted labels (0 or 1 binary)
    y_predicted_ = pd.Series(y_predicted_array, index=X_test.index, name='prediction') #transform into a pandas object with their index 

    # 2. Predict probabilities
    y_probabilities_array = model.predict_proba(X_test)[:, 1]  # continues value between 0 and 1 of the exact prob of the model
    y_probabilities = pd.Series(y_probabilities_array, index=X_test.index, name='probability') #same as before

    # 3. Return both outputs
    return y_predicted_, y_probabilities

# =========================================================
# STEP 9 - EVALUATE MODEL
# =========================================================

def evaluate_model(y_test: pd.Series, y_pred: pd.Series, y_prob: pd.Series) -> None:
    """
    Evaluate classification performance.

    Parameters
    ----------
    y_test : pd.Series
        True labels.

    y_pred : pd.Series
        Predicted labels.

    y_prob : pd.Series
        Predicted probabilities.

    Returns
    -------
    None
    """

    # TODO:
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
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")
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

def display_confusion_matrix(y_test: pd.Series, y_pred: pd.Series) -> None:
    """
    Display confusion matrix.

    Parameters
    ----------
    y_test : pd.Series
        True labels.

    y_pred : pd.Series
        Predicted labels.

    Returns
    -------
    None
    """

    # TODO:
    # 1. Compute confusion matrix
    confusion_matrix_created = confusion_matrix(y_test, y_pred) #show TN, FP, FN, TP in the classical confusion matrix format

    confusion_matrix_created_df = pd.DataFrame( #printing only the confusion matrix without labels is not very clear, so we add them
        confusion_matrix_created, 
        index=['Real: 0', 'Real: 1'], 
        columns=['Predicted: 0', 'Predicted: 1']
    )

    # with this the matrix is immediatly readable

    # 2. Display results
    print(f"\nConfusion Matrix: \n{confusion_matrix_created_df}")

# =========================================================
# STEP 11 - FEATURE IMPORTANCE
# =========================================================

def feature_importance_analysis(model: Pipeline, numeric_cols: List[str], categorical_cols: List[str]) -> pd.DataFrame:
    """
    Analyze feature importance.

    Parameters
    ----------
    model : Pipeline
        Trained machine learning pipeline.

    numeric_cols : List[str]
        Numerical feature names.

    categorical_cols : List[str]
        Categorical feature names.

    Returns
    -------
    pd.DataFrame
        DataFrame containing feature importances.
    """

    # TODO:
    # 1. Extract feature names
    model_preprocessor = model[0] #select the preprocessor from the previous pipeline (the first element of the pipeline)
    feature_names = model_preprocessor.get_feature_names_out() #extract the list of the features generated by the preprocessor

    # 2. Extract feature importance values
    model_classifier = model[-1]  #select the random forest classifier from the previous pipeline (the last element of the pipeline)
    # Here we have to implement thi section because feature importance is calculated in different ways for different models
    if hasattr(model_classifier, 'feature_importances_'):
        importance_of_the_features = model_classifier.feature_importances_ #use this selection for tree models like random forest
        
    elif hasattr(model_classifier, 'coef_'): #for the linear models they're named coef
        importance_of_the_features = [abs(val) for val in model_classifier.coef_[0]] #we take the absolute value of the coefficients to have a measure of importance, 
        #in this way we don't consider the direction (positive or negative) and select only the most important

    else:
        importance_of_the_features = [0.0] * len(feature_names) #if the model doesn't have a method to calculate the importance, we set all the values to 0
        # an implementation made just to avoid errors

    # we have to normalize the coefficients to have a measure of importance that is comparable between the different features, 
    # because they can be in different scales, having linear values with a logistic regression
    total_importance = sum(importance_of_the_features)
    if total_importance > 0:
        # We transform the values into percentages to have comparable results
        importance_of_the_features = [(val / total_importance) for val in importance_of_the_features]

    # 3. Create importance DataFrame
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance_of_the_features}) #link together the names and their values
    importance_df = importance_df.sort_values(by='Importance', ascending=False) #order the feature from the most important to the least important

    # 4. Plot top features
    top_n = 5 #we select the top 5 features to plot them in a bar chart
    top_features = importance_df.head(top_n)
    total_importance = importance_df['Importance'].sum()
    top_5_weight = importance_df['Importance'].head(top_n).sum() / total_importance * 100 
    #calculate the importance (%) of the top 5 features over the all features

    plt.figure(figsize=(8, 6)) # increase the size to see better the plot
    plt.barh(top_features['Feature'][::-1],  #we invert the order to have the most important first
             top_features['Importance'][::-1],  #same as before
             color='navy') #add the color
    
    plt.xlabel('Importance') #label of x axis
    plt.suptitle('Top Features for importance', fontsize=18, fontweight='bold') # big title
    plt.title(f'Top {top_n} features account for {top_5_weight:.2f}% of total importance', fontsize=12) #subtitle with the percentage of importance of the top 5 features
    plt.gcf().subplots_adjust(left=0.25) #adjust the left margin to avoid that the feature names are cutted in the plot
    plt.show() #plot
    # 5. Return DataFrame
    return importance_df

# =========================================================
# STEP 12 - SAVE PREDICTIONS
# =========================================================

def save_predictions(test_df: pd.DataFrame, y_pred: pd.Series, y_prob: pd.Series, output_file: str) -> None:
    """
    Save prediction results to CSV file.

    Parameters
    ----------
    test_df : pd.DataFrame
        Original test dataset.

    y_pred : pd.Series
        Predicted labels.

    y_prob : pd.Series
        Prediction probabilities.

    output_file : str
        Output CSV filename.
        
        churn_predictions_dataset1_train.csv
        churn_predictions_dataset2_train.csv
        churn_predictions_dataset1_test.csv
        churn_predictions_dataset2_test.csv
        churn_predictions_dataset1.csv
        churn_predictions_dataset2.csv

        ```
    Returns
    -------
    None
    """

    # TODO:
    # 1. Create output DataFrame
    output_dataframe = test_df.copy() 
    #we copy the input dataset, so we have all the original features and then we can add the predictions on it

    # 2. Add predictions
    output_dataframe['predicted_class'] = y_pred #we add the predictions evaluated in the previuos step
    output_dataframe['predicted_probability'] = y_prob #and also the probabilities of the predictions

    # 3. Save to CSV file
    output_dataframe.to_csv(output_file, index=False) #save the csv


if __name__ == "__main__":

    # TODO:
    #Use this main function to run and test your functions
    datasets_info = {
        "Dataset 1": r"C:\Documenti\UNIMIB\Marketing Analytics\HW1_fcolombini4\dataset1_HW1.csv",
        "Dataset 2": r"C:\Documenti\UNIMIB\Marketing Analytics\HW1_fcolombini4\dataset2_HW1.csv"
    }

    for ds_name, ds_path in datasets_info.items():
        print(f"\nANALYSIS for {ds_name.upper()}")

        train_df, test_df = load_datasets(ds_path) #load the data
        print(f"\n{ds_name} Overview:")
        dataset_overview(train_df, test_df)

        # PREPROCESSING
        X_train, y_train, X_test, y_test = split_features_target(train_df, test_df, target_column="churn") # select the target column for the dataset, in this case "churn"
        categorical_cols, numeric_cols = identify_column_types(X_train) #identify categorica and numerical columns
        preprocessor = build_preprocessor(numeric_cols, categorical_cols) #build the preprocessor with the two lists of columns

        # MODELLING
        models_dict = build_model(preprocessor) #apply the preprocessor to build a random forest model

        best_roc_auc = 0
        best_model_name = ""
        best_trained_model = None
        best_predictions_test = None
        best_probabilities_test = None

        for model_name, pipeline in models_dict.items():
            print(f"Training and evaluating {model_name}...")
            trained_model = train_model(pipeline, X_train, y_train)
            predictions_test, probabilities_test = make_predictions(trained_model, X_test)
            
            # Evaluate the ROC AUC score for the current model
            current_auc = roc_auc_score(y_test, probabilities_test)
            print(f"-> ROC AUC {model_name}: {current_auc:.4f}")
            
            # If the score is better than the best one, update the best model information
            # It always save the first model, and after update it only if the next one is better
            if current_auc > best_roc_auc:
                best_roc_auc = current_auc
                best_model_name = model_name
                best_trained_model = trained_model
                best_predictions_test = predictions_test
                best_probabilities_test = probabilities_test

        print(f"\n[Best Model for AUC {ds_name.upper()}]: {best_model_name} with ROC AUC of {best_roc_auc:.4f}\n")

        # RESULTS
        evaluate_model(y_test, best_predictions_test, best_probabilities_test) #evaluation (accuracy, precision, recall, f score, roc auc)
        display_confusion_matrix(y_test, best_predictions_test) #confusion matrix (TN, FP, FN, TP)
        feature_importance_analysis(best_trained_model, numeric_cols, categorical_cols) #feature importance analysis (top 5 features)
        predictions_train, probabilities_train = make_predictions(best_trained_model, X_train) #predictions on the train dataset, to save them later in csv, with the best model

        #save the data in different csv files
        ds_prefix = ds_name.lower().replace(" ", "") #create a prefix for the file names based on the dataset name to add dinamicity to the code
        save_predictions(X_train, predictions_train, probabilities_train, f"churn_predictions_{ds_prefix}_train.csv")
        save_predictions(X_test, predictions_test, probabilities_test, f"churn_predictions_{ds_prefix}_test.csv")

        #full dataset with all the features and the predictions
        full_df = pd.concat([train_df, test_df])
        X_full = pd.concat([X_train, X_test])
        predictions_full, probabilities_full = make_predictions(best_trained_model, X_full)
        save_predictions(X_full, predictions_full, probabilities_full, f"churn_predictions_{ds_prefix}.csv")
        
        print(f"\nResults for {ds_name} saved in CSV files.")
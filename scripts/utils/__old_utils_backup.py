import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter

def preprocess_data(df):
    """
    Generic preprocessing:
    - Removes non-numeric columns except 'wine_quality_score'
    - Drops constant columns
    - Imputes missing values
    - Scales features to [0, 1]
    - Filters rare classes (classification)
    """
    y = df["wine_quality_score"]
    X = df.drop(columns=["wine_quality_score", "date"], errors="ignore")
    X = X.select_dtypes(include=[np.number])
    X = X.dropna(axis=1, how="all")
    X = X.loc[:, X.std() > 0]
    X = pd.DataFrame(SimpleImputer(strategy="mean").fit_transform(X), columns=X.columns)
    X = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=X.columns)

    # Filter rare classes (for classification)
    if y.dtype == "int" or y.nunique() < 20:
        valid_classes = y.value_counts()[lambda x: x >= 2].index
        mask = y.isin(valid_classes)
        return X[mask], y[mask]

    return X, y

def split_and_resample(X, y):
    """
    Splits into train/test and applies SMOTE to the training set.
    Works for both regression and classification.
    """
    stratify = y if len(np.unique(y)) < 20 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=stratify, test_size=0.25, random_state=42)

    if y.dtype == "int" or len(np.unique(y)) < 20:
        # Classification: Apply SMOTE
        min_class_size = min(Counter(y_train).values())
        k_neighbors = max(1, min(min_class_size - 1, 5))
        sm = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_train, y_train = sm.fit_resample(X_train, y_train)

    return X_train, y_train, X_test, y_test

def save_model(pipeline, model_dir, region):
    """
    Saves a trained pipeline to .joblib with timestamp
    """
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(model_dir, f"{region}_{timestamp}.joblib")
    joblib.dump(pipeline, model_path)
    return model_path

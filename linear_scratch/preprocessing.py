import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def make_preprocessor(numeric_features):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])
    return ColumnTransformer(transformers=[
        ("num", num_pipeline, numeric_features)
    ])

def preprocess(X_train, X_test, numeric_features=None):
    if numeric_features is None:
        numeric_features = list(range(X_train.shape[1]))
    preprocessor = make_preprocessor(numeric_features)
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)
    return X_train_prep, X_test_prep, preprocessor


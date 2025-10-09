import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ColumnNameFixer(BaseEstimator, TransformerMixin):
    """Corrects the column name 'MDEV' to 'MEDV' if it exists."""
    def __init__(self):
        self.column_mapping = {'MDEV': 'MEDV'} #MEDV

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if 'MDEV' in X.columns:
            X.rename(columns=self.column_mapping, inplace=True)
        return X
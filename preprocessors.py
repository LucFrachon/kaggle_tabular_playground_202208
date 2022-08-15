import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder,
    StandardScaler
)

from config import config as cfg


class OrdinalEncoderDF(BaseEstimator, TransformerMixin):
    def __init__(self, unknown_value = -1):
        self.transformer = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=unknown_value
        )
        self.unknown_value = unknown_value
        self.feature_names_in_ = None
        self.n_features_in_ = None
        
    def fit(self, x, y = None):
        self.feature_names_in_ = list(x.columns)
        self.n_features_in_ = len(self.feature_names_in_)
        self.transformer.fit(x)
        return self
    
    def transform(self, x):        
        x_out = self.transformer.transform(x)
        return pd.DataFrame(
            x_out,
            columns=x.columns,
            index=x.index
        )

    def get_feature_names_out(self, input_features = None):
        return np.array(self.feature_names_in_)
        
        
class OneHotEncoderDF(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.transformer = OneHotEncoder(
            sparse=False,
            handle_unknown='ignore'
        )
        self.feature_names_in_ = None
        self.n_features_in_ = None
        self.features_out = None

    def fit(self, x, y = None):
        self.feature_names_in_ = list(x.columns)
        self.n_features_in_ = len(self.feature_names_in_)
        self.transformer.fit(x)
        return self
    
    def transform(self, x):
        x_out = self.transformer.transform(x)
        return pd.DataFrame(
            x_out,
            columns=self.get_feature_names_out(),
            index=x.index
        )

    def get_feature_names_out(self, input_features = None):
        return self.transformer.get_feature_names_out(self.feature_names_in_)
    

class StandardScalerDF(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.transformer = StandardScaler()
        self.feature_names_in_ = None
        self.n_features_in_ = None

    def fit(self, x, y = None):
        self.feature_names_in_ = list(x.columns)
        self.n_features_in_ = len(self.feature_names_in_)
        self.transformer.fit(x)
        return self
    
    def transform(self, x):
        x_out = self.transformer.transform(x)
        return pd.DataFrame(x_out, columns=x.columns, index=x.index)
    
    def get_feature_names_out(self, input_features = None):
        return np.array(self.feature_names_in_)
        

class GroupKNNImputerDF(BaseEstimator, TransformerMixin):
    def __init__(
        self, 
        n_neighbors = 10,
        group_by = None,
        add_indicator = False
    ):
        self.n_neighbors = n_neighbors
        self.add_indicator = add_indicator
        self.transformer = KNNImputer(
            n_neighbors=n_neighbors,
            add_indicator=add_indicator
        )
        self.feature_names_in_ = None
        self.n_features_in_ = None
        self.new_features = None
        self.group_by = group_by if isinstance(group_by, str) else group_by[0]
        self.data_has_grp_var = None
        
    def fit(self, x, y = None):
        self.data_has_grp_var = self.group_by in x.columns
        self.feature_names_in_ = x.columns
        self.n_features_in_ = len(self.feature_names_in_)
        
        if (self.group_by is not None) and self.data_has_grp_var:
            assert x[self.group_by].isnull().sum() == 0, \
                "The variable used to create groups cannot have missing values."
            x_temp = x.drop(columns=self.group_by)
        else:
            x_temp = x
        
        self.transformer.fit(x_temp)
        #
        if self.add_indicator:
            self.new_features = self.transformer.indicator_.get_feature_names_out(
                self.transformer.feature_names_in_
            )

        return self

    def _transform_df(self, _x):
        x_out = self.transformer.transform(_x)

        if _x.isnull().sum().sum() > 0 and self.add_indicator:
            x_out = pd.DataFrame(
                x_out,
                index=_x.index,
                columns=self.transformer.get_feature_names_out(_x.columns)
            )
        elif _x.isnull().sum().sum() == 0 and self.add_indicator:
            # no missing values in _x; sklearn doesn't add indicator columns
            # so we must add them ourselves
            x_out = pd.DataFrame(x_out, index=_x.index, columns=_x.columns)
            # Add flags where we had missing values
            for col in self.new_features:
                x_out[col] = np.where(
                    _x[col.split('missingindicator_')[1]].isnull(), 1, 0)

        else:  # not adding indicator columns
            x_out = pd.DataFrame(x_out, index=_x.index, columns=_x.columns)

        return x_out

    def _group_transform(self, _x):
        x_grp = _x.groupby(self.group_by)         
        x_out = pd.concat(
            [self._transform_df(grp_df.drop(self.group_by, axis=1))
                for g, grp_df in x_grp],
            axis=0
        )
        x_out = pd.concat([_x[[self.group_by]], x_out], axis=1)
        return x_out
        
    def transform(self, x):
        if (self.group_by is not None) and self.data_has_grp_var:
            return self._group_transform(x)
        else:
            return self._transform_df(x)
    
    def get_feature_names_out(self, input_features = None):
        if self.add_indicator:
            return np.concatenate([self.feature_names_in_, self.new_features], axis=0)
        else:
            return np.array(self.feature_names_in_)

    
class ColumnRemover(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_remove = None):
        self.cols_to_remove = (
            cols_to_remove if isinstance(cols_to_remove, list) else [cols_to_remove]
        )
        self.remaining_cols = None
        self.feature_names_in_ = None
        self.n_features_in_ = None

    def fit(self, x, y = None):
        self.feature_names_in_ = list(x.columns)
        self.n_features_in_ = len(self.feature_names_in_)
        self.remaining_cols = [c for c in x.columns if c not in self.cols_to_remove]
        return self
    
    def transform(self, x):
        if self.cols_to_remove is None:
            return x
        else:
            return x[self.remaining_cols]
        
    def get_feature_names_out(self, input_features = None):
        return np.array(self.remaining_cols)


class PreprocessingOperations(BaseEstimator, TransformerMixin):
    def __init__(self, unknown_value, group_by = None, cat_vars = cfg.CAT_VARS):
        self.unknown_value = unknown_value
        self.group_by = group_by
        self.onehot_encoder = OneHotEncoderDF()
        self.ordinal_encoder = OrdinalEncoderDF(
            unknown_value=self.unknown_value
        )
        self.scaler = StandardScalerDF()

        self.feature_names_in_ = None
        self.n_features_in_ = None
        self.cat_vars = cat_vars
        self.noncat_vars = None  # defined in fit()

    def fit(self, x, y = None):
        self.feature_names_in_ = list(x.columns)
        self.n_features_in_ = len(self.feature_names_in_)

        self.cat_vars = [c for c in x.columns if c in self.cat_vars]
        self.noncat_vars = [
            c for c in x.columns if (c not in self.cat_vars) and
            (c != self.group_by)
        ]

        # Separate categorical variables from the rest
        x_cat, x_notcat = x[self.cat_vars], x[self.noncat_vars]

        # One-Hot Encoding of categorical variables (including the grouping variable)
        self.onehot_encoder.fit(x_cat)

        # Ordinal encoding of the grouping variable
        if self.group_by is not None:
            self.ordinal_encoder.fit(x[[self.group_by]])

        # Scaling
        self.scaler.fit(x_notcat)

        return self

    def transform(self, x):
        x = x.copy()
        # Separate categorical variables from the rest
        x_cat, x_notcat = x[self.cat_vars], x[self.noncat_vars]

        x_onehot = self.onehot_encoder.transform(x_cat)
        x_scaled = self.scaler.transform(x_notcat)

        if self.group_by is not None:
            x_grouping = self.ordinal_encoder.transform(x[self.group_by].to_frame())
            return pd.concat([x_grouping, x_scaled, x_onehot], axis=1)
        else:
            return pd.concat([x_scaled, x_onehot], axis=1)

    def get_feature_names_out(self, input_features = None):
        return np.array(
            list(self.ordinal_encoder.get_feature_names_out()) +
            list(self.scaler.get_feature_names_out()) +
            list(self.onehot_encoder.get_feature_names_out())
        )

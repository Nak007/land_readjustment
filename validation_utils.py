'''
Available methods are the followings:
[1] ValidateParams
[2] OptimizeDtypes

Authors: Danusorn Sitdhirasdr <danusorn.si@gmail.com>
versionadded:: 30-06-2026

'''
import numpy as np, pandas as pd
from sklearn.utils.validation import (column_or_1d, check_array, check_consistent_length)
from sklearn.base import BaseEstimator, TransformerMixin

__all__  = ["ValidateParams", 
            "OptimizeDtypes"]

class ValidateParams:
    
    '''
    Functions to validate input and parameters
    '''
    
    def Interval(self, Param, Value, dtype=int, 
                 left=None, right=None, closed="both"):

        '''
        Validate numerical input.

        Parameters
        ----------
        Param : str
            Parameter's name

        Value : float or int
            Parameter's value

        dtype : {int, float}, default=int
            The type of input.

        left : float or int or None, default=None
            The left bound of the interval. None means left bound is -∞.

        right : float, int or None, default=None
            The right bound of the interval. None means right bound is +∞.

        closed : {"left", "right", "both", "neither"}
            Whether the interval is open or closed. Possible choices are:
            - "left": the interval is closed on the left and open on the 
              right. It is equivalent to the interval [ left, right ).
            - "right": the interval is closed on the right and open on the 
              left. It is equivalent to the interval ( left, right ].
            - "both": the interval is closed.
              It is equivalent to the interval [ left, right ].
            - "neither": the interval is open.
              It is equivalent to the interval ( left, right ).

        Returns
        -------
        Value : float or int
            Parameter's value

        '''
        Options = {"left"    : (np.greater_equal, np.less),       # a<=x<b
                   "right"   : (np.greater, np.less_equal),       # a<x<=b
                   "both"    : (np.greater_equal, np.less_equal), # a<=x<=b
                   "neither" : (np.greater, np.less)}             # a<x<b

        f0, f1 = Options[closed]
        c0 = "[" if f0.__name__.find("eq")>-1 else "(" 
        c1 = "]" if f1.__name__.find("eq")>-1 else ")"
        v0 = "-∞" if left is None else str(dtype(left))
        v1 = "+∞" if right is None else str(dtype(right))
        if left  is None: left  = -np.inf
        if right is None: right = +np.inf
        interval = ", ".join([c0+v0, v1+c1])
        tuples = (Param, dtype.__name__, interval, Value)
        err_msg = "%s must be %s or in %s, got %s " % tuples    

        if isinstance(Value, dtype):
            if not (f0(Value, left) & f1(Value, right)):
                raise ValueError(err_msg)
        else: raise ValueError(err_msg)
        return Value

    def StrOptions(self, Param, Value, options, dtype=str):

        '''
        Validate string or boolean inputs.

        Parameters
        ----------
        Param : str
            Parameter's name
            
        Value : float or int
            Parameter's value

        options : set of str
            The set of valid strings.

        dtype : {str, bool}, default=str
            The type of input.
        
        Returns
        -------
        Value : float or int
            Parameter's value

        '''
        if Value not in options:
            err_msg = f'{Param} ({dtype.__name__}) must be either '
            for n,s in enumerate(options):
                if n<len(options)-1: err_msg += f'"{s}", '
                else: err_msg += f' or "{s}" , got %s'
            raise ValueError(err_msg % Value)
        return Value
    
    def check_range(self, param0, param1):
        
        '''
        Validate number range.
        
        Parameters
        ----------
        param0 : tuple(str, float)
            A lower bound parameter e.g. ("name", -100.)
            
        param1 : tuple(str, float)
            An upper bound parameter e.g. ("name", 100.)
            
        '''
        if param0[1] >= param1[1]:
            raise ValueError(f"`{param0[0]}` ({param0[1]}) must be less"
                             f" than `{param1[0]}` ({param1[1]}).")
            
    def check_y_inputs(self, y_proba, y_true=None):
        
        '''
        Validate inputs for calibration models.

        Parameters
        ----------
        y_proba : array-like, shape (n_samples,)
            Predicted probabilities ranges from 0 to 1.
            
        y_true : array-like, shape (n_samples,), default=None
            True binary labels (0/1).
            
        Returns
        -------
        y_proba : ndarray shape (n_samples,)
            Validated probabilities.
            
        y_true : ndarray shape (n_samples,)
            Validated labels. This returns None when `y_ture` is not 
            provided.
            
        '''
        # Ensure shapes
        y_proba = column_or_1d(y_proba)
        y_proba = check_array(y_proba, ensure_2d=False, dtype=float)

        # Probability range
        eps = np.finfo(float).eps
        y_proba = np.clip(y_proba, eps, 1 - eps)
        
        if y_true is not None:
            
            # Check lengths
            y_true = column_or_1d(y_true)
            check_consistent_length(y_proba, y_true)

            # Check that y_true only contains {0,1}
            unique_labels = np.unique(y_true)
            if not np.all(np.isin(unique_labels, [0, 1])):
                raise ValueError("`y_true` must be binary (0 or 1).")
            return y_proba, y_true
        
        else: return y_proba, None
        
    def check_class_weight(self, class_weight):
        
        '''
        Validate the `class_weight` parameter.

        Parameters
        ----------
        class_weight : {dict, "balanced"} or None
            Weights associated with classes in the form `{label: weight}`.
            If set to "balanced", the weights are automatically adjusted
            inversely proportional to class frequencies.

        Returns
        -------
        class_weight : dict, str or None
            Normalized class_weight.
            
        '''
         # Validate the `class_weight` parameter.
        if class_weight is None:
            return None
        elif isinstance(class_weight, str):
            args = ("class_weight", class_weight, ["balanced"], str)
            return self.StrOptions(*args)
        elif isinstance(class_weight, dict):
            return class_weight
        else:raise ValueError(f"Invalid value for `class_weight`: {class_weight}. "
                              f"Expected dict, 'balanced', or None.")
            
    def check_estimator(self, estimator):
        
        '''
        Validate that an estimator follows the scikit-learn regression 
        estimator interface.

        Parameters
        ----------
        estimator : object
            Estimator instance.

        Raises
        ------
        ValueError
            If estimator does not implement required methods/attributes.
            
        '''
        for m in ["get_params", "set_params", "fit", "predict", "predict_proba"]:
            if not callable(getattr(estimator, m, None)):
                raise ValueError(f"Estimator missing required method `{m}()`.")

        if estimator.get_params().get("fit_intercept", None) is None:
            raise ValueError("Estimator missing required parameter "
                             "`fit_intercept` in get_params().")
        
        # Fit model with dummy X and y
        N, rnd = 10, np.random.RandomState(0)
        estimator.fit(rnd.rand(N).reshape(-1,1), 
                      rnd.randint(0, 2, size=N))
        
        for a in ["coef_", "intercept_"]:
            if getattr(estimator, a, None) is None:
                raise ValueError(f"Estimator missing required attribute "
                                 f"`{attr}` after fit().")

    def check_bins(self, bins, data=None):
        
        '''
        Validate 'bins'

        Parameters
        ----------
        bins : int, sequence
            Number of bins or sequence to divide probabilities.
            
        Returns
        -------
        bins : int, sequence
            
        '''
        # Case 1: bins is integer
        if isinstance(bins, int):
            if bins <= 0:
                raise ValueError(f"Number of bins must be positive,"
                                 f" got {bins}.")

        # Case 2: bins is a sequence of bin edges
        elif isinstance(bins, (list, np.ndarray)):
            
            bins = np.asarray(bins)
            if bins.ndim != 1:
                raise ValueError("Bin edges must be a 1D sequence")
            elif not np.all(np.isfinite(bins)):
                raise ValueError("Bin edges contain non-finite values (NaN/inf)")
            elif not np.all(np.diff(bins) > 0):
                raise ValueError("Bin edges must be strictly increasing")
            elif len(bins) < 2:
                raise ValueError("At least two bin edges are required")
            else: pass

        else: raise TypeError("bins must be int or sequence")
        
        return bins

class OptimizeDtypes(BaseEstimator, TransformerMixin, ValidateParams):
    
    '''
    Optimize DataFrame column dtypes for improved memory efficiency
    while preserving data integrity and usability.

    Parameters
    ----------
    max_category : int, default=100
        Maximum number of unique non-null values allowed for converting
        object columns to categorical dtype.

    max_category_ratio : float or None, default=None
        Maximum allowed ratio of unique non-null values relative to the
        total number of non-null observations for categorical conversion.

        Example:
        - 0.05 means convert to category only if unique_ratio <= 5%

        If None, ratio-based filtering is disabled.

    float_tolerance : float, default=1e-8
        Absolute tolerance used to determine whether floating-point
        values are effectively integer-like during dtype inference.

    Notes
    -----
    - NaN values are preserved.
    - Float downcasting considers numeric range only, not precision loss.
    - Integer-like float columns (e.g. [1.0, 2.0]) may be converted to
      integer dtypes when possible.
    - Datetime and timedelta columns are preserved unchanged.
    
    '''
    
    # Numerical types
    num_types = {"iinfo" : [np.int8, np.int16, np.int32, np.int64],
                 "finfo" : [np.float16, np.float32, np.float64]}

    def __init__(self, max_category=100, max_category_ratio=None, 
                 float_tolerance=1e-8):
        
        # Validate all parameters
        self.max_category = self.Interval("max_category", 
                                          max_category, 
                                          int, 2, None, "left")
      
        if max_category_ratio is not None:
            self.max_category_ratio = self.Interval("max_category_ratio", 
                                                    max_category_ratio, 
                                                    float, 0., 1., "both")
        else: self.max_category_ratio = max_category_ratio
        
        eps = np.finfo(float).eps
        self.float_tolerance = self.Interval("float_tolerance", 
                                             float_tolerance, float, 
                                             eps, 1 - eps, "both")
                
    def fit(self, X):

        '''
        This function analyzes each column in the input DataFrame and
        determines the most memory-efficient dtype that preserves the
        underlying data characteristics. The inferred dtypes are stored
        in `self.dtypes_` and later applied using `transform()`.

        Conditions:
        - Integer-like numeric columns → smallest compatible integer dtype
        - Continuous numeric columns → smallest compatible float dtype
        - Boolean columns → int8
        - Low-cardinality object columns → category
        - High-cardinality object columns → object
        - Datetime and timedelta columns → preserved unchanged

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame used to infer optimized column dtypes.

        Returns
        -------
        self
            Fitted dtype optimizer.

        Attributes
        ----------
        dtypes_ : dict
            Dictionary mapping column names to inferred optimized dtypes.

        Notes
        -----
        - This method only learns optimized dtypes and does not modify
          the input DataFrame.
        - Columns containing only missing values are ignored.
        - Integer-like detection uses `float_tolerance` to account for
          floating-point precision noise.
        - Object columns may be converted to categorical dtype depending
          on `max_category` and `max_category_ratio`.

        '''

        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"`X` must be pd.DataFrame. Got {type(X)}.")
        
        self.dtypes_ = dict()
        
        for col in X.columns:
            
            x = X[col]

            # Preserve datetime / timedelta
            if (pd.api.types.is_datetime64_any_dtype(x) or 
                pd.api.types.is_timedelta64_dtype(x)):
                
                continue

            # Boolean
            elif pd.api.types.is_bool_dtype(x):
                
                self.dtypes_[col] = np.int8

            # Numeric
            elif pd.api.types.is_numeric_dtype(x):

                # Skip fully missing columns
                a = x.to_numpy(dtype=np.float64)
                if np.isnan(a).all(): continue

                # Determine integer-like
                kwargs = dict(atol=self.float_tolerance, equal_nan=True)
                is_int = np.all(np.isclose(a, np.round(a), **kwargs))
                info_type = "iinfo" if is_int else "finfo"
                
                # Max-Min range
                a_min = np.nanmin(a)
                a_max = np.nanmax(a)

                # Numerical optimization
                for dtype in self.num_types[info_type]:
                    limits = getattr(np, info_type)(dtype)
                    if (a_min >= limits.min) & (a_max <= limits.max): 
                        attr = f"{dtype.__name__.capitalize()}Dtype" 
                        self.dtypes_[col] = getattr(pd, attr)()
                        break

            # Object / string 
            elif pd.api.types.is_object_dtype(x):

                # Check number of unique elements in x
                n_unique = x.nunique(dropna=True)
                n_total = x.notna().sum()
                use_category = n_unique <= self.max_category
                
                # Determine ratio of uniqueness
                if self.max_category_ratio is not None:
                    unique_ratio = n_unique / n_total if n_total > 0 else 1
                    use_category &= (unique_ratio <= self.max_category_ratio)
                
                if use_category: self.dtypes_[col] = "category"
                else: self.dtypes_[col] = "object"

            # Category
            elif isinstance(x.dtype, pd.CategoricalDtype):
                
                self.dtypes_[col] = "category"
        
        return self
    
    def transform(self, X):

        '''
        Transform DataFrame columns to the optimized dtypes determined
        during fitting.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        X_transformed : pd.DataFrame
            DataFrame with converted column dtypes based on `self.dtypes_`.
        
        Attributes
        ----------
        memory_report : dict
            Summary of DataFrame memory usage before and after dtype
            optimization. Memory statistics are reported in megabytes,
            where 1 MB = 1024² bytes.

            Contains:
            - before_mb : Memory usage of the original DataFrame.
            - after_mb  : Memory usage of the optimized DataFrame.
            - delta_mb  : Reduction in memory usage.
            - delta_pct : Percentage reduction in memory usage.
            
        failed_dtypes_ : dict
            Dictionary containing columns whose dtype optimization failed
            during transformation, along with the corresponding error
            messages and original dtypes.
            
        Notes
        -----
        - Only columns stored in `self.dtypes_` are transformed.
        - Columns not seen during fitting are preserved unchanged.
        - Conversion failures are ignored and original dtype is retained.
        
        '''
        # Validation
        if not hasattr(self, "dtypes_"):
            raise AttributeError("This transformer is not fitted yet. "
                                 "Call `fit()` before `transform()`.")

        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"`X` must be pd.DataFrame. Got {type(X)}.")

        X_transformed = X.copy()
  
        # Apply learned dtypes
        self.failed_dtypes_ = dict()
        for col, dtype in self.dtypes_.items():
            
            if col not in X_transformed.columns:
                continue
            
            try: X_transformed[col] = X_transformed[col].astype(dtype)
            except Exception as e:
                warn(f"Column '{col}' could not be converted "
                     f"to {dtype}. Original dtype "
                     f"'{X_transformed[col].dtype}' was retained. "
                     f"Reason: {e}", UserWarning)
                
                self.failed_dtypes_[col] = {"target_dtype"  : str(dtype),
                                            "original_dtype": str(X_transformed[col].dtype),
                                            "reason"        : str(e)}
        
        # DataFrame memory usage
        self.memory_report = self.memory_usage(X, X_transformed)
    
        return X_transformed

    def fit_transform(self, X):

        '''
        Fit the transformer and apply dtype optimization to the input
        DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        X_transformed : pd.DataFrame
            DataFrame with optimized column dtypes.
            
        '''
        return self.fit(X).transform(X)

    def memory_usage(self, df_before, df_after):
        
        '''
        Generate memory usage comparison report between two DataFrames.

        Parameters
        ----------
        df_before : pd.DataFrame
            Original DataFrame before optimization.

        df_after : pd.DataFrame
            Optimized DataFrame after dtype conversion.

        Returns
        -------
        memory_report : dict
            Summary of DataFrame memory usage before and after dtype
            optimization. Memory statistics are reported in megabytes,
            where 1 MB = 1024² bytes.

            Contains:
            - before_mb : Memory usage of the original DataFrame.
            - after_mb  : Memory usage of the optimized DataFrame.
            - delta_mb  : Reduction in memory usage.
            - delta_pct : Percentage reduction in memory usage.
            
        '''
        # Convert bytes → MB        
        before_mb = df_before.memory_usage(deep=True).sum() / 1024**2
        after_mb  = df_after.memory_usage(deep=True).sum() / 1024**2
        
        # Optimized delta
        delta_mb  = before_mb - after_mb
        delta_pct = round(delta_mb / before_mb,4) if before_mb > 0 else 0
        num_format = lambda x : float(np.round(x, 4))
        
        return {"before_mb": num_format(before_mb),
                "after_mb" : num_format(after_mb),
                "delta_mb" : num_format(delta_mb),
                "delta_pct": num_format(delta_pct)*100}
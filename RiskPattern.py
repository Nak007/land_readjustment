'''
Available methods are the followings:
[1] Monitoring
[2] ModifiedEncoder

Authors: Danusorn Sitdhirasdr <danusorn.si@gmail.com>
versionadded:: 29-02-2024

'''
import pandas as pd, numpy as np, re
from collections import namedtuple, OrderedDict
import AssoruleMining as ARM
from AssoruleMining import *
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

from itertools import product
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import matplotlib.transforms as transforms

__all__ = ["Monitoring", "ModifiedEncoder"]

class ValidateParams:
    
    '''Validate parameters'''
    
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
        Options = {"left"    : (np.greater_equal, np.less), # a<=x<b
                   "right"   : (np.greater, np.less_equal), # a<x<=b
                   "both"    : (np.greater_equal, np.less_equal), # a<=x<=b
                   "neither" : (np.greater, np.less)} # a<x<b

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

class SetParameters:
    
    # Set1 : ['#1f77b4', '#ff7f0e', '#2ca02c', '#E9EAEC']
    # Set2 : ['#1B9CFC', '#FAD02C', '#FC427B', '#E9EAEC']
    # Set3 : ["#677880", "#292400", "#EF3340", "#EBECE0"]
        
    def __init__(self):
        
        '''Initial parameters'''
        self.n_xticks = 9
        self.n_yticks = 6
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#E9EAEC']
        self.format = "{:,.2%}".format
        
    def __tickers__(self, ax, axis="y", percent=False):
        
        '''Set tickers'''
        attr0 = f"n_{axis}ticks".lower()
        attr1 = f"{axis}axis".lower()
        t = mpl.ticker.MaxNLocator(getattr(self, attr0))
        getattr(ax, attr1).set_major_locator(t)
        ax.tick_params(axis=axis, labelsize=10)
        if percent:
            t = ticker.PercentFormatter(xmax=1)
            getattr(ax, attr1).set_major_formatter(t)
        
    def __legend__(self, ax, patches, labels):
        
        '''Set legend to the upper right corner'''
        kwds = dict(loc='upper left', edgecolor="none",
                    prop=dict(weight="ultralight", size=12), 
                    ncol=1, 
                    borderaxespad=0.2, 
                    bbox_to_anchor=(1.02, 1.0), 
                    markerscale=0.5,
                    columnspacing=0.3, 
                    handletextpad=0.2)
        ax.legend(patches, labels, **kwds)
        
    def __scalelimit__(self, ax, axis="y", scale=0.9):
        
        '''Set axis limit'''
        a_min, a_max = getattr(ax, f"get_{axis}lim")()
        getattr(ax,f"set_{axis}lim")(a_min, a_max/scale)
        
    def __axzorder__(self, ax):
        
        '''Rearrange zorder of axes'''
        ax.set_zorder(1)  # default zorder is 0 for ax1 and ax2
        ax.patch.set_visible(False) # prevents ax1 from hiding ax2
        
    def __spine__(self, ax, spine_types:list, visible=True):
        
        '''Set spine visibiliy'''
        if not isinstance(spine_types, list):
            spine_types = [spine_types]
        for spine in spine_types:
            ax.spines[spine].set_visible(visible)
            
    def __vbartext__(self, ax, bars, fontsize=10, rotation=90, 
                     num_format="{:,.0f}".format):
        
        '''Set annotation for vertial bar chart'''
        def_text = dict(textcoords='offset points', ha="center",
                        fontsize=fontsize, rotation=rotation)
        pos_text = {"inside" : {**dict(va="top", xytext=(0,-2), 
                                       color="grey"), **def_text}, 
                    "outside": {**dict(va="bottom", xytext=(+2,0), 
                                       color="black"), **def_text}}
        
        renderer = plt.gcf().canvas.get_renderer()
        for bar in bars:

            xy = (bar.get_x() + bar.get_width()/2, bar.get_height())
            args = (num_format(xy[1]), xy)
            
            # Render shape
            txt = ax.annotate(*args, **pos_text["inside"])
            bar_bbox = bar.get_window_extent(renderer=renderer)
            txt_bbox = txt.get_window_extent(renderer=renderer)

            # If the text overflows bar then draw it above the bar
            if txt_bbox.height * 1.15 > bar_bbox.height: 
                txt.remove()
                ax.annotate(*args, **pos_text["outside"])

class CalculateOKR:
    
    okr_segments = ["modl_seg", "fico_grp", "sa_se"]
    
    def __findokr__(self, X, indicator='flg_90pM04'):
        
        '''
        Get Objective-Key-Results (OKR) indicator
        
        Parameters
        ----------
        X : pd.DataFrame
            The data to be used to determine OKR segment. `X` must contain 
            columns, and values as follows
            
            Column      Value
            ------      -----
            "modl_seg"  {'1]NTB', '2]NTC', '3]ETC'}
            "fico_grp"  {'01]No-FICO', '02]FICO'}
            "sa_se"     {'01]SA', '02]SE'}
            
        indicator : str, default='flg_90pM04'
            Name of OKR indicator under "okr_type" column.
            
        Returns
        -------
        okr_value : float
        
        '''
        # Find unique values for respective criteria
        values = X[self.okr_segments].drop_duplicates().to_dict()
        values = dict([(k, set(v.values())) for k,v in values.items()])

        # Find Objective Key Result (OKR)
        cond = self.okrs["okr_type"]==indicator
        for c in self.okr_segments:
            if len(values[c])>1: cond &= self.okrs[c].isin(["ALL"])
            else: cond &= self.okrs[c].isin(values[c])
        okr_value = self.okrs.loc[cond, "modified_okr"].values[0]
        return float(okr_value)

class CreateResult:
    
    periods = [3, 6, 9, 12]

    def __CreateResult__(self):
        
        '''
        Create report
        
        Attributes
        ----------
        results_ : dict
            A dict with keys as column headers and values as columns, that 
            can be imported into a pandas DataFrame. All scores are 
            available in the results_ dict with defined key formats as 
            follows:
            
            - '<ind>_latest' : the latest value of indicator
            - '<ind>_okr' : okr of indicator
            - '<ind>_delta' : '<ind>_latest' - '<ind>_okr' (difference)
            - '<var>_Last<period>M' : multi-period coefficients (slopes) 
              of variable
            
        '''
        self.results_ = dict()
        for n,key in enumerate(self.content_.keys()):
            data = self.content_[key]
            info = {"pattern" : key}
            info.update(self.__extract__(data))
            info.update(self.__latest__(data))
            info.update(self.__indcoef__(data))
            info.update(self.__chgcoef__(data))
            for k,v in info.items():
                try: self.results_[k] = np.r_[self.results_[k],[v]]
                except: self.results_[k] = np.r_[[v]]

    def __extract__(self, data):
        
        '''Extract only str, int, and float from fields'''
        values = dict()
        for field in data._fields:
            value = getattr(data,field)
            if (value is None) | isinstance(value, (str,int,float)):
                values[field] = value
        return values
    
    def __latest__(self, data):
        
        '''Lastest value of indicators'''
        values = dict()
        for field in data.indicator._fields:
            t = getattr(data.indicator, field)
            values.update({f"{field}_latest": t.values[-1], 
                           f"{field}_okr": t.okr,
                           f"{field}_delta": t.values[-1] - t.okr})
        return values
    
    def __slope__(self, x, field):
        
        '''Calculate slope of x'''
        strformat = "{}_Last{}M".format
        x0 = np.arange(len(x)).reshape(-1,1)
        x1 = np.array(x).reshape(-1,1)
        
        coefs = dict()
        reg = LinearRegression(fit_intercept=True)
        for n in self.periods:
            reg.fit(x0[-n:,:], x1[-n:,:])
            key = strformat(field, str(n).zfill(2))
            coefs[key] = reg.coef_[0][0]
        return coefs
    
    def __indcoef__(self, data):
        
        '''Calculate beta of indicators'''
        coefs = dict()
        for field in data.indicator._fields:
            t = getattr(data.indicator, field)
            coefs.update(self.__slope__(t.values, field))
        return coefs
    
    def __chgcoef__(self, data):
        
        '''Calculate beta of % change'''
        coefs = dict()
        for field in ["limit", "count"]:
            values = getattr(data, field)
            change = np.diff(values) / values[:-1]
            coefs.update(self.__slope__(change, field))
        return coefs

class RuleExpression:
    
    const = {False : ["not in", "<="],
             True  : ["in", ">"]}
    
    '''
    Create expression statement from rule generated from 
    `TreeRuleMining`
    
    Attributes
    ----------
    expr : dict
        For each key, it contains a list of a set of expression 
        statments that can be used in pd.DataFrame.query.
        
    '''
    def __expression__(self, estimator):
        
        '''
        Create expression statement.
        
        Parameters
        ----------
        estimator : TreeRuleMining instance
            An instance must only be fitted with data that processed by
            `ModifiedEncoder`.
        
        Returns
        -------
        self
        
        '''
        if not isinstance(estimator, ARM.TreeRuleMining):
            raise ValueError(f"`estimator` must be TreeRuleMining. "
                             f"Got {type(estimator)} instead.")
        
        self.expr_ = dict()
        self.cond_ = dict()
        for key in estimator.rules.keys():
            rule = estimator.rules[key].rule
            self.cond_[key] = self.__cond__(rule)
            self.expr_[key] = self.__expr__(self.cond_[key])
        return self
    
    def __split__(self, name):
        
        '''Split column and value'''
        n = name[::-1].rfind("(")
        return name[:-n-1].strip(), name[-n:-1].strip()
    
    def __cond__(self, rule):
        
        '''Find condition'''
        cond = dict()
        for name, operand, _ in rule:
            col, val = self.__split__(name)
            if col not in cond.keys(): 
                cond[col] = {">":[], "<=": []}
            cond[col][operand].append(val)
        return cond
    
    def __expr__(self, rule):
        
        '''Create expression to be evaluated'''
        expr = []
        for key in rule.keys():
            logic, sign = self.const[len(rule[key][">"])>0]
            val = [f"'{v}'" for v in rule[key][sign]]
            val = ",".join(sorted(val))
            expr += [f"(`{key}` {logic} ({val}))"]
        return expr

class PlotIndicator:
    
    def plot_base(self, display=None, which=None, ax=None):
        
        '''
        Plot OKRs.
        
        Parameters
        ----------
        display : list of str, default=None
            List of OKR indicators to be displayed. If None, all indicators
            are selected.
        
        ax : matplotlib axis, default=None
            If None, it defaults to predefined axis instance.
            
        Returns
        -------
        ax : matplotlib axis
            The axis instance.
        
        '''
        # Check whether it is fitted
        if getattr(self, "content_", None) is None:
            raise ValueError("This instance is not fitted yet. "
                             "Call 'fit' with appropriate arguments "
                             "before using this instance.")

        # Validate indicators
        if isinstance(display, list):
            for flag in display:
                self.StrOptions('display', flag, self.flags, str)
            flags = display.copy()
        elif display is None: flags = self.flags
        else: raise ValueError(f"`display` must be list of str. "
                               f"Got {type(display)} instead.")
            
        # Validate which and key index
        which = self.StrOptions('which', which, self.rules_, str)
        index = self.__index__(which)
        cohort = self.content_[index].cohort
    
        # Create axis
        ax1 = plt.subplots(figsize=(10,4.5))[1] if ax is None else ax
        ax2 = ax1.twinx()
        patches, labels = [], []
        
        # Plot indicators and corresponding thresholds
        for n,key in enumerate(flags):
            
            # Indicator values
            indicator = getattr(self.content_[index].indicator, key)

            # Plot line
            values = indicator.values
            kwds = dict(color=self.colors[n], lw=4, solid_capstyle="round")
            patches += [ax1.plot(cohort[:len(values)], values, **kwds)[0]]
            
            # OKR threshold
            okr = indicator.okr
            kwds = dict(lw=1, ls="--", color=self.colors[n])
            patches += [ax1.axhline(indicator.okr, **kwds)]
            
            # labels
            labels += [f"{key} ({self.format(indicator.values[-1])})", 
                       f"OKR ({self.quarter}, {self.format(okr)})"]
        
        # Plot bar chart (final credit limit)
        limit = self.content_[index].limit / 10**6
        patches += [ax2.bar(cohort, limit, color=self.colors[3])]
        labels  += ["Credit Limit (MB)"]
        self.__vbartext__(ax2, patches[-1])
        
        # Number and format of tickers
        self.__tickers__(ax1, "y", True)
        self.__tickers__(ax1, "x", False)
        
        # Limit
        self.__scalelimit__(ax1, "y", 0.95)
        self.__scalelimit__(ax2, "y", 0.85)
        ax1.set_xlim(-1, len(limit))

        # Axis labels
        ax1.set_xlabel("Month", fontsize=13)
        ax1.set_ylabel("Indicator", fontsize=13)
        ax1.set_title(f"Product : {self.product}, "
                      f"As-of : {self.asof}, "
                      f"Pattern : {which}", fontsize=13)
        
        # Visibility of spines and legend
        ax2.axes.get_yaxis().set_visible(False)
        self.__spine__(ax1, ["top", "right"], False)
        self.__spine__(ax2, ["top", "right"], False)
        self.__axzorder__(ax1)
        self.__legend__(ax1, patches, labels)
        ax1.grid(False)
        ax2.grid(False)
        plt.tight_layout()
        
        return ax1

class Monitoring(ValidateParams, SetParameters, CalculateOKR, 
                 RuleExpression, PlotIndicator, CreateResult):
    
    '''
    Create pattern results and plot indicators against Objectives and 
    Key Results (thresholds)
    
    Parameters
    ----------
    okrs : pd.DataFrame
        The data to be used to determine OKR threshold. `okrs` must 
        contain columns, and values (subject to change) as follows
        
        Column          Value
        ------          -----
        "product_tp"    {'CC', 'XPC', 'KPL', "ALL"}
        "modl_seg"      {'1]NTB', '2]NTC', '3]ETC'}
        "fico_grp"      {'01]No-FICO', '02]FICO'}
        "sa_se"         {'01]SA', '02]SE'}
        "okr_type"      {"flg_06pM01", "flg_60pM03", "flg_90pM04"}
        "quarter"       {"q1", "q2", "q3", "q4"}
        "modified_okr"  float
  
    estimator : TreeRuleMining instance, default=None
        An instance must only be fitted with data that processed by
        `ModifiedEncoder`. If None, self.expr_ or query expression is
        assigned to an empty dict. 

    product : str, default="ALL"
        Name of credit product.
        
    quarter : {"q1", "q2", "q3", "q4"}, default="q4"
        ith quarter of OKR.
    
    Attributes
    ----------
    expr_ : dict
        For each key, it contains a list of a set of expression 
        statments that can be used in pd.DataFrame.query.
        
    '''
    def __init__(self, okrs, estimator=None, product="ALL", quarter="q4"):

        # Validate product
        unq_products = list(okrs["product_tp"].unique())
        self.product = self.StrOptions('product', product, unq_products, str)
        
        # Validate quarter
        unq_quarters = ["q1", "q2", "q3", "q4"]
        self.quarter = self.StrOptions('quarter', quarter, unq_quarters, str)
        
        # OKR table
        cond = (okrs["product_tp"]==self.product) & (okrs["quarter"]==self.quarter)
        self.okrs = okrs.loc[cond].reset_index(drop=True).copy()
        self.flags = list(sorted(self.okrs["okr_type"].unique()))
        
        # Create rule expression
        super().__init__()
        self.estimator = estimator
        if self.estimator is not None: 
            self.__expression__(self.estimator)
        else: self.expr_ = dict()
        
    def fit(self, X, queries=None):
        
        '''
        Fit X.
        
        Parameters
        ----------
        X : pd.DataFrame
            The data to be used to determine OKR segment and indicator. 
            `X` must contain columns, and values (subject to change) as 
            follows
            
            Column      Value
            ------      -----
            "modl_seg"  {'1]NTB', '2]NTC', '3]ETC'}
            "fico_grp"  {'01]No-FICO', '02]FICO'}
            "sa_se"     {'01]SA', '02]SE'}
            indicator   {0, 1}, it must also exist in `okrs` under 
                        "okr_type" column e.g. "flg_90pM04".
            "Rule_XX"   {True, False}, rule columns that created from 
                        `TreeRuleMining` instance e.g. "Rule_1". This is
                        not required when `queries` is provided.
                        
        queries : dict, default=None
            A dict with keys as rule number and values as query expression 
            (str) to evaluate in pd.DataFrame.query e.g. {"Rule_1":(`A`=1) 
            and (`B` in (3,4))}. The rules will be generated corresponding 
            to the number of valid keys found in queries. If `queries` is 
            provided, it will override all rules contain inside `estimator`.
            
        Attributes
        ----------
        content_ : collections.OrderedDict
            A dict with Keys as rule-ids and its value is namedtuple with 
            the following fields:
            - "product"   : Product name
            - "asof"      : Maximum date (YYYY-MM) of X
            - "quarter"   : nth quarter of OKR
            - "name"      : Pattern name
            - "query"     : Query expression (pd.query)
            - "indicator" : Tuple of indicators
            - "limit"     : Sum of credit limits
            - "count"     : Number of applications
            - "cohort"    : Label of cohorts
        
        results_ : dict
            A dict with keys as column headers and values as columns, that 
            can be imported into a pandas DataFrame. All scores are 
            available in the results_ dict with defined key formats as 
            follows:
            
            - '<ind>_latest' : the latest value of indicator
            - '<ind>_okr' : okr of indicator
            - '<ind>_delta' : '<ind>_latest' - '<ind>_okr' (difference)
            - '<var>_Last<period>M' : multi-period coefficients (slopes) 
              of variable
              
        Returns
        -------
        self
            
        '''
        # Intialize parameters
        X_new = X.copy()
        
        # Create rules from valid queries and columns
        if isinstance(queries, dict):
            X_out = self.transform(X, queries)
            X_new = X_new.merge(X_out, left_index=True, right_index=True)

        # Select rules from columns in X_
        if self.estimator is None:
            arm_rules = [c for c in list(X_new) 
                         if len(re.findall("^Rule_[1-9]",c))]
        
        # Otherwise select rules from estimator
        else: arm_rules = list(self.estimator.rules.keys())
        self.rules_ = ["Rule_all", "Rule_0"] + arm_rules
        self.asof = X_new["cohort_fr"].max()
        
        # Create Rule_all and Rule_0
        triggers = X_new[arm_rules].astype(int).sum(1)
        X_new["Rule_0"] = triggers==0 # not triggered
        X_new["Rule_all"] = True # Portfolio
        
        # Initialize namedtuple
        keys = ['product', 'asof', 'quarter', 'name', 'query', 
                'indicator', 'limit', 'count', 'cohort']
        Pattern = namedtuple('Pattern', keys)
        IndData = namedtuple('IndData', list(self.flags))
        Values  = namedtuple('values', ["values", "okr"])
 
        # Create content from all rules.
        self.content_ = OrderedDict()
        for which in self.rules_:
            
            cond = X_new[which].astype(bool)==True
            inds = dict()
            for key in self.flags:
                data = X_new.loc[cond].groupby("cohort_fr").agg({key:"mean"})
                data = data.loc[data[key].notna()].values.flatten()
                inds[key] = Values(data, self.__findokr__(X_new, key))
            
            # Credit limits
            limits = X_new.loc[cond].groupby("cohort_fr")\
            .agg({"fnl_cr_lmt":["sum", "count"]})

            # Query expression for pd.query
            query = (" and ".join(self.expr_[which]) 
                     if which in self.expr_.keys() else None)
                   
            # New pattern is assigned to the next index.
            key  = self.__index__(which)
            self.content_[key] = Pattern(product=self.product, 
                                         asof=self.asof, 
                                         quarter=self.quarter,
                                         name=which,
                                         query=query,
                                         indicator=IndData(**inds), 
                                         limit=limits.values[:,0].flatten(), 
                                         count=limits.values[:,1].flatten(),
                                         cohort=np.array(limits.index))
        
        # Create report
        self.__CreateResult__()

        return self
    
    def __index__(self, which):
        
        '''Create key index'''
        yyyymm = "".join(self.asof.split("-"))
        rindex = which.split("_")[-1].zfill(3)            
        return f"P{yyyymm}{rindex}".upper()
    
    def transform(self, X, queries):
        
        '''
        Transform X into rules based on queries.
        
        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_columns)
            The data to be queried to generate rules.

        queries : dict, default=None
            A dict with keys as rule number and values as query expression 
            (str) to evaluate in pd.DataFrame.query e.g. {"Rule_1":(`A`=1) 
            and (`B` in (3,4))}. The rules will be generated corresponding 
            to the number of valid keys found in queries. 
            
        Returns
        -------
        X_out : pd.DataFrame of shape (n_samples, n_rules)
            Tranformed X. 
        
        '''
        if not isinstance(queries, dict):
             raise ValueError(f"`queries` must be dict. "
                              f"Got {type(queries)} instead.")
        else: rules = [c for c in queries.keys() 
                       if len(re.findall("^Rule_[1-9]",c))]
        
        X_out = dict()
        for key in rules:
            index = X.query(queries[key]).index
            X_out[key] = np.where(X.index.isin(index), True, False)
                
        return pd.DataFrame(X_out)
 
    def plot(self, display=None, which=None, ax=None):
        
        '''
        Plot OKRs.
        
        Parameters
        ----------
        display : list of str, default=None
            List of OKR indicators to be displayed. If None, all indicators
            are selected.
        
        ax : matplotlib axis, default=None
            If None, it defaults to predefined axis instance.
            
        Returns
        -------
        ax : matplotlib axis
            The axis instance.
        
        '''
        return self.plot_base(display, which, ax)

class ModifiedEncoder:
    
    '''
    Modified OneHotEncoder, sklearn version 1.0.2, that encodes 
    categorical features as a one-hot numeric array. The new features 
    under same category are mutually exclusive.
    
    Parameters
    ----------
    categories : list
        A list of column labels to be encoded.
        
    kwargs : dict, default=None
        Keyword arguments for OneHotEncoder. If None, it defaults to
        {handle_unknown : 'ignore'}.
    
    '''
    def __init__(self, categories, kwargs=None):
        
        self.categories_ = categories
        enc_kwargs = dict(handle_unknown='ignore')
        if isinstance(kwargs, dict): enc_kwargs.update(kwargs)
        self.encoder = OneHotEncoder(**enc_kwargs)
    
    def fit(self, X):
        
        '''
        Fit OneHotEncoder to X.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            The data to determine the categories of each feature.

        Returns
        -------
        self
            Fitted encoder.
        '''
        self.encoder.fit(X[self.categories_])
        combine = zip(self.categories_, self.encoder.categories_)
        self.groups_  = [(c, n[1], "{} ({})".format(*n)) for c,v in combine 
                         for n in list(product([c],v))]
        self.columns_ = np.array(self.groups_)[:,-1]
        self.n_features_new = len(self.columns_ )
        return self
        
    def transform(self, X):
        
        '''
        Transform X using one-hot encoding.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            The data to encode.

        Returns
        -------
        X_out : (n_samples, n_features_new)
            Transformed input.
        '''
        X_out = self.encoder.transform(X[self.categories_])
        X_out = pd.DataFrame(X_out.toarray(), columns=self.columns_)
        return X_out.astype(int)
    
    def fit_transform(self, X):
        
        '''
        Fit to data, then transform it.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        X_new : pd.DataFrame of shape (n_samples, n_features_new)
            Transformed dataframe.
        '''
        self.fit(X[self.categories_])
        X_new = self.transform(X[self.categories_])
        return X_new
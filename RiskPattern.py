'''
Available methods are the followings:
[1] Monitoring <class>
[2] ModifiedEncoder <class>
[3] MonitoringAction <class>
[4] create_folder
[5] get_cohort
[6] ConfigParser

Authors: Danusorn Sitdhirasdr <danusorn.si@gmail.com>
versionadded:: 29-02-2024

'''
import pandas as pd, numpy as np, re, os, glob
from collections import namedtuple, OrderedDict
from AssoruleMining import *
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from itertools import product
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import matplotlib.transforms as transforms
import warnings
from datetime import datetime, timedelta, date

__all__ = ["Monitoring", 
           "ModifiedEncoder", 
           "MonitoringAction", 
           "create_folder", 
           "get_cohort", 
           "ConfigParser"]

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
    # Set2 : ['#4b7bec', '#f7b731', '#eb3b5a', '#E9EAEC']
    # Set3 : ["#677880", "#292400", "#EF3340", "#EBECE0"]
        
    def __init__(self):
        
        '''Initial parameters'''
        self.n_xticks = 9
        self.n_yticks = 6
        self.colors = ['#4b7bec', '#f7b731', '#eb3b5a', '#E9EAEC']
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
    
    # Short, medium, and long term
    periods = [3, 6, 12]

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
            ratio = t.values[-1] / t.okr
            values.update({f"{field}_latest": t.values[-1], 
                           f"{field}_okr": t.okr,
                           f"{field}_ratio" : t.values[-1] / t.okr})
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
    `TreeRuleMining`.
    
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
            `ModifiedEncoder` (onehot-encoded).
        
        Returns
        -------
        self
        
        '''
        if not isinstance(estimator, TreeRuleMining):
            raise ValueError(f"`estimator` must be TreeRuleMining. "
                             f"Got {type(estimator)} instead.")
        elif getattr(estimator, "rules", None) is None:
            raise ValueError("`estimator` must be fitted.")
        else: pass
                
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
        
        # Set n_xticks
        n = len(limit)
        if (n/2-n//2)==0: self.n_xticks = n/2+1
        else: self.n_xticks = int(np.ceil(n/2))
        
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
        okr_quarters = np.unique(okrs["quarter"]).tolist()
        if self.quarter not in okr_quarters: self.quarter = okr_quarters[0]
        
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
                         if len(re.findall("^Rule_[1-9]+",c))]
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
            if isinstance(queries, dict): 
                query = queries[which] if which in queries.keys() else None
                   
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
                       if len(re.findall("^Rule_[1-9]+",c))]
        if len(rules)==0: warnings.warn("No rule found.")

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

class MonitoringAction(ValidateParams):
    
    '''
    Methods
    -------
    1. `define_action` : assign action to pattern from `estimator`.
    2. `add_pattern` : add new pattern to monitoring list.
    3. `get_query` : extract query from monitoring list.
    4. `add_status` : add new status from `estimator` to patterns
    
    Parameters
    ----------
    mapping : dict
        Map new value under `okr_type` column in `cascaded_okr.csv`
        e.g. {"old_value" : "new_value"}. 
    
    factor : float, default=0.8
        The factor with range of (0,1). For more information, see 
        `define_actions`.
    
    '''
    
    def __init__(self, mapping, factor=0.8):
        
        attrs = {"ind2"   : (ind2 := mapping["ind2"]),
                 "ind3"   : (ind3 := mapping["ind3"]),
                 "ratio2" : f"{ind2}_ratio",
                 "ratio3" : f"{ind3}_ratio",
                 "ind3m3" : f"{ind3}_Last03M", 
                 "factor" : self.Interval("factor", factor, 
                                          float, 0, 1, "neither"), 
                 "user"   : os.environ.get('USERNAME'), 
                 "action" : ["action", "monitor", "no-action"],
                 "status" : ["(1) monitor", "(2) cancelled", "(3) closed"],
                 "format" : '%Y-%m-%d %H:%M:%S'}
        
        for key,value in attrs.items():
            setattr(self, key, value)
            
    def __estimator__(self, estimator):
        
        cls_name = estimator.__class__.__name__
        if cls_name != "Monitoring":
            raise ValueError(f"`estimator` must be Monitoring. "
                             f"Got {cls_name} instead.")
            
        if getattr(estimator, "results_", None) is None:
            raise ValueError("`estimator` must be fitted.")
            
    def __dataframe__(self, X, name):
        
        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"`{name}` must be pandas DataFrame. "
                             f"Got {type(X)} instead.")

    def define_action(self, estimator):
        
        '''
        Assign action to all patterns found in `estimator`. The criteria 
        are as follows:
        
            - "action"    : (R(2) > 1.0) and (R(3) > 1.0) 
            - "monitor"   : ((R(2) < 1.0) and (R(3) > 1.0)) or
                            ((factor <= R(3) <= 1.0) and (C(3,3) > 0))
            - "no-action" : (0.0 <= R(3) < factor) or none of above

            where 
            - R(k) = I(k) / OKR(k)
            - I(k) = Indicator k from the latest mature month 
            - C(k,m) = coefficient (trend) of indicator k in last m months
            - k = {1, 2, 3}, nth indicator

        Parameters
        ----------
        estimator : RiskPattern.Monitoring instance
            An instance must be fitted.
        
        Returns
        -------
        results : pd.DataFrame
            An attribute `Monitoring.result_` is converted into a pandas 
            DataFrame with additional columns as follows: 
                - "action"    : {"action", "monitor", "no_action"}
                - "update0"   : date-time ('%Y-%m-%d %H:%M:%S')
                - "username0" : user name
        
        Attributes
        ----------
        book_cohort_ : str
            The booking cohort i.e. max(self.result_["asof"]).
            
        prev_cohort_ : str
            The cohort one month before `book_cohort_`.
            
        '''
        self.__estimator__(estimator)
        results = pd.DataFrame(estimator.results_)
            
        # Action ==> "action"
        cond = (results[self.ratio2]>1) & (results[self.ratio3]>1)
        results["action"] = np.where(cond, self.action[0], self.action[-1])
        
        # Action ==> "monitor"
        args = (self.factor, 1, "both")
        cond = (results[self.ratio3]>1) & (results[self.ratio2]<1 )
        cond|= (results[self.ind3m3]>0) & (results[self.ratio3].between(*args))
        results["action"] = np.where(cond, self.action[1], results["action"]) 
        
        # Action ==> "no-action"
        cond = results["name"].isin(["Rule_all", "Rule_0"])
        results["action"] = np.where(cond, self.action[-1], results["action"]) 
        results["update"] = datetime.today().strftime(self.format)
        results["username"] = self.user
        
        # Booking and previous cohorts
        self.book_cohort_ = max(results["asof"])
        self.prev_cohort_ = self.__previous__(self.book_cohort_,-1)
        self.cohorts_ = [self.book_cohort_, self.prev_cohort_]
        
        return results
        
    def __previous__(self, cohort, lag=0):
        
        '''Get cohort given lag'''
        cohort = np.datetime64(cohort,'M')
        cohort = cohort + np.timedelta64(int(lag),'M')
        cohort = date.fromisoformat(f'{cohort}-01') 
        return cohort.strftime("%Y-%m")
    
    def add_pattern(self, results, patterns):
        
        '''
        Add new "action" pattern to the monitoring list. The pattern is 
        cancelled when it falls under conditions as follows:
            - patterns["asof"].shift(1) in [<current>, <previous>], and
            - patterns["query"].shift(1) == patterns["query"], and
            - patterns["status"].shift(1) in [<monitor>, <cancelled>]
        
        Parameters
        ----------
        results : pd.DataFrame
            New patterns from `define_action`.
        
        patterns : pandas DataFrame
            A current monitoring list of patterns.
            
        Returns
        -------
        new_patterns : pd.DataFrame
            Updated monitoring list with new patterns added (if any).
        
        '''
        self.__dataframe__(results , "results" )
        self.__dataframe__(patterns, "patterns")
        new_patterns = patterns.copy()
        cond = results["action"]==self.action[0]
        
        if sum(cond)>0:
            
            # New patterns (action-required)
            new_patterns = results.loc[cond].copy()
            new_patterns["status"] = self.status[0]
            new_patterns["period"] = 0

            # Concatenate with exising files
            objs = (patterns, new_patterns)
            new_patterns = pd.concat(objs, ignore_index=True)
            new_patterns.sort_values(["query","asof"], inplace=True)

            # Cancel pattern
            cond0 = new_patterns["asof"].shift(1).isin(self.cohorts_)
            cond1 = new_patterns["query"].shift(1)==new_patterns["query"]
            cond2 = new_patterns["status"].shift(1).isin(self.status[:2])
            new_patterns.loc[cond0 & cond1 & cond2, "status"] = self.status[1]
        
        return new_patterns
    
    def get_query(self, patterns, cohort=None):
        
        '''
        Extracts query, whose status is not "(2) cancelled", and has
        existed before defined `cohort`.
        
        Parameters
        ----------
        patterns : pandas DataFrame
            A current monitoring list of patterns.
            
        cohort : str, default=None
            A date in year-month format ("YYYY-MM") e.g. "2024-01". If 
            None, it defaults to `prev_cohort_`.
            
        Returns
        -------
        queries : dict
            A dict with "<name>_<pattern>" as keys and queries as values.
        
        '''
        self.__dataframe__(patterns, "patterns")
            
        # Conditions
        if cohort is None: cohort = self.prev_cohort_
        cond  = patterns["status"]!=self.status[1] 
        cond &= patterns["asof"]<=cohort
        
        # Determine query with "(1) monitor" status
        queries = patterns.loc[cond].groupby(["pattern"])\
        .agg({"status":"max", "query":"max"}).reset_index()
        queries = queries.loc[queries["status"]==self.status[0]]
        queries = queries[["pattern", "query"]].values
        queries = dict([(f"Rule_9999_{p}", q) for p,q in queries])
        if len(queries)==0: warnings.warn('No query found.')
        
        return queries
    
    def add_status(self, estimator, patterns):
        
        '''
        Add new status from `estimator` to patterns, whose status is 
        "(1) monitor". The status is closed or "(3) closed" when pattern's
        indicators are below thresholds, otherwise remains unchanged.
        
        Parameters
        ----------
        estimator : RiskPattern.Monitoring instance
            An instance must be fitted.
        
        patterns : pandas DataFrame
            A current monitoring list of patterns.
            
        Returns
        -------
        new_status : pd.DataFrame
            Updated monitoring list with new status added (if any).
            
        '''
        self.__estimator__(estimator)
        new_status = pd.DataFrame(estimator.results_)
            
        # Default values
        cond = new_status["name"].isin(["Rule_all", "Rule_0"])
        new_status = new_status.loc[cond==False].reset_index(drop=True)
        new_status["action"] = self.action[0]
        new_status["update"] = datetime.today().strftime(self.format)
        new_status["username"] = self.user
        
        # Extract "pattern"
        f0 = lambda x: re.findall("P[0-9]+", x)[-1]
        new_status["pattern"] = new_status["pattern"].apply(lambda x:f0(x))
        new_status["name"] = "MONITOR"
                   
        # Get latest status and period
        aggfnc = {"status":"max", "period":"max"}
        status = patterns.groupby("pattern").agg(aggfnc).reset_index()
        status["period"] = status["period"] + 1
        status["status"] = self.status[0]

        # Status ==> "(3) closed"
        new_status = new_status.merge(status, how="left", on="pattern")
        cond = new_status[self.ratio3]<=1
        new_status.loc[cond, "status"] = self.status[-1]
        new_status = pd.concat((patterns, new_status), ignore_index=True)

        return new_status

def get_cohort(cohort, month=0):
    
    '''
    Calculate cohort given lag

    Parameters
    ----------
    cohort : str
        A date with "YYYY-MM" format

    month : int, default=0
        Number of months to be added to `cohort`.
    '''
    cohort = np.datetime64(cohort,'M')
    cohort = cohort + np.timedelta64(int(month),'M')
    cohort = date.fromisoformat(f'{cohort}-01') 
    return cohort.strftime("%Y-%m")

def create_folder(*paths, remove=True):

        '''
        Create new folder.

        Parameters
        ----------
        *paths : tuple of str
            All members of paths.

        remove : bool, default=True
            If True, it will remove all files and subfolders under this 
            path. This is only relevant when path already exists.

        Returns
        -------
        path : str

        '''
        path = os.path.join(*paths)
        if os.path.exists(path)==False: 
            os.makedirs(path)
            msg = 'New directory <{}> is successfully created.'
            print(msg.format(path))
        elif remove:
            for file in os.listdir(path): 
                try: os.remove(os.path.join(path, file))
                except: os.rmdir(os.path.join(path, file)) 
            msg = '<{}> already exists. All files and folders have been deleted.'
            warnings.warn(msg.format(path))
        else:  warnings.warn(f"<{path}> already exists.")
        return path

class ConfigParser(ValidateParams):
    
    '''
    Configurations
    
    Parameters
    ----------
    mature_lag : int, default=0
        Time Lag in months between booking and mature month.
    
    product : str, default=None
        Type of product e.g. "CC", "KCL", "XPC". 

    observed_mths : int, default=None
        Number of months to be observed before mature cohort.
    
    mapping : dict, default=None
        Map new value under `okr_type` column in `cascaded_okr.csv`
        e.g. {"old_value" : "new_value"}. 
    
    laggings : list of (str, int), default=None
        Time Lag in months for all indicators e.g. (indicator, lag). 
        This is used to replace any premature indicators with NaN. 

    prim_keys : list of str, default=None
        Primary keys for any newly created data e.g. ["apl_grp_no", 
        "ip_id", "product_tp", "cohort_fr"].
    
    categories : list of str, default=None
        Categorical features that will be used to compute pattern(s)
    
    frac : float, default=0.05
        The minimum percent of samples required to be at a leaf node.
    
    n_samples : int, default=500
        The minimum number of samples required to be at a leaf node.
    
    main_folder : str, default=None 
        Path of main folder where all sub-folders and files are created. 
        if path contains "<USER>" e.g. "D:\\Users\\<USER>\\**\\TEST", 
        it will be replaced with log-in user name.
    
    data_path : str, default=None
        Path of data. if path contains "<USER>", it will be replaced 
        with log-in user name.
    
    '''
    
    def __init__(self, mature_lag=0, product=None, observed_mths=18, 
                 mapping=None, laggings=None, prim_keys=None, 
                 categories=None, frac=0.05, n_samples=500, 
                 main_folder=None, data_path=None):
        
        self.create_date = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        args = ("mature_lag", mature_lag, int, None, None, "neither")
        self.mature_lag = self.Interval(*args)
        self.product = str(product)
        self.mapping = mapping
        self.laggings = laggings
        self.prim_keys = prim_keys
        self.categories = categories
        args = ("frac", frac, float, 0, 1, "right")
        self.frac = self.Interval(*args)
        args = ("n_samples", n_samples, int, 100, None, "left")
        self.n_samples = self.Interval(*args)
        self.main_folder = main_folder
        self.data_path = data_path
        self.data_folder = os.path.split(data_path)[0]
        args = ("observed_mths", observed_mths, int, 10, None, "left")
        self.observed_mths = self.Interval(*args)
        self.observed_mths = np.fmax(self.observed_mths, 
                                     self.laggings[2][1])
        self.msgs = ["This directory does not exist <{}>.".format, 
                     "There are {:,d} possible paths. "\
                     "Please be more specific.".format]
        
    def update(self, cohort=None):
        
        '''
        Update configuarions
        
        Parameters
        ----------
        cohort : str, default=None
            A date with "YYYY-MM" format. If None, cohort-related 
            parameters will not be created or updated.
            
        '''
        
        user = os.environ.get('USERNAME')
        for attr in ["main_folder", "data_folder", "data_path"]:
            path = getattr(self, attr).replace("<USER>",user)
            found = glob.glob(path, recursive=True)
            if len(found)==0: raise ValueError(self.msgs[0](path))
            elif len(found)>1: raise ValueError(self.msgs[1](len(found)))
            else: setattr(self, attr, found[0])
          
        # Update other parameters
        if cohort is not None:
            self.okr_path, self.quarter = self.__okr__(cohort)
            self.mature_indic1 = self.__cohort__(cohort, self.mature_lag)
            self.mature_indic2 = self.__cohort__(self.mature_indic1, -self.laggings[1][1])
            self.mature_indic3 = self.__cohort__(self.mature_indic1, -self.laggings[2][1])
            self.start_cohort  = self.__cohort__(self.mature_indic1, -self.observed_mths)

            # Excel and text file
            a  = [self.product] + cohort.split("-") 
            a += self.mature_indic1.split("-")
            self.xls_file = "RESULT_{}{}{}M{}{}.xlsx".format(*a)
            self.txt_file = "ACTION_{}{}{}M{}{}.txt".format(*a)

    def __okr__(self, cohort):
    
        '''OKR file path and quarter'''
        # Find okr_20**.csv from designated directory
        cohort = int(cohort.replace("-",""))
        asof = []
        for f in os.listdir(self.data_folder):
            if len(re.findall("^okr_[0-9]{4}",f)):
                asof += [int(re.findall("[0-9]{4}",f)[0])]

        # Convert to YYYYMM by quarter
        asof = product(np.array(asof)*100, np.arange(3,13,3))
        asof = np.array([sum(c) for c in asof])
        asof = str(asof[cohort<=asof][0])

        # Create file path
        f = "okr_{}.csv".format(int(asof[:4]))
        q = "q{:.0f}".format(np.ceil(int(asof[-2:])/3))
        return os.path.join(self.data_folder, f), q
        
    def __cohort__(self, cohort, month=0):
    
        '''Calculate cohort given month'''
        cohort = np.datetime64(cohort,'M')
        cohort = cohort + np.timedelta64(int(month),'M')
        cohort = date.fromisoformat(f'{cohort}-01') 
        return cohort.strftime("%Y-%m")
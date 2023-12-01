#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd, numpy as np, os
from AssoruleMining import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from itertools import product
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker


# In[2]:


mpl.rcParams['lines.linewidth'] = 4
mpl.rcParams['lines.solid_capstyle'] = 'round'
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['figure.figsize'] = [6.5, 4.5]
mpl.rcParams['axes.grid'] = False
pd.options.display.max_rows = 1000
pd.options.display.max_columns = 1000


# In[3]:


def find_slope(y, n=None, x=None, intercept=True):
    ''''''
    n = len(y) if n is None else int(n)
    y = np.array(y)[-n:].reshape(-1,1)
    if x is None: x = np.arange(n).reshape(-1,1)
    if intercept: x = np.hstack((x, np.ones((n,1))))
    a = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y).ravel()
    if intercept: return a[0], a[1]
    else: return a[0], 0


# In[4]:


folder ="D:\\Users\\danusorn.s\\OneDrive - KASIKORNBANKGROUP\\dpd_data_sharing"
os.listdir(folder)


# In[5]:


X = pd.read_csv(folder + '\\cc_dpd_master.txt.gz', sep="|", low_memory=False)
modl_seg = {"NTB":"1]NTB", "NTC":"2]NTC", "ETC":"3]ETC"}
X["modl_seg"] = X["modl_seg"].apply(lambda x: modl_seg[x])


# In[6]:


usecols = ["product_tp", "credit_seg", "fico_grp", "SA_SE", "okr_type", "quarter", "modified_okr"]
OKR = pd.read_csv(folder + '\\cascaded_okr.csv', low_memory=False)
OKR = OKR.loc[OKR["used_in_3mth"]=='ALL', usecols].rename(columns={"credit_seg":"modl_seg"})
OKR.reset_index(drop=True, inplace=True)


# Convert categorical to numerical variable.

# In[7]:


prim_keys  = ["main_cc_cst_no", "ip_id", "product_tp", "cohort_fr"]
categories = ['age_grp', 'ocp_grp', 'SA_SE', 'incm_grp', 'incmv_grp', 'src_incm_grp', 
              'fnl_cr_lmt_grp', 'apl_chnl', 'fico_grp', 'fico_range', 'credit_seg', 
              'mth_tnr_range']


# In[8]:


enc = OneHotEncoder(handle_unknown='ignore').fit(X[categories])
cat_columns = ["{} ({})".format(*n) 
               for c,v in zip(categories, enc.categories_) 
               for n in list(product([c],v))]
cat_X = pd.DataFrame(enc.transform(X[categories]).toarray(), 
                     columns=cat_columns).astype(int)
cat_X = X[prim_keys].merge(cat_X, left_index=True, right_index=True)


# In[9]:


cat_X.head(3)


# - 'flg_06pM01'
# - 'flg_60pM03'
# - 'flg_90pM04'

# In[10]:


latest   = cat_X["cohort_fr"]==np.sort(cat_X["cohort_fr"].unique())[-1]
latest_X = cat_X.loc[latest].reset_index(drop=True).copy()
latest_y = X.loc[latest, 'flg_06pM01'].values
print("Selected cohort : {}".format(*latest_X["cohort_fr"].unique()))
print("Number of flags : {:,.0f}".format(sum(latest_y)))


# Applying Tree-Rule-Mining

# In[11]:


max_depth = int(np.log2(latest_X.shape[0]))
estimator = DecisionTreeClassifier(max_depth=max_depth,
                                   max_features=len(cat_columns), 
                                   random_state=0, 
                                   min_samples_leaf=50, 
                                   class_weight="balanced")


# In[12]:


TRM = TreeRuleMining(estimator, exclude=True, metric="precision", max_iter=20)
TRM.fit(latest_X[cat_columns], latest_y)


# In[13]:


TRM.evaluate(latest_X[cat_columns], latest_y, cumulative=True).fillna(0)


# In[14]:


for n in np.arange(TRM.n_rules)+1:
    print(f"Rule: {n}")
    print_rule(TRM.rules[f"Rule_{n}"])


# In[15]:


new_X = X.merge(TRM.transform(cat_X[cat_columns], TRM.n_rules), 
                left_index=True, right_index=True)


# In[16]:


new_X.head()


# In[17]:


class ObjKeyResults:
    
    '''Objectives and Key Results'''
    
    def __init__(self, OKRs, prod="ALL", qtr="q4", 
                 mapping={"flg_06pM01" : "a_lmt_fpd",
                          "flg_60pM03" : "a_lmt_61pM03",
                          "flg_90pM04" : "a_lmt_91pM04"}):
  
        cond = (OKR["product_tp"]==prod) & (OKR["quarter"]==qtr) 
        self.OKRs = OKRs.loc[cond].reset_index(drop=True).copy()
        self.cols = ["modl_seg", "fico_grp", "SA_SE"]
        self.mapping = mapping
        self.n_ticks = 6
        
    def plot(self, X, col="flg_06pM01"):
        
        values = X[["modl_seg", "fico_grp", "SA_SE"]].copy()
        values = values.drop_duplicates().to_dict().items()
        values = dict([(k,np.unique(list(v.values()))) for k,v in values])
        okr = self.__okr__(values, self.mapping[col])
        
         
        data = X.groupby("cohort_fr").agg({col:"mean"}).reset_index()
        ax = plt.subplots()[1]
        ax.plot(data["cohort_fr"], data[col], label=col)
        ax.axhline(okr, lw=1, ls="--", label="OKR = {:.2%}".format(okr))
        

        mth = 12
        slope, intercept = find_slope(data[col][-mth:])
        x = np.arange(len(data))[-mth:]
        y = np.arange(mth)
        ax.plot(x, y*slope + intercept, ls="-", lw=1, 
                label="{:,d}-month trend ({:.2%})".format(mth,slope))
        
        mth = 6
        slope, intercept = find_slope(data[col][-mth:])
        x = np.arange(len(data))[-mth:]
        y = np.arange(mth)
        ax.plot(x, y*slope + intercept, ls="-", lw=1, 
                label="{:,d}-month trend ({:.2%})".format(mth,slope))
        
        mth = 3
        slope, intercept = find_slope(data[col][-mth:])
        x = np.arange(len(data))[-mth:]
        y = np.arange(mth)
        ax.plot(x, y*slope + intercept, ls="-", lw=1, 
                label="{:,d}-month trend ({:.2%})".format(mth,slope))
        
        
        ax.legend(loc="best", framealpha=0)
        self.__tickers__(ax)
        
        return ax
    
    def __okr__(self, values, indicator):
        
        '''Get OKR given indicator'''
        cond = self.OKRs["okr_type"]==indicator
        for c in self.cols:
            if len(values[c])>1: 
                cond &= self.OKRs[c].isin(["ALL"])
            else: cond &= self.OKRs[c].isin(values[c])
        return self.OKRs.loc[cond,"modified_okr"].values[0]
    
    def __tickers__(self, ax):
        
        '''Set tickers'''
        t = ticker.PercentFormatter(xmax=1)
        ax.yaxis.set_major_formatter(t)
        n_ticks = mpl.ticker.MaxNLocator(self.n_ticks)
#         ax.yaxis.set_major_locator(n_ticks)
        ax.xaxis.set_major_locator(n_ticks)
        ax.tick_params(axis='both', labelsize=10)


# In[ ]:





# In[ ]:





# In[18]:


for n in range(1,TRM.n_rules+1):
    ax = ObjKeyResults(OKR, prod="CC").plot(new_X.loc[new_X[f"Rule_{n}"]], col="flg_06pM01")


# In[ ]:


pd.to_datetime(new_X["cohort_fr"], format="%Y-%m") # + np.timedelta64(1,'M')


# In[ ]:





# In[21]:


x = new_X.loc[new_X["Rule_1"]].groupby("cohort_fr").agg({"flg_06pM01":"mean"})
y = new_X.loc[new_X["Rule_1"]].groupby("cohort_fr").agg({"flg_90pM04":"mean"})[:]


# In[28]:


y.values.ravel()/x.values.ravel()


# In[23]:





# In[ ]:





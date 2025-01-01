'''
Available methods are the followings:
[1] VariableClustering
[2] random_variables

Author: Danusorn Sitdhirasdr <danusorn.si@gmail.com>
versionadded:: 10-09-2021

This algorithm is developed from `VarClusHi`.
(https://pypi.org/project/varclushi/)

'''
import pandas as pd, numpy as np
from warnings import warn
import collections
from factor_analyzer import Rotator
from itertools import permutations
from math import factorial
import math

__all__ = ['VariableClustering', 'random_variables', 
           '_pca_', '_variance_', 'reassign_var']

class FindNetwork:
    
    def __init__(self, min_cnt=1):
        
        self.min_cnt = min_cnt
    
    
    
    def fit(self, items):
        
        '''
        =================================================================
        
        Parameters
        ----------
        items : list of sets, of shape (n_sets,)
            A list of sets e.g. [{1, 2}, {5, 8, 9}, ...].
         
        Returns
        -------
        self
        
        '''
        # Initialize parameters
        items = [set(item) for item in items if len(item)>1]

        
        self.sets_ = dict()
        a = 0
        
        
        while True:
            a += 1
            item = items.pop(rand.randint(0, len(items)))
            k, items, p = self.__extendset__(item, items)
            
            if len(k)>2: self.sets_[len(self.sets_)] = k
           
            if len(items)==1: break
            if a>100:break
        
        return self
    
    
    
    def __checkitems__(self, items):
        
        '''
        '''
        # Convert all items into sets, ignoring set that contains 1 member.
        items = [set(item) for item in items if len(item)>1]
        
        items = np.r_[[list(n) for n in items]].ravel()
        
unq,cnt = np.unique(items, return_counts=True)
unq_items = unq[np.argsort(cnt)][::-1]
cnt_items = cnt[np.argsort(cnt)][::-1]
    
    
    
    def __extendset__(self, item, test_items):
    
        '''
        Extend a set by identifying and combining sets that share one or 
        more common elements (joint sets). The algorithm stops when there
        is no joint set.

        Parameters
        ----------
        item : set
            An input set e.g. {1, 3, 5}.

        test_items : list of sets, of shape (n_sets,)
            A list of sets e.g. [{1, 2}, {5, 8, 9}, ...].
        
        Returns
        -------
        item : set
            If `n_jointsets` equals 0, an input set is returned; otherwise, 
            a joint set is returned .
            
        test_items : list of sets
            If `n_jointsets` equals 0, an initial list is returned; 
            otherwise, a list without a joint set is returned. 
        
        n_jointsets : int
            Number of joint sets found in `test_items`.

        '''
        old = item.copy()
        rem = test_items.copy()
        while True:
            new, rem, isjoint = self.__joinset__(old, rem)
            if isjoint: old = new.copy()
            else: return new, rem, len(new)-len(item)

    def __joinset__(self, item, test_items):
    
        '''
        Find a joint set, whose one or more elements are common among 
        them. An example of a joint set, showing the common element, is 
        J={1,2,3,4} and K={5,2,6,7}. The number two (2) is the common 
        element among the two sets and therefore considers these sets 
        joint.

        Parameters
        ----------
        item : set
            An input set e.g. {1, 3, 5}.

        test_items : list of sets, of shape (n_sets,)
            A list of sets e.g. [{1, 2}, {5, 8, 9}, ...].
        
        Returns
        -------
        item : set
            If `isjoint` is True, a joint set is returned; otherwise, an 
            input set is returned.
            
        test_items : list of sets
            If `isjoint` is True, a list without a joint set is returned; 
            otherwise, an initial list is returned.
        
        isjoint : boolean
            If True, a joint set is found.
            
        '''
        test = np.array(test_items).copy()
        for n,t in enumerate(test):
            if ((item.isdisjoint(t)==False) & 
                (t.issubset(item)==False)):
                test = test[np.arange(len(test))!=n]
                return item.union(t), test.tolist(), True
        return item, test_items, False

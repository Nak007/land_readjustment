'''
Available methods are the followings:
[1] UnionJointSet

Author: Danusorn Sitdhirasdr <danusorn.si@gmail.com>
versionadded:: 31-01-2025

''' 
import pandas as pd, numpy as np, os, re, time
from warnings import warn
import collections
try: import progressbar
except: 
    !pip install progressbar

__all__ = ['UnionJointSet']

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
        err_msg = "`%s` must be %s or in %s, got %s " % tuples    

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

class UnionJointSet(ValidateParams):

    '''
    Sets are merged when a "Joint" set is identified, where one or 
    more items are shared among the sets. The algorithm proceeds as 
    follows:
    
        1. Generate a list of unique items (`selected_items`), 
           focusing on their frequency of occurrence. Items with a 
           frequency below the specified threshold (`min_cnt`) will 
           be excluded.
        2. Iterate through the list one item at a time, merging as 
           many joint sets as possible to form a new set. A new set
           must contain more than or equal to defined threshold 
           (`min_items`) to qualify.
        3. Any items found in the newly created set will be removed 
           from `selected_items` to minimize time complexity.
        4. This process continues until there is no item left in
           `selected_items`.
        
    Parameters
    ----------
    min_cnt : int, default=1
        Minimum frequency of occurrence.

    min_items : int, default=3
        Minimum number of items in merged set.

    Attributes
    ----------
    items_ : dict
        In each iteration, the newly formed set that contains more items 
        than defined thresholds will be stored in `self.items_`. For 
        example, it may look like this: {0: {1,2,3}, 1: {4,5}, ...}.

    n_jointsets_ : int
        The number of joint sets found.
        
    References
    ----------
    [1] https://github.com/niltonvolpato/python-progressbar
        
    '''
    def __init__(self, min_cnt=1, min_items=3):

        # Validate parameters
        self.Interval("min_cnt", min_cnt, int, 0, None, "right")
        self.Interval("min_items", min_items, int, 2, None, "right")
        self.min_cnt = min_cnt
        self.min_items = min_items
    
    def fit(self, items):
        
        '''
        Fit the model.
        
        Parameters
        ----------
        items : list of sets, of shape (n_sets,)
            A list of sets e.g. [{1, 2}, {5, 8, 9}, ...].
         
        Returns
        -------
        self
        '''
        # Initialize parameters
        valid_items, selected_items = self.__checkitems__(items)
        n_items = len(selected_items)
        self.items_ =dict()

        # Initialize progressbar
        widgets = [progressbar.Timer(), ',', progressbar.Percentage(), ' ', 
                   progressbar.Bar("="), ' [', progressbar.ETA(), '] ',]
        kwargs = dict(widgets=widgets, maxval=n_items)
        bar = progressbar.ProgressBar(**kwargs).start()
        time.sleep(0.1)
        
        # Loop through pre-selected items
        while True:

            # Determine set from valid items
            args = ({selected_items[0]}, valid_items)
            new_item, valid_items, _ = self.__extendset__(*args)

            # Remove new items form selected items
            selected_items = list(set(selected_items).difference(new_item))

            # Select set that contains more than define threshold
            if len(new_item) > self.min_items: 
                self.items_[len(self.items_)] = new_item
            
            if len(selected_items)==0: break
            bar.update(n_items-len(selected_items))
            
        bar.finish()
        self.n_jointsets_ = len(self.items_.keys())
        return self

    def __checkitems__(self, items):
        
        '''
        Validate items
        
        Parameters
        ----------
        items : list of sets, of shape (n_sets,)
            A list of sets e.g. [{1, 2}, {5, 8, 9}, ...].
            
        Returns
        -------
        valid_items : list of sets
            A list of sets that contain more than one member.

        selected_items : list
            A list of items that pass the minimum count (`self.min_cnt`).
            
        '''
        # Convert all items into sets, ignoring set that contains 1 member.
        valid_items = [set(item) for item in items if len(set(item))>1]
        valid_items = np.unique(valid_items).tolist()
        
        # Choose the unique items that occur more frequently than the 
        # specified threshold.
        item_list = list()
        for n in valid_items: item_list.extend(list(n))
        unqs,cnts = np.unique(item_list, return_counts=True)
        unq_items = unqs[np.argsort(cnts)][::-1]
        cnt_items = cnts[np.argsort(cnts)][::-1]
        selected_items = unq_items[cnt_items>self.min_cnt]
        return valid_items, selected_items
    
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
        # Initialize parameters
        old_item = item.copy()
        filtered_items = test_items.copy()
        while True:
            args = (old_item, filtered_items)
            new_item, filtered_items, joint = self.__joinset__(*args)
            if joint==False: break
            old_item = new_item.copy()
        return new_item, filtered_items, len(test_items)-len(filtered_items)

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
            # Union two sets when they have one or more common items
            if (item.isdisjoint(t)==False):
                test = test[np.arange(len(test))!=n]
                return item.union(t), test.tolist(), True
        return item, test_items, False

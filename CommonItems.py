'''
Available methods are the followings:
[1] UnionJointSet

Author: Danusorn Sitdhirasdr <danusorn.si@gmail.com>
versionadded:: 31-01-2025

''' 
import pandas as pd, numpy as np, os, re, time
from warnings import warn
from itertools import product, combinations, chain
from collections import namedtuple
from pyvis.network import Network
try: import progressbar
except: 
    get_ipython().system('pip install progressbar')

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

class ColorGradient():

    def __hex2RGB__(self, hex_str):

        '''
        Convert hex to RGB e.g. #FFFFFF to [255,255,255]
        '''
        # Pass 16 to the integer function for change of base
        return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]

    def __gradient__(self, hex1, hex2, n=2):

        '''
        Given two hex colors, returns a color gradient with n colors.

        Parameters
        ----------
        hex1, hex2 : hex color
            Hex colors
        
        n : int, default=2
            Number of colors between 2 hex colors.

        Returns
        -------
        hex_colors : list of hex colors
        
        '''
        self.Interval("n", n, int, 1, None, "right")
        rgb1 = np.array(self.__hex2RGB__(hex1))/255
        rgb2 = np.array(self.__hex2RGB__(hex2))/255
        rgb_colors = [((1-mix)*rgb1 + (mix*rgb2)) 
                      for mix in [x/(n-1) for x in range(n)]]
        return ["#" + "".join([format(int(round(val*255)), "02x") 
                               for val in item]) for item in rgb_colors]

class UnionJointSet(ValidateParams, ColorGradient):

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
    
    When the degree of centrality is calculated, it assumes 
    interconnection among all nodes within each set. However, the 
    actual connections can be better represented by dividing the items 
    into smaller sets, ideally pairs of two items.
        
    Parameters
    ----------
    min_cnt : int, default=1
        Minimum frequency of occurrence.

    min_items : int, default=3
        Minimum number of items in merged set.

    Attributes
    ----------
    items_ : dict of collections.namedtuple 
        In each iteration, any newly formed set containing more items 
        than the defined thresholds will be stored in `self.items_`, 
        using indices as keys and a namedtuple ("Network") as values, 
        with the following fields:
    
        Field       Description
        -----       -----------
        n_nodes     Number of nodes 
        members     List of members
        n_edges     Number of edges for each member (links) 
        centrality  Degree of Centrality for each member, with values 
                    ranging from 0 to 1.
        edge_index  Indices of edges that connect to the members. This 
                    can be utilized with `self.edges_`.

    n_jointsets_ : int
        The number of joint sets found.

    results_ : dict of numpy ndarrays
        A dict with keys as column headers and values as columns, that 
        can be imported into a pandas DataFrame. The data is identical
        to that contained in `self.items_`.

    References
    ----------
    [1] https://github.com/niltonvolpato/python-progressbar
        
    '''

    fields = ['network', 'members', 'n_edges', 'centrality']
    pyvis_kwargs = dict(notebook=True, cdn_resources="remote",
                        bgcolor='white', font_color="white",
                        height="500px", width="100%")
    label = "Network: {} \n ID: {} \n Edges: {:,.0f} \n Centrality: {:.3g}".format
    
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
            A list of sets e.g. [{1, 2}, {5, 8, 9}, ...]. When the degree 
            of centrality is calculated, it assumes interconnection among 
            all nodes within each set. However, the actual connections 
            can be better represented by dividing the items into smaller 
            sets, ideally pairs of two items.
         
        Returns
        -------
        self
        
        '''
        # Initialize parameters
        valid_items, selected_items = self.__checkitems__(items)
        n_items = len(selected_items)
        self.items_ =dict()

        # All edges among nodes
        self.edges_ = np.unique([set(n) for item in valid_items 
                                 for n in combinations(item,2)])
        self.n_edges = len(self.edges_)

        # All nodes and their centrality degrees. 
        a = [n for nodes in self.edges_ for n in nodes]
        self.nodes_, self.n_edges_ = np.unique(a, return_counts=True)

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
                node_index = np.isin(self.nodes_, list(new_item))
                edge_index = [edge.isdisjoint(new_item)==False 
                              for edge in self.edges_]
                values = {"n_nodes" : (n_nodes:=len(new_item)),
                          "members" : self.nodes_[node_index],
                          "n_edges" : (n_edges:=self.n_edges_[node_index]),
                          "centrality" : n_edges/(n_nodes-1),
                          "edge_index" : np.arange(self.n_edges)[edge_index]}
                values = namedtuple("Network", values.keys())(**values)
                self.items_[len(self.items_)] = values

            if len(selected_items)==0: break
            bar.update(n_items-len(selected_items))
            
        bar.finish()
        self.n_jointsets_ = len(self.items_.keys())
        self.__todict__()
        
        return self

    def __todict__(self):

        '''
        Private Function: Convert `self.items_` into dict.

        Attributes
        ----------
        results_ : dict of numpy ndarrays
            A dict with keys as column headers and values as columns, that 
            can be imported into a pandas DataFrame. The data is identical
            to that contained in `self.items_`.

        '''
        if getattr(self,"items_",None) is None:
            raise ValueError("This instance is not fitted yet. Call 'fit' with"
                             "appropriate arguments before using this estimator.")
        
        self.results_ = dict([(c,[]) for c in self.fields])
        for key in self.items_.keys():
            self.results_["network"].extend([key]*self.items_[key].n_nodes)
            for fld in self.fields:
                self.results_[fld].extend(getattr(self.items_[key],fld,[]))

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

    def networkdata(self, network=0):

        '''
        Transformed data.
        
        Parameters
        ----------
        network : int, default=0
            Index of network from `self.items_`.

        Returns
        -------
        Data : dict of collections.namedtuple 
            A transformed data ready for processing in `pyvis.network`, 
            stored in a namedtuple called 'Data' with the following fields:
         
            Field       Description
            -----       -----------
            nodes       The ids of the node. The id is mandatory for nodes 
                        and they have to be unique.
            edges       List of all edges.
            title       Title to be displayed for each node.
            color       List of colors for each node.

        '''
        if getattr(self,"items_",None) is None:
            raise ValueError("This instance is not fitted yet. Call "
                             "'fit' with appropriate arguments before "
                             "using this estimator.")
        else: self.Interval("network", network, int, 
                            0, self.n_jointsets_-1, "both")
            
        # Initialize parameters
        items = self.items_[network]

        # list of titles
        data = zip([network]*items.n_nodes, items.members, 
                   items.n_edges, items.centrality)
        title = [self.label(*a) for a in data]

        # List of colors
        unique = np.unique(items.centrality)
        args = ('#ecf0f1', '#FC427B', len(unique))
        color_dict = dict(zip(unique, self.__gradient__(*args)))
        color = [color_dict[n] for n in items.centrality]

        # Reassign number to all nodes and edges
        all_edges = self.edges_[items.edge_index]
        allids = np.unique([list(e) for e in all_edges])
        id2int = dict([(id,n) for n,id in enumerate(allids)])
        edges = [[id2int[n] for n in edge] for edge in all_edges]
        nodes = list(id2int.values())

        # Store in "Data"
        values = {"nodes" : nodes, "edges" : edges,
                  "title" : title, "color" : color, 
                  "size" : [15]*len(nodes)}
        return namedtuple("Data", values.keys())(**values)
        
    def visualize(self, network=0):

        '''
        Visualize network graph.
        
        Parameters
        ----------
        network : int, default=0
            Index of network from `self.items_`.

        Returns
        -------
        network : pyvis.network.Network
            Generate a static HTML file named 'NETWORK_000.html' and save 
            it locally before opening it.
            
        '''
        # Plot network graph
        data = self.networkdata(network)
        net = Network(**self.pyvis_kwargs)
        net.add_nodes(data.nodes, title=data.title, 
                      size=data.size, color=data.color)
        net.add_edges(data.edges)
        net.inherit_edge_colors(False)

        # File name
        k = int(np.ceil(np.log10(self.n_jointsets_)))
        name = f"NETWORK_{str(network).zfill(k)}.html"
        
        return net.show(name)

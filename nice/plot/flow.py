import os
import sys
import time
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection

default_prop_cycle = matplotlib.rcParamsDefault['axes.prop_cycle'].by_key()['color'].copy()

def flow_plot(graph, flow_graph, data, terminals, epsilon = 0):

    fig, ax = plt.subplots(figsize = (8, 6))
    
    cmap = Colormap('viridis')
    
    nodes = {
        'plot': {
            's': 50,
            'ec': 'k',
            'fc': 'gray',
            'zorder': 1,
        },
    }
    
    edges = {
        'plot': {
            'lw': 1,
            'color': 'lightgray',
            'zorder': 0, 
        },
    }
    
    kw = {
        'edges': edges,
        'nodes': nodes,
    }
    
    plots = plot_graph(graph, ax = ax, **kw)
    
    nodes = {
        'field': 'plot',
        'cmap': cmap,
        'selection': [k for k, v in flow_graph._node.items() if v['plot'] > epsilon],
        'plot': {
            's': 75,
            'ec': 'k',
            'zorder': 4,
        },
        'colorbar': {
            'label': 'Volume [-]',
        },
    }
    
    edges = {
        'plot': {
            'lw': 2,
            'color': 'k',
            'zorder': 2,
        },
    }
    
    kw = {
        'edges': edges,
        'nodes': nodes,
    }
    
    plots = plot_graph(flow_graph, ax = ax, **kw)
    
    nodes = {
        'selection': terminals,
        'plot': {
            's': 200,
            'ec': 'k',
            'fc': 'red',
            'zorder': 5,
        },
    }
    
    kw = {
        'nodes': nodes,
    }

    plots = plot_graph(graph, ax = ax, **kw)
    
    kw = {
        'facecolor': 'whitesmoke',
    }
    
    ax.set(**kw)
    
    kw = {
        'ls': 'dashed',
    }
    
    ax.grid(**kw)
import json
import momepy
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx

from scipy.spatial import KDTree

def cypher(graph):

	encoder = {k: idx for idx, k in enumerate(graph.nodes)}
	decoder = {idx: k for idx, k in enumerate(graph.nodes)}

	return encoder, decoder

# Functions for NLG JSON handling 

class NpEncoder(json.JSONEncoder):
	'''
	Encoder to allow for numpy types to be converted to default types for
	JSON serialization. For use with json.dump(s)/load(s).
	'''
	def default(self, obj):

		if isinstance(obj, np.integer):

			return int(obj)

		if isinstance(obj, np.floating):

			return float(obj)

		if isinstance(obj, np.ndarray):

			return obj.tolist()

		return super(NpEncoder, self).default(obj)

def nlg_to_json(nlg, filename):
	'''
	Writes nlg to JSON, overwrites previous
	'''

	with open(filename, 'w') as file:

		json.dump(nlg, file, indent = 4, cls = NpEncoder)

def nlg_from_json(filename):
	'''
	Loads graph from nlg JSON
	'''

	with open(filename, 'r') as file:

		nlg = json.load(file)

	return nlg

# Functions for NetworkX graph .json handling

def graph_to_json(graph, filename, **kwargs):
	'''
	Writes graph to JSON, overwrites previous
	'''

	with open(filename, 'w') as file:

		json.dump(nlg_from_graph(graph, **kwargs), file, indent = 4, cls = NpEncoder)

def graph_from_json(filename, **kwargs):
	'''
	Loads graph from nlg JSON
	'''

	with open(filename, 'r') as file:

		nlg = json.load(file)

	return graph_from_nlg(nlg, **kwargs)

# Functions for converting between NLG and NetworkX graphs

def graph_from_nlg(nlg, **kwargs):

	return nx.node_link_graph(nlg, multigraph = False, **kwargs)

def nlg_from_graph(nlg, **kwargs):

	nlg = nx.node_link_data(nlg, **kwargs)

	return nlg

# Functions for graph operations

def subgraph(graph, nodes):

	_node = graph._node
	_adj = graph._adj

	node_list = [(n, _node[n]) for n in nodes]

	edge_list = []

	for source in nodes:
		for target in nodes:

			edge_list.append((source, target, _adj[source].get(target, None)))

	edge_list = [e for e in edge_list if e[2] is not None]

	subgraph = graph.__class__()

	subgraph.add_nodes_from(node_list)

	subgraph.add_edges_from(edge_list)

	subgraph.graph.update(graph.graph)

	return subgraph

def supergraph(graphs):

	supergraph = graphs[0].__class__()

	nodes = []

	edges = []

	names = []

	show = True

	for graph in graphs:

		for source, _adj in graph._adj.items():

			nodes.append((source, graph._node[source]))

			for target, edge in _adj.items():

				edges.append((source, target, edge))

	supergraph.add_nodes_from(nodes)

	supergraph.add_edges_from(edges)

	return supergraph

def remove_self_edges(graph):

	graph.remove_edges_from(nx.selfloop_edges(graph))

	return graph

def path_cost(graph, path, fields = []):

	_adj = graph._adj

	costs = {field: 0 for field in fields}

	for idx in range(1, len(path)):

		source = path[idx - 1]
		target = path[idx]

		for field in fields:

			costs[field] += _adj[source][target][field]

	return costs


import json
import momepy
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx

from sklearn.cluster import DBSCAN, KMeans

def cluster(graph, n_clusters, **kwargs):

    node_functions = kwargs.get('node_functions', {})
    edge_functions = kwargs.get('edge_functions', {})

    x = np.array([n['x'] for n in graph._node.values()])
    y = np.array([n['y'] for n in graph._node.values()])
    k = np.array(list(graph.nodes))

    coordinates = [(n['x'], n['y']) for n in graph._node.values()]

    clustering = KMeans(n_clusters = n_clusters).fit(coordinates)

    clusters = clustering.labels_

    nodes_to_clusters = {k[idx]: clusters[idx] for idx in range(len(k))}
    clusters_to_nodes= {idx: k[clusters == idx] for idx in np.unique(clusters)}

    nodes = []

    for cluster_id, cluster in clusters_to_nodes.items():

        node = {
            field: fun([graph._node[n].get(field, np.nan) for n in cluster]) \
            for field, fun in node_functions.items()
        }

        nodes.append((cluster_id, node))

    edge_list = list(graph.edges)
    cluster_ids = list(clusters_to_nodes.keys())

    _adj = {
        s: {
            t: {
                'present': False,
                **{f: [] for f in edge_functions},
            } for t in cluster_ids
        } for s in cluster_ids
    }

    for edge in edge_list:

        s, t = edge
        _adj[nodes_to_clusters[s]][nodes_to_clusters[t]]['present'] = True

        for field, fun in edge_functions.items():

            _adj[nodes_to_clusters[s]][nodes_to_clusters[t]][field].append(
                graph._adj[s][t].get(field, np.nan)
            )

    edges = []

    for s in cluster_ids:
        for t in cluster_ids:
            if _adj[s][t]['present']:

                edge = {
                    field: fun(_adj[s][t].get(field, np.nan)) \
                    for field, fun in edge_functions.items()
                }

                edges.append((s, t, edge))

    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)

    return g
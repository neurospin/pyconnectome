##########################################################################
# NSAp - Copyright (C) CEA, 2013 - 2016
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Mapping of the Structural Core of the Human Cerebral Cortex.
"""

# System import
import os
import numpy
import networkx as nx

# Bct import
import bct


def create_graph(connectome, labels):
    """ Create a graph structure from the connectome matrix.

    Parameters
    ----------
    connectome: array (N, N)
        a matrix representing the structural connections.
    labels: list of str (N,)
        the labels used to create the connectome matrix.

    Returns
    -------
    graph: Graph
        a graph structure.
    """
    graph = nx.from_numpy_matrix(connectome)
    for index, name in enumerate(labels):
        name = name.rstrip("\n")
        graph.node[index] = {"label": name}
    return graph


def basic_network_analysis(graph, outdir=None):
    """ Compute basic network features.

    Network features:

    degrees:
        node degree is the number of links connected to the node.
    strengths:
        node strength is the sum of weights of links connected to the node.
    ccs:
        clustering coefficient is a measure of the degree to which nodes in
        a graph tend to cluster together.
    avg_ccs:
        the average clustering coefficient.
    components:
        the sorted by size connected components in the graph.
    bet_cen:
        betweenness centrality is an indicator of a node's centrality in a
        network. It is equal to the number of shortest paths from all vertices
        to all others that pass through that node.  A node with high
        betweenness centrality has a large influence on the transfer of items
        through the network, under the assumption that item transfer follows
        the shortest paths.
    clo_cen:
        closeness centrality (or closeness) of a node is a measure of
        centrality in a network, calculated as the sum of the length of the
        shortest paths between the node and all other nodes in the graph. Thus
        the more central a node is, the closer it is to all other nodes.
    eig_cen:
        eigenvector centrality (also called eigencentrality) is a measure of
        the influence of a node in a network. It assigns relative scores to all
        nodes in the network based on the concept that connections to
        high-scoring nodes contribute more to the score of the node in
        question than equal connections to low-scoring nodes.

    Note:

    Vertices that primarily either sit at the center of a hub or acts a
    bridge between two hubs have higher betweenness centrality. The bridge
    vertices have high betweenness because all paths connecting the hubs pass
    through them, and the hub center vertices have high betweenness because
    all intra-hub paths pass through them.

    Parameters
    ----------
    graph: Graph
        the graph reprensenting the connectome network.
    outdir: str (optional, default None)
        if specified save some snapshots.

    Returns
    -------
    outputs: dict
        the network features.
    snaps: list of file
        the generates snaps.
    """
    # Degree distribution
    degrees = graph.degree()

    # Stengths distribution
    strengths = graph.degree(weight="weight")

    # Clustering coefficients
    ccs = nx.clustering(graph)
    avg_ccs = sum(ccs.values()) / len(ccs)

    # Extract the main connected component
    components = sorted(nx.connected_components(graph), key=len, reverse=True)

    # Node centrality measures
    bet_cen = nx.betweenness_centrality(graph, weight="weight")
    clo_cen = nx.closeness_centrality(graph)
    eig_cen = nx.eigenvector_centrality(graph, weight="weight")
    high_bet_cen = highest_centrality(bet_cen)
    high_clo_cen = highest_centrality(clo_cen)
    high_eig_cen = highest_centrality(eig_cen)

    # Summarize results in a dictionnary
    params = locals()
    outputs = dict([(name, params[name])
                    for name in ("degrees", "strengths", "ccs", "avg_ccs",
                                 "components", "bet_cen", "clo_cen",
                                 "eig_cen", "high_bet_cen", "high_clo_cen",
                                 "high_eig_cen")])

    # Snaps
    snaps = []
    if outdir is not None:
        import pylab as plt
        if not os.path.isdir(outdir):
            raise ValueError("'{0}' is not a valid directory.".format(outdir))
        for measures, label in [(degrees, "degree"),
                                (strengths, "strength"),
                                (bet_cen, "betweenness"),
                                (clo_cen, "closeness"),
                                (eig_cen, "eigencentrality")]:
            outfile = os.path.join(outdir, label + ".png")
            snaps.append(outfile)
            plt.figure()
            n, bins, patches = plt.hist(measures.values(), 20, normed=1,
                                        facecolor="green", alpha=0.75)
            plt.xlabel(label)
            plt.ylabel("Number of nodes")
            plt.savefig(outfile)
            plt.close()

    return outputs, snaps


def highest_centrality(cent_dict):
    """ Extract the highest central node.

    Parameters
    ----------
    cent_dict: dict
        the centrality measures for each node.

    Returns
    -------
    high_cent: 2-uplet
        a tuple (node, value) with the node with largest value.
    """
    cent_items = [(b, a) for (a, b) in cent_dict.items()]
    cent_items.sort()
    cent_items.reverse()
    return tuple(reversed(cent_items[0]))


def advanced_network_analysis(graph, kstep=1, sstep=600., outdir=None):
    """ Map structural cores to delineate network modules, and to identify
    hub regions that link distinct clusters.

    Definition:

    The k-core is the largest subnetwork comprising nodes of degree at
    least k.

    The s-core is the largest subnetwork comprising nodes of strength at
    least s.

    The optimal community structure is a subdivision of the
    network into nonoverlapping groups of nodes which maximizes the number
    of within-group edges and minimizes the number of between-group edges.

    The rich club coefficient, R, at level k is the fraction of edges that
    connect nodes of degree k or higher out of the maximum number of edges
    that such nodes might share.

    Network features:

    kcores:
        each node associated k-core number.
    klevels:
        size of s-cores when the s-level increase.
    scores:
        each node associated s-core number.
    slevels:
        size of s-cores when the s-level increase.
    community:
        the computed community structure.
    qstat:
        the objective modularity function optimized q-statistic.
    rich_clubs:
        vector of rich-club coefficients for levels 1 to klevel=the maximum
        degree of the adjacency matrix.

    Parameters
    ----------
    graph: Graph
        the graph reprensenting the connectome network.
    kstep: int (optional, default 1)
        the k-core size increment.
    sstep: float (optional, default 600.)
        the s-core size increment.
    outdir: str (optional, default None)
        if specified save some snapshots.

    Returns
    -------
    outputs: dict
        the network features.
    snaps: list of file
        the generates snaps.
    """
    adjacency_matrix = numpy.ascontiguousarray(nx.to_numpy_matrix(graph))

    # K-core decomposition
    k = 0
    kcores = numpy.zeros(adjacency_matrix.shape[0], dtype=int)
    klevels = []
    kxs = []
    processed_indices = set()
    while True:
        if not graph.is_directed():
            kcore, kn, peelorder, peellevel = bct.kcore_bu(adjacency_matrix, k,
                                                           peel=True)
            klevels.append(kn)
            kxs.append(k)
        else:
            kcore, kn, peelorder, peellevel = bct.kcore_bd(adjacency_matrix, k,
                                                           peel=True)
        for indices in peelorder:
            new_indices = set(indices) - processed_indices
            processed_indices = processed_indices.union(new_indices)
            kcores[list(new_indices)] = k
        if kn == 0:
            break
        k += 1

    # S-core decompositon
    scores = numpy.zeros(adjacency_matrix.shape[0], dtype=int)
    slevels = []
    sxs = []
    processed_indices = set()
    s = 0
    while True:
        if not graph.is_directed():
            score, sn = bct.score_wu(adjacency_matrix, s)
            slevels.append(sn)
            sxs.append(s)
        else:
            raise NotImplementedError
        ff = numpy.where(score == 0)
        for node_index in ff[0]:
            if node_index in processed_indices:
                continue
            if not (score[node_index, :] == 0).all():
                continue
            scores[node_index] = s
            processed_indices.add(node_index)
        if sn == 0:
            break
        s += sstep

    # Community detection
    community, qstat = bct.community_louvain(
        adjacency_matrix, gamma=1, ci=None, B="modularity", seed=None)

    # Hub detection
    if not graph.is_directed():
        rich_clubs = bct.rich_club_wu(adjacency_matrix, klevel=None)
    else:
        rich_clubs = bct.rich_club_wd(adjacency_matrix, klevel=None)

    # Summarize results in a dictionnary
    params = locals()
    outputs = dict([(name, params[name])
                    for name in ("kcores", "klevels", "kxs", "scores",
                                 "slevels", "sxs", "community", "qstat",
                                 "rich_clubs")])

    # Snaps
    snaps = []
    if outdir is not None:
        import pylab as plt
        if not os.path.isdir(outdir):
            raise ValueError("'{0}' is not a valid directory.".format(outdir))
        for x, measures, label in [(kxs, klevels, "klevels"),
                                   (sxs, slevels, "slevels")]:
            outfile = os.path.join(outdir, label + ".png")
            snaps.append(outfile)
            plt.figure()
            plt.plot(x, measures, "-bo")
            plt.xlabel(label)
            plt.ylabel("Number of nodes")
            plt.savefig(outfile)
            plt.close()

    return outputs, snaps

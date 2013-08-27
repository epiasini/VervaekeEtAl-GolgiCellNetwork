#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
For n_trials times, load cell positions and edge information for the Vervaeke2010 algorithm saved on disk by the golgi_network_generate.py script, convert it to an unweighted undirected graph and save it as a graphml file. Finally, display some information on the statistical structure of the graph and compare it to the experimental spatial dependence of pairwise connectivity probability from Vervaeke2010 and to the theoretical (analytical) prediction for the degree distribution of an infinite network.
'''
import csv
import numpy as np
import networkx as nx
import itertools

from scipy.spatial import distance
from scipy import optimize
from scipy.stats import poisson, binom
from matplotlib import pyplot as plt

def vervaeke2010_spatial_dependence(r):
    return (-1745. + 1836./(1 + np.exp((r-267)/39)))/100

def infinite_net_analytical_sol(rho_goc=4.6e-6, l=80., a=0.856, r_0=122., delta=16.9):
    return a * rho_goc * l * np.pi * r_0**2 * (1 + np.pi**2./3.*(delta/r_0)**2 - 1./6.*(l/r_0)**2)

def calculate_degree(n=28, r_max=155., a=-17.45, b=-18.36, c=39, r_0=267, dx=0.01):
    x = np.arange(0, r_max, dx)
    numerical_sol = (n-1)*(a - (2*b)*np.trapz(degree_integrand(x, r_0=r_0, c=c), dx=dx)/r_max**2)
    a, r_0, delta = fermi_dirac_fit(plot=False)
    return numerical_sol, infinite_net_analytical_sol(a=a, r_0=r_0, delta=delta)

if __name__ == "__main__":
    n_trials = 100
    n_conn_types = 1
    positions = np.loadtxt(open("cell_positions.csv", "rb"), delimiter=",")
    n_cells = positions.shape[0]/n_trials
    positions = positions.reshape(n_trials, n_cells, 3)
    distances = np.array([distance.pdist(p) for p in positions])
    distances_squareform = np.array([distance.squareform(v) for v in distances])

    # load edge list for each trial
    edges = np.loadtxt(open("edge_lists.csv", "rb"), delimiter=",", dtype=np.int)
    edge_lists = [[]]
    previous_edge = (0,0)
    trial = 0
    for e in edges:
        if e[0] < previous_edge[0]:
            trial += 1
            edge_lists.append([])
        edge_lists[-1].append(e)
        previous_edge = e

    for k,l in enumerate(edge_lists):
        edge_lists[k] = np.array(l)
    # for each trial, create a temporary list where all edges are ordered (i<j)
    ordered_edge_lists = [l.tolist() for l in edge_lists]
    for edge_list in ordered_edge_lists:
        for k,e in enumerate(edge_list):
            if e[0] > e[1]:
                edge_list[k] = [e[1], e[0]]
    ordered_edge_lists = [[tuple(l2) for l2 in l1] for l1 in ordered_edge_lists]
    # for each trial, filter the ordered edge list and keep only its
    # unique elements. The result is a list of lists of all connected
    # cell pairs, regardless of the number of gap junctions between
    # them
    cell_pair_lol = [np.array(list(set(l))) for l in ordered_edge_lists]
    # but for the 2010 connection algorithm there shouldn't be any
    # multiedge in the graph
    for k, cell_pair_list in enumerate(cell_pair_lol):
        assert len(cell_pair_list)==len(ordered_edge_lists[k])
    graphs = [nx.empty_graph(n_cells, create_using=nx.Graph()) for each in cell_pair_lol]
    for k,g in enumerate(graphs):
        g.add_edges_from(cell_pair_lol[k])
        nx.write_graphml(g, '../dataSets/graphs/graph_2010_ncells{0}_trial{1:02}.graphml'.format(n_cells, k))

    average_clustering_coefficients = np.array([nx.average_clustering(g) for g in graphs])
    print("Average clustering: {0} ± {1}".format(average_clustering_coefficients.mean(),
                                                 np.sqrt(average_clustering_coefficients.var())))
    average_shortest_path_lengths = np.array([nx.average_shortest_path_length(g) for g in graphs])
    print("Average shortest path length: {0} ± {1}".format(average_shortest_path_lengths.mean(),
                                                           np.sqrt(average_shortest_path_lengths.var())))
    laplacian_spectra = [np.sort(nx.laplacian_spectrum(g)) for g in graphs]
    eigenratios = np.array([s[-1]/(s[2]) for s in laplacian_spectra])
    print("Average eigenratio: {0} ± {1}".format(eigenratios.mean(),
                                                 np.sqrt(eigenratios.var())))
    # for each trial, calculate the degree sequence of the
    # network. Note that the degree of a cell is defined as the number
    # of unique cells it's connected to; the number of gap junctions
    # on a cell is an obvious upper bound for the cell's degree.
    degree_sequences = np.array([[len([pair for pair in l if any((pair[0]==k, pair[1]==k))]) for k in range(n_cells)] for l in cell_pair_lol])
    # for each trial, for each connected pair, calculate the distance
    # between somata
    edge_lengths = []
    for trial, cell_pair_list in enumerate(cell_pair_lol):
        lengths = np.array([distances_squareform[trial][tuple(e)] for e in cell_pair_list])
        edge_lengths.append(lengths)

    alpha = degree_sequences.mean()
    #deg_dist_simulated = binom(n_cells-1, alpha/n_cells)
    #deg_dist_theoretical = binom(n_cells-1, infinite_net_analytical_sol(rho_goc=4.6e-6,
    #                                                                    l=80.,
    #                                                                    a=0.856,
    #                                                                    r_0=122.,
    #                                                                    delta=16.9)/n_cells)
    deg_dist_simulated = poisson([alpha])
    deg_dist_theoretical = poisson([infinite_net_analytical_sol(rho_goc=4.6e-6,
                                                                        l=80.,
                                                                        a=0.856,
                                                                        r_0=122.,
                                                                        delta=16.9)])
    print("Average degree is {0}".format(alpha))

    hist_all = np.histogram(distances, range=[0.,160.])
    hist_connected = np.histogram(np.concatenate(edge_lengths), range=[0.,160.])

    fig, ax1 = plt.subplots()
    ax1.hist(x=(np.ravel(distances), np.concatenate(edge_lengths)), range=[0., 160.])
    ax2 = ax1.twinx()
    ax1.set_xlabel(r'inter-somatic distance ($\mu m$)')
    # figure out the centres of the histogram bars on the x axis
    x = hist_all[1][:-1]+(hist_all[1][1]-hist_all[1][0])/2
    # plot the conected/total cells ratio
    ax2.plot(x, np.nan_to_num(np.asarray(hist_connected[0], dtype=np.float)/np.asarray(hist_all[0], dtype=np.float)), marker='o', color='r', label='Vervaeke2010 - model')
    # plot the fit to the experimental data (Vervaeke2010, figure 7)
    ax2.plot(x, vervaeke2010_spatial_dependence(x), marker='o', color='k', label='Vervaeke2010 - exp')
    ax2.set_ylim((0., 1.1))
    ax2.legend(loc='lower center')

    fig3, ax3 = plt.subplots()
    ax3.hist(list(itertools.chain(*degree_sequences)), bins=15, normed=True)
    k_range = np.arange(0,26,1)
    ax3.plot(k_range, deg_dist_theoretical.pmf(k_range), marker='o', label='analytical')
    ax3.plot(k_range, deg_dist_simulated.pmf(k_range), marker='o', label='poisson from sim. data')
    ax3.set_xlabel('degree')
    ax3.legend(loc='best')
    plt.show()

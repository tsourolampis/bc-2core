import networkx as nx
from collections import defaultdict, deque
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import random
import os


def _single_source_shortest_path_basic(G, s):
    S = []
    P = {v: [] for v in G}  # predecessors
    sigma = defaultdict(float)  # Initialize sigma as defaultdict of float
    D = {}  # distance from s
    sigma[s] = 1.0
    D[s] = 0
    Q = deque([s])
    while Q:  # use BFS to find shortest paths
        v = Q.popleft()
        S.append(v)
        Dv = D[v]
        sigmav = sigma[v]
        for w in G[v]:
            if w not in D:
                Q.append(w)
                D[w] = Dv + 1
            if D[w] == Dv + 1:  # this is a shortest path, count paths
                sigma[w] += sigmav
                P[w].append(v)
    return S, P, sigma, D



def _accumulate_basic_mem_efficient(betweenness, S, P, sigma, s, degree_1_seq):
    δ, ζ = dict.fromkeys(S, 0), dict.fromkeys(S, 0)
    while S:
        w = S.pop()
        coeff = (1 + δ[w]) / sigma[w]
        coeff_zeta = (degree_1_seq[w] + ζ[w]) / sigma[w]
        for v in P[w]:
            if v!=s:
                δ[v] += sigma[v] * coeff
                ζ[v]  += sigma[v] * coeff_zeta
        if w != s :
            betweenness[w] += (1+degree_1_seq[s])*(δ[w]+ζ[w])
    return betweenness



def bc_one_round_peeling_mem_efficient(G, k=None):

    betweenness = dict.fromkeys(G, 0.0)

    n = len(G.nodes())
    original_nodes = G.nodes()

    V1 = [node for node, degree in G.degree() if degree == 1]
    deg1_neighbors = {node: sum(1 for neighbor in G.neighbors(node) if neighbor in V1)
                                  for node in G.nodes() if G.degree(node) >= 2}

    for u in V1:
        deg1_neighbors[u] = 0.0

    print(f"Fraction of degree-1 nodes {len(V1)/n}")
    Y = {node for node in G.nodes() if node in deg1_neighbors and deg1_neighbors[node] > 0 and G.degree(node) >= 2}


    Gtilde = G.copy()
    Gtilde.remove_nodes_from(V1)
    seed = random.Random(42)
    flag = False

    if k == None:
        nodes = list(Gtilde.nodes())
    elif  k >= Gtilde.number_of_nodes():
        nodes = list(Gtilde.nodes())
        print("Sample size ",k," greater than number of nodes ",Gtilde.number_of_nodes())
    else:
        nodes = seed.sample(list(Gtilde.nodes()), k)
        flag = True

    for idx, s in enumerate(nodes):
        S, P, sigma, D = _single_source_shortest_path_basic(Gtilde, s)
        betweenness  = _accumulate_basic_mem_efficient(betweenness, S, P, sigma, s, deg1_neighbors)


    if flag:
        betweenness = {node:  len(Gtilde.nodes())/k * value for node, value in betweenness.items()}



    for u in Gtilde.nodes():
        betweenness[u] += (2*n-3-deg1_neighbors[u])*deg1_neighbors[u]

    for u in G.nodes():
         betweenness[u] = betweenness[u]/( (len(G.nodes()) -1)*(len(G.nodes()) -2) )


    return betweenness



def _accumulate_zetas(S, P, sigma, s, delta_matrix, zeta_matrix, degree_1_seq):

    delta = dict.fromkeys(S, 0.0)
    ζ = dict.fromkeys(S, 0.0)


    while S:

        w = S.pop()
        coeff = (1 + delta[w]) / sigma[w]
        coeff_zeta = (degree_1_seq[w] + ζ[w]) / sigma[w]

        for v in P[w]:
            if v != s:
                c = sigma[v] * coeff
                delta[v] += c
                ζ[v]  += sigma[v] * coeff_zeta


        delta_matrix[s][w] = delta[w]
        zeta_matrix[s][w] = ζ[w]


def bc_one_round_peeling(G, k=None):

    betweenness = dict.fromkeys(G, 0.0)  # b[v]=0 for v in G
    n = len(G.nodes())
    original_nodes = G.nodes()

    V1 = [node for node, degree in G.degree() if degree == 1]
    deg1_neighbors = {node: sum(1 for neighbor in G.neighbors(node) if neighbor in V1)
                                  for node in G.nodes() if G.degree(node) >= 2}

    for u in V1:
        deg1_neighbors[u] = 0.0

    Y = {node for node in G.nodes() if node in deg1_neighbors and deg1_neighbors[node] > 0 and G.degree(node) >= 2}


    delta_matrix = np.zeros((n, n))
    final_delta_matrix = np.zeros((n, n))

    zeta_matrix = np.zeros((n, n))
    eta_matrix = np.zeros((n, n))

    Gtilde = G.copy()
    Gtilde.remove_nodes_from(V1)



    seed = random.Random(42)

    flag = False

    if k == None:
        nodes = list(Gtilde.nodes())
    elif  k >= Gtilde.number_of_nodes():
        nodes = list(Gtilde.nodes())
        print("Sample size ",k," greater than number of nodes ",Gtilde.number_of_nodes())
    else:
        nodes = seed.sample(list(Gtilde.nodes()), k)
        flag = True

    for idx, s in enumerate(nodes):
        S, P, sigma, D = _single_source_shortest_path_basic(Gtilde, s)
        _accumulate_zetas(          S,
                                        P,
                                        sigma,
                                        s,
                                        delta_matrix,
                                        zeta_matrix,
                                        deg1_neighbors)

    # update delta values

    if flag:
        delta_matrix = len(Gtilde.nodes())/k * delta_matrix
        zeta_matrix  = len(Gtilde.nodes())/k * zeta_matrix

    for s in original_nodes:
        for u in original_nodes:
            if u in V1:
                final_delta_matrix[s][u] = 0
            elif s in V1 and u not in V1 and s in G[u]:
                final_delta_matrix[s][u] = n-2
            elif s in V1 and u not in V1 and s not in G[u]:
                y = list(G.neighbors(s))[0]
                final_delta_matrix[s][u] = delta_matrix[y][u]+zeta_matrix[y][u]
            elif s not in V1 and u not in V1:
                final_delta_matrix[s][u] = delta_matrix[s][u]+zeta_matrix[s][u]

    for u in Y:
        excluded_nodes = set(G.nodes()) - {u} - set(G.neighbors(u)).intersection(V1)
        for s in excluded_nodes:
            eta_matrix[s][u] = deg1_neighbors[u]




    final_delta_matrix += eta_matrix


    return final_delta_matrix



def shortest_path_through_u(G, s, t, u):
    
    
    all_paths = list(nx.all_shortest_paths(G, source=s, target=t))
    σst = len(all_paths)
    σst_u = sum(u in path for path in all_paths)
    return σst_u, σst


def bc_multiple_round_peeling_sigmas(G):
    
    Gi = G.copy()
    n = len(G.nodes())
    layers = []
    deg1_neighbors = []
    Y_sets = []  # Store Y sets for each layer

    i = 1 
    while True:
        V1 = [node for node, degree in Gi.degree() if degree == 1]
        if not V1:
            break
        else:
            print(f"Layer {i} has {len(V1)} degree-1 nodes")
            i+=1 
        layers.append(V1)
        current_deg1_neighbors = {node: 
                                  sum(1 for neighbor in Gi.neighbors(node) 
                                      if neighbor in V1) 
                                  for node in Gi.nodes() if Gi.degree(node) >= 2}
        deg1_neighbors.append(current_deg1_neighbors)
        Y = {node for node in Gi.nodes() if node in current_deg1_neighbors and current_deg1_neighbors[node] > 0 and Gi.degree(node) >= 2}
        Y_sets.append(Y)
        Gi.remove_nodes_from(V1)

        
#    bc_gi = {node: 0 for node in G.nodes()}
    
    if Gi.nodes():
        bc_gi = nx.betweenness_centrality(Gi)
    
    for u in G.nodes():
        if u not in bc_gi.keys():
            bc_gi[u]=0.0 
            
    bc_scores = {node: 0.0 for node in G.nodes()}   
    iteration = 1 
    
    if not layers: 
        return nx.betweenness_centrality(G)
        
    while layers:

        V1 = layers.pop()
        Y = Y_sets.pop()
        current_deg1_neighbors = deg1_neighbors.pop()
        
        if len(Gi.nodes())==0:
            Gi =  G.subgraph(V1).copy()
            
        else:
            n_before = len(Gi.nodes())
            n_after = n_before + len(V1) 
            
            for u in Gi.nodes():
                bc_scores[u] = bc_gi[u] * (n_before - 1) * (n_before - 2) 
                k = current_deg1_neighbors[u]                        
                bc_scores[u] += k * (k - 1) + 2 * k * (n_after - k - 1)  
        
                Z = 0
            
            
                for y in Gi.nodes():
                    for yprime in Y:
                        assert Y.issubset(Gi.nodes())
                        if y != u and yprime != u and y != yprime:
                            σst_u, σst = shortest_path_through_u(Gi, y, yprime, u)
                            if σst != 0:
                                Z += current_deg1_neighbors[yprime] * σst_u / σst

                bc_scores[u] += 2 * Z
               
                Z = 0 
                
                for y in current_deg1_neighbors:
                    for yprime in current_deg1_neighbors:
                        if y != u and yprime != u and y != yprime:
                            σst_u, σst = shortest_path_through_u(Gi, y, yprime, u)
                            if σst != 0:
                                Z += current_deg1_neighbors.get(y, 0) * current_deg1_neighbors.get(yprime, 0) * σst_u / σst
                bc_scores[u] += Z

                if n_after > 2:
                    bc_scores[u] /= (n_after-1)*(n_after-2)

                
                bc_gi.update({k: v for k, v in bc_scores.items() if v != 0})
           

        Gi = G.subgraph(Gi.nodes() | V1).copy()

    return bc_scores



    
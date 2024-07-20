# Compute Betweenness Centralities from 2-core


A central task in network analysis is to identify important nodes in a graph. Betweenness centrality (BC) is a popular centrality measure  that captures the significance of nodes based on the number of shortest path each node intersects with. Despite the important insights it can provide, the best known exact computation due to Brandes is prohibitively expensive for large-scale networks. 

In this repo, we provide a prototype implementation of the algorithm presented in ``A Note on Computing Betweenness Centrality  from the 2-core'' that removes 1-round of peeling of degree-1 nodes, computes the BC scores on the remaining graph and then updates the values. 


from bc_alg import *
from scipy.io import mmread

def bc_scores(delta):
    bc = {}
    n = len(delta)
    
    for i in range(n):
        bc[i] =1/((n-1)*(n-2)) * sum(delta[:,i])

    return bc

def read_mtx_file(file_path):
    """Read a .mtx file and convert it to an undirected NetworkX graph."""
    G = nx.Graph()
    print(file_path)

    with open(file_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith('%'):
                continue  # Skip comments
            parts = line.strip().split()
            if len(parts) == 3 and i == 2:
                print(f"Skipping matrix dimensions line: {line.strip()}")
                continue
            if len(parts) == 2:
                try:
                    u, v = int(parts[0]), int(parts[1])
                    G.add_edge(u, v)
                except ValueError as e:
                    print(f"Error parsing line {i}: {line.strip()} - {e}")
    print(f"Finished reading {file_path}. Number of edges: {G.number_of_edges()}")
    return G




def read_edges_file(file_path):
    """Read an .edges file and convert it to an undirected NetworkX graph."""
    G = nx.Graph()
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('%'):
                continue  # Skip comments
            # Try splitting by comma
            parts = line.strip().split(',')
            if len(parts) == 1:
                # If no comma is found, split by space
                parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    u, v = int(parts[0]), int(parts[1])
                    G.add_edge(u, v)
                except ValueError as e:
                    print(f"Error parsing line: {line.strip()} - {e}")
    return G


def read_graph_files(directory='/Users/labis/data/'):
    graph_files = [f for f in os.listdir(directory) if f.endswith('.mtx') or f.endswith('.edges')]
    graphs = []
    for file_name in graph_files:
        file_path = os.path.join(directory, file_name)
        print(f"Reading file: {file_path}")
        if file_name.endswith('.mtx'):
            G = read_mtx_file(file_path)
        elif file_name.endswith('.edges'):
            G = read_edges_file(file_path)
        else:
            continue
        print(f"Adding graph to list. Number of nodes: {G.number_of_nodes()}, Number of edges: {G.number_of_edges()}")
        if G.number_of_edges()>0:
            graphs.append((G, file_name))
    return graphs


def run_mem_efficient_exact(G,  filename='', to_plot=False):

        start_time = time.time()    
        print("Groundtruth BC scores") 
        scores = nx.betweenness_centrality(G) 
        end_time = time.time()
        time_groundtruth = end_time - start_time
        print(f"Groundtruth BC scores required: {time_groundtruth:.4f} seconds")

        start_time = time.time()    
        print("Memory efficient version with peeling") 
        bc_babis = bc_one_round_peeling_mem_efficient(G)
        end_time = time.time()
        time_peeling = end_time - start_time
        print(f"Memory efficient version with peeling required: {time_peeling:.4f} seconds")
    
        df = pd.DataFrame({
            'groundtruth': list(scores.values()),
            'bc_babis': list(bc_babis.values())
             })
        
        if to_plot: 
            plt.figure(figsize=(8, 6))
            plt.plot(df['groundtruth'], df['bc_babis'], 'o--', label='1-round of peeling')
            plt.plot(df['groundtruth'], df['groundtruth'], 'xr--', label='exact y=x line')


            plt.xlabel('BC groundtruth')
            plt.ylabel('BC computation of mine')
            plt.legend()
    
        return df, time_groundtruth , time_peeling


def run_mem_efficient_sample(G, k=10, filename=''):
        start_time = time.time()    
        print("Brandes-Pich BC scores") 
        scores_brandes = nx.betweenness_centrality(G, k) 
        end_time = time.time()
        time_brandes = end_time - start_time
        print(f"Groundtruth BC scores required: {time_brandes:.4f} seconds")

        start_time = time.time()    
        print("Memory efficient version with peeling and sampling") 
        bc_babis_sample = bc_one_round_peeling_mem_efficient(G, k)
        end_time = time.time()
        time_peeling_sample = end_time - start_time
        print(f"Memory efficient version with peeling required: {time_peeling_sample:.4f} seconds")

        df = pd.DataFrame({
            'brandes': list(scores_brandes.values()),
            'bc_babis_sample': list(bc_babis_sample.values())
             })

        return df,  time_brandes, time_peeling_sample 
    
    
    
def run_mem_efficient_range_k(G,  filename=''):
    
    print("Dataset\t"+filename)
    # run exact once 
    df_exact, time_groundtruth , time_peeling = run_mem_efficient_exact(G,  filename)
    bc_exact = df_exact['groundtruth'] 

    times = []
    times.append(( time_groundtruth, time_peeling, time_groundtruth/time_peeling))
    
    l1_results = []
    l2_results = []
    for k in [5] + list(range(10, 101, 10)):
        if k > len(G):
            continue
        df_approx,  time_brandes, time_peeling_sample  = run_mem_efficient_sample(G, k, filename)    
        
        times.append((k, time_brandes,time_peeling_sample, time_brandes/time_peeling_sample))
        df = pd.concat([df_exact, df_approx], axis=1)
        
        
        bc_brandes_pich = df['brandes']
        bc_babis_sample  = df['bc_babis_sample']
                       
        bc_diff_brandes = np.array([bc_exact[node] - bc_brandes_pich[node] for node in bc_exact.keys() ])
        bc_diff_babis = np.array([bc_exact[node] - bc_babis_sample[node] for node in bc_exact.keys() ])
        
        l1_brandes = np.sum( np.abs(bc_diff_brandes) )
        l2_brandes = np.sqrt (np.sum( bc_diff_brandes**2 ) )
    
        l1_babis = np.sum( np.abs( bc_diff_babis ) )
        l2_babis = np.sqrt( np.sum( bc_diff_babis**2 ) )

        l1_results.append((k, l1_brandes, l1_babis ))
        l2_results.append((k, l2_brandes, l2_babis ))
    
    
    x = l1_results
    k_values = [item[0] for item in x]
    col2_values = [100*item[1]/sum(df['groundtruth'] ) for item in x]
    col3_values = [100*item[2]/sum(df['groundtruth'] ) for item in x]

    plt.figure()
    plt.plot(k_values, col3_values, 'o--', label='1-round Peeling', markersize=8)
    plt.plot(k_values, col2_values, 's--', label='Brandes-Pich', markersize=8)


    plt.title(r'Relative $\ell_1$ error for '+filename, fontsize=14)
    plt.xlabel(r'Pivot sample size $k$', fontsize=14)
    plt.ylabel(r'$\frac{|bc_{true}-bc_{approx}|_1}{|bc_{true}|_1} \times 100\%$', fontsize=14)
    plt.legend()
    
    plt.savefig(filename+'_range_k_relative_err.jpg', format='jpg', dpi=300)  # Save as high-resolution JPG
    plt.savefig(filename+'_range_k_relative_err.pdf', format='jpg', dpi=300)  # Save as high-resolution JPG
     

    return times, l1_results
     

def run_mem_efficient_synthetic(k=10, core_size=50, central_node=0, degree_1_nodes_list = [100, 500, 1000, 2000, 3000, 4000, 5000, 10000], filename='synth'):
    avg_times = [] 
    std_times = [] 
    
    avg_l1_results = [] 
    std_l1_results = []
    

    for total_degree_1_nodes in degree_1_nodes_list:
        times_tmp = []
        l1_results = [] 
        for _ in range(5): 
            G = create_bowtie_graph_with_degree_1_nodes(core_size, central_node, total_degree_1_nodes)
            
            df_exact, time_groundtruth , time_peeling = run_mem_efficient_exact(G,  filename)
            df_approx,  time_brandes, time_peeling_sample  = run_mem_efficient_sample(G, k, filename)            
            df = pd.concat([df_exact, df_approx], axis=1)

            bc_exact = df['groundtruth'] 
            bc_brandes_pich = df['brandes']
            bc_babis_sample  = df['bc_babis_sample']
        
            times_tmp.append([time_groundtruth , time_peeling, time_brandes, time_peeling_sample])
            

            l1_results = []
            bc_diff_brandes = np.array([bc_exact[node] - bc_brandes_pich[node] for node in bc_exact.keys() ])
            bc_diff_babis = np.array([bc_exact[node] - bc_babis_sample[node] for node in bc_exact.keys() ])
        
            l1_brandes = np.sum( np.abs(bc_diff_brandes) )/sum(df['groundtruth'] )
            l1_babis = np.sum( np.abs( bc_diff_babis ) )/sum(df['groundtruth'] )
            l1_results.append([l1_brandes, l1_babis ])
    
        times_tmp = np.array(times_tmp)
        l1_results_tmp = np.array(l1_results)
        
        avg_times.append(np.mean(times_tmp, axis=0))
        std_times.append(np.std(times_tmp, axis=0))
        avg_l1_results.append(np.mean(l1_results_tmp, axis=0))
        std_l1_results.append(np.std(l1_results_tmp, axis=0))
    
    avg_l1_results = np.array(avg_l1_results)
    l1_brandes = avg_l1_results[:, 0]
    l1_peeling = avg_l1_results[:, 1]

    plt.figure(figsize=(10, 6))

    n_values = [degree_1_nodes+core_size for degree_1_nodes in degree_1_nodes_list] 
    
    plt.plot(n_values, l1_peeling, marker='o', label=f'1-round of peeling, {k}')
    plt.plot(n_values, l1_brandes, marker='s', label=f'Brandes-Pich, {k}')

    
    plt.title(r'Avg. Relative $\ell_1$ error', fontsize=14)
    plt.xlabel(r'|V|', fontsize=14)
    plt.ylabel(r'$\frac{|bc_{true}-bc_{approx}|_1}{|bc_{true}|_1} \times 100\%$', fontsize=14)
 
    plt.legend()
    plt.grid(True)
    plt.show()
       
    plt.savefig(filename+'_range_V1_relative_err.jpg', format='jpg', dpi=300)  # Save as high-resolution JPG
    plt.savefig(filename+'_range_V1_relative_err.pdf', format='jpg', dpi=300)  # Save as high-resolution JPG
    
    return avg_times, std_times, avg_l1_results, std_l1_results


def create_bowtie_graph_with_degree_1_nodes(core_size, central_node, total_degree_1_nodes):
    """
    Create a bowtie-like graph structure with a specified number of degree 1 nodes.
    
    Parameters:
    - core_size: The number of core nodes
    - central_node: The index of the central node with high betweenness centrality
    - total_degree_1_nodes: The total number of degree 1 nodes to add
    
    Returns:
    - G: The generated graph with the specified structure and degree 1 node distribution
    """
    G = nx.Graph()
    
    core_nodes = list(range(core_size))
    G.add_nodes_from(core_nodes)
    
    # Creating a bowtie-like structure for high betweenness centrality
    subgraph1 = list(range(0, core_size // 2))
    subgraph2 = list(range(core_size // 2, core_size))
    
    # Connect subgraph1 to central node and central node to subgraph2
    G.add_edges_from((n, central_node) for n in subgraph1 if n != central_node)
    G.add_edges_from((central_node, n) for n in subgraph2 if n != central_node)
    
    # Fully connect the nodes within each subgraph
    G.add_edges_from((i, j) for i in subgraph1 for j in subgraph1 if i < j)
    G.add_edges_from((i, j) for i in subgraph2 for j in subgraph2 if i < j)
    
    degree_1_node_id = core_size
    remaining_nodes = total_degree_1_nodes
    for i in range(core_size):
        if i == central_node:
            continue  # Skip the central node
        
        num_degree_1_nodes = total_degree_1_nodes // (2 ** (i + 1))
        if num_degree_1_nodes > remaining_nodes:
            num_degree_1_nodes = remaining_nodes
        
        for _ in range(num_degree_1_nodes):
            G.add_node(degree_1_node_id)
            G.add_edge(i, degree_1_node_id)
            degree_1_node_id += 1
        
        remaining_nodes -= num_degree_1_nodes
        if remaining_nodes <= 0:
            break
    
    return G



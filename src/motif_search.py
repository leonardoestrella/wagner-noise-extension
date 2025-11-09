"""
Motif search and analysis for gene regulatory networks.
Identifies and classifies network motifs including feedforward and feedback loops,
and calculates their concentrations relative to all subgraphs of the same size.
"""
import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

"""
-----------------
Loops Names
-----------------
"""
FFL_loops_names = ["Coherent Type 1 (C1)", "Coherent Type 2 (C2)", 
                   "Coherent Type 3 (C3)","Coherent Type 4 (C4)",
                   "Incoherent Type 1 (I1)", "Incoherent Type 2 (I2)",
                   "Incoherent Type 3 (I3)", "Incoherent Type 4 (I4)"]

"""
------------------
VISUALIZATION
------------------
"""

def visualize_graph(adj_matrix, phen_opt=None):
    """
    Visualize a weighted directed graph with edge colors by sign and node colors by phenotype.

    Parameters
    ----------
    adj_matrix : np.ndarray
        Weighted adjacency matrix representing a directed graph.
    phen_opt : list or np.ndarray, optional
        Optional phenotype or node values for coloring:
        1 -> green, -1 -> red, else gray.

    Returns
    -------
    None
    """
    g = ig.Graph.Weighted_Adjacency(adj_matrix.tolist(), mode='directed')
    g.vs["name"] = list(range(adj_matrix.shape[0]))

    # Edge visual properties
    edge_colors = ["green" if w > 0 else "red" for w in g.es["weight"]]
    edge_widths = [abs(w) * 4 for w in g.es["weight"]]

    # Node visual properties
    if phen_opt is not None:
        node_colors = ["green" if v == 1 else "red" if v == -1 else "gray" for v in phen_opt]
    else:
        node_colors = ["gold"] * len(g.vs)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 7))
    ig.plot(
        g,
        target=ax,
        layout="kk",
        vertex_label=g.vs["name"],
        vertex_color=node_colors,
        vertex_size=35,
        edge_color=edge_colors,
        edge_width=edge_widths,
        edge_arrow_size=1.3,
    )
    plt.title("Gene Regulatory Network Visualization")
    plt.show()

"""
------------------
FEEDFORWARD LOOPS
------------------
"""

def classify_ffl(graph, x_idx, y_idx, z_idx):
    '''Classifies an FFL based on the signs of its edge weights.'''
    s_xy = np.sign(graph[x_idx, y_idx])
    s_yz = np.sign(graph[y_idx, z_idx])
    s_xz = np.sign(graph[x_idx, z_idx])
    sign_indirect = s_xy * s_yz
    is_coherent = (sign_indirect == s_xz)
    if is_coherent:
        if s_xy > 0 and s_yz > 0 and s_xz > 0: return "Coherent Type 1 (C1)"
        if s_xy > 0 and s_yz < 0 and s_xz < 0: return "Coherent Type 2 (C2)"
        if s_xy < 0 and s_yz > 0 and s_xz < 0: return "Coherent Type 3 (C3)"
        if s_xy < 0 and s_yz < 0 and s_xz > 0: return "Coherent Type 4 (C4)"
    else:
        if s_xy > 0 and s_yz > 0 and s_xz < 0: return "Incoherent Type 1 (I1)"
        if s_xy > 0 and s_yz < 0 and s_xz > 0: return "Incoherent Type 2 (I2)"
        if s_xy < 0 and s_yz > 0 and s_xz > 0: return "Incoherent Type 3 (I3)"
        if s_xy < 0 and s_yz < 0 and s_xz < 0: return "Incoherent Type 4 (I4)"
    return "Unknown"

def count_ffl_types(adj_matrix, visualize = False, phen_opt = None):
    '''Counts occurrences of each FFL type in an adjacency matrix.'''

    # Step 1: Your adjacency matrix
    gene_names = [i for i in range(adj_matrix.shape[0])]

    if visualize:
        visualize_graph(adj_matrix, phen_opt)

    # Step 2: Create the igraph object
    g = ig.Graph.Weighted_Adjacency(adj_matrix.tolist(), mode='directed')
    g.vs["name"] = gene_names

    # Step 3: Find all FFLs
    ffl_motif = ig.Graph.Formula('X -> Y, X -> Z, Y -> Z')
    ffl_instances = g.get_subisomorphisms_vf2(ffl_motif)
    results = []
    for instance in ffl_instances:
        x_idx, y_idx, z_idx = [instance[ffl_motif.vs.find(name=n).index] for n in ['X', 'Y', 'Z']]
        ffl_type = classify_ffl(g, x_idx, y_idx, z_idx)
        results.append({
            "X_node": g.vs[x_idx]["name"],
            "Y_node": g.vs[y_idx]["name"],
            "Z_node": g.vs[z_idx]["name"],
            "FFL_Type": ffl_type
        })

    # Step 4: Count occurrences of each FFL type
    type_counts = {}

    for ffl_name in FFL_loops_names: # Initialize all names
        type_counts[ffl_name] = 0

    for result in results:
        ffl_type = result["FFL_Type"]
        if ffl_type not in type_counts:
            type_counts[ffl_type] = 0
        type_counts[ffl_type] += 1

    return results, type_counts

"""
------------------
FEEDBACK LOOOPS
------------------
"""

def classify_feedback_loop(graph, loop_nodes):
    """
    Classify a feedback loop as reinforcing, balancing, or neutral based on edge signs.

    Parameters
    ----------
    graph : igraph.Graph
        Directed graph with 'weight' attributes on edges.
    loop_nodes : list of int
        Node indices forming the feedback loop in order.

    Returns
    -------
    str
        "Reinforcing Feedback" if product of edge signs > 0,
        "Balancing Feedback" if < 0,
        "Neutral or Undefined" if the product is zero,
        "Invalid" if loop is incomplete.
    """
    signs = []
    for i in range(len(loop_nodes)):
        src = loop_nodes[i]
        tgt = loop_nodes[(i + 1) % len(loop_nodes)]
        eid = graph.get_eid(src, tgt, directed=True, error=False)
        if eid == -1:
            return "Invalid"
        signs.append(np.sign(graph.es[eid]['weight']))

    product_sign = np.prod(signs)
    if product_sign > 0:
        return "Reinforcing Feedback"
    elif product_sign < 0:
        return "Balancing Feedback"
    else:
        return "Neutral or Undefined"

def count_feedback_loops(adj_matrix, max_size=4):
    """
    Count and classify all feedback loops (directed cycles) up to a specified size.

    Parameters
    ----------
    adj_matrix : np.ndarray
        Weighted adjacency matrix representing a directed graph.
    max_size : int, optional
        Maximum loop size to consider.

    Returns
    -------
        loops_by_size: {
            loop_size: {
                "Positive": [list of loops],
                "Negative": [list of loops],
                ...
            },
            ...
        },
        counts: {
            loop_size: {
                "Positive": int,
                "Negative": int,
                ...
            },
            ...
        }
    """
    # Step 1: Initialize igraph
    node_names = list(range(adj_matrix.shape[0]))
    g = ig.Graph.Weighted_Adjacency(adj_matrix.tolist(), mode='directed')
    g.vs["name"] = node_names

    # Step 2: Find all simple cycles up to max_size
    all_cycles = g.simple_cycles(min=2, max=max_size, output='vpath')

    # Step 3: Group results
    loops_by_size = {}
    counts = {}

    for cycle in all_cycles:
        loop_size = len(cycle)
        fbl_type = classify_feedback_loop(g, cycle)
        if fbl_type == "Invalid":
            continue

        loops_by_size.setdefault(loop_size, {})
        loops_by_size[loop_size].setdefault(fbl_type, [])
        loops_by_size[loop_size][fbl_type].append([g.vs[idx]["name"] for idx in cycle])

    # Step 4: Build counts dictionary
    for size, type_dict in loops_by_size.items():
        counts[size] = {fbl_type: len(loop_list) for fbl_type, loop_list in type_dict.items()}

    return loops_by_size, counts

def calculate_motif_concentration(adj_matrix, motif_counts, subgraph_size):
    """
    Calculate the concentration of each motif type relative to all subgraphs of the same size.
    
    Parameters
    ----------
    adj_matrix : np.ndarray
        Weighted adjacency matrix of the network
    motif_counts : dict
        Dictionary containing counts of each motif type
    subgraph_size : int
        Size of the subgraphs to consider (e.g., 3 for FFLs, 2-4 for feedback loops)
    
    Returns
    -------
    dict
        Dictionary of motif concentrations where each value is the frequency of that
        motif type divided by the total number of connected subgraphs of the same size
    """
    # Create igraph object
    g = ig.Graph.Weighted_Adjacency(adj_matrix.tolist(), mode='directed')
    
    # Get all possible k-node subgraphs
    total_subgraphs = 0
    for nodes in combinations(range(g.vcount()), subgraph_size):
        sub = g.subgraph(nodes)
        if sub.is_connected(mode='weak'):
            total_subgraphs += 1
    
    # If there are no subgraphs of this size, return zero concentrations
    if total_subgraphs == 0:
        return {motif: 0.0 for motif in motif_counts.keys()}
    
    # Calculate concentration for each motif type
    concentrations = {}
    for motif_type, count in motif_counts.items():
        concentrations[motif_type] = count / total_subgraphs
        
    return concentrations

def analyze_network_motifs(adj_matrix, max_fbl_size=4):
    """
    Comprehensive analysis of network motifs including FFLs and feedback loops.
    
    Parameters
    ----------
    adj_matrix : np.ndarray
        Weighted adjacency matrix of the network
    max_fbl_size : int, optional
        Maximum size of feedback loops to consider
        
    Returns
    -------
    dict
        Complete analysis including:
        - FFL counts and concentrations
        - Feedback loop counts and concentrations by size
        - Summary statistics
    """
    results = {}
    
    # Analyze FFLs (always size 3)
    ffl_details, ffl_counts = count_ffl_types(adj_matrix)
    ffl_concentrations = calculate_motif_concentration(adj_matrix, ffl_counts, 3)
    
    results['feedforward_loops'] = {
        'counts': ffl_counts,
        'concentrations': ffl_concentrations,
        'details': ffl_details
    }
    
    # Analyze feedback loops of different sizes
    feedback_details, feedback_counts = count_feedback_loops(adj_matrix, max_fbl_size)
    
    results['feedback_loops'] = {
        'by_size': {}
    }
    
    for size in range(2, max_fbl_size + 1):
        if size in feedback_counts:
            concentrations = calculate_motif_concentration(
                adj_matrix, feedback_counts[size], size
            )
            results['feedback_loops']['by_size'][size] = {
                'counts': feedback_counts[size],
                'concentrations': concentrations,
                'details': feedback_details.get(size, {})
            }
    
    # Calculate summary statistics
    total_ffls = sum(ffl_counts.values())
    total_fbls = sum(
        sum(size_counts.values())
        for size_counts in feedback_counts.values()
    )
    reinforcing_total = sum(
        size_counts.get('Reinforcing Feedback', 0)
        for size_counts in feedback_counts.values()
    )
    balancing_total = sum(
        size_counts.get('Balancing Feedback', 0)
        for size_counts in feedback_counts.values()
    )
    
    results['summary'] = {
        'total_ffls': total_ffls,
        'total_fbls': total_fbls,
        'ffl_proportion': {
            'coherent': sum(count for type_name, count in ffl_counts.items() if 'Coherent' in type_name),
            'incoherent': sum(count for type_name, count in ffl_counts.items() if 'Incoherent' in type_name)
        },
        'fbl_proportion': {
            'reinforcing': reinforcing_total,
            'balancing': balancing_total
        }
    }
    
    return results

def plot_motif_concentrations(results, plot_type='all'):
    """
    Visualize motif concentrations as bar plots.
    
    Parameters
    ----------
    results : dict
        Results from analyze_network_motifs
    plot_type : str
        'all', 'ffl', or 'fbl' to specify which motifs to plot
    """
    if plot_type in ['all', 'ffl']:
        # Plot FFL concentrations
        ffl_conc = results['feedforward_loops']['concentrations']
        plt.figure(figsize=(12, 6))
        plt.bar(ffl_conc.keys(), ffl_conc.values())
        plt.xticks(rotation=45)
        plt.title('Feed-Forward Loop Concentrations')
        plt.ylabel('Concentration')
        plt.tight_layout()
        plt.show()
    
    if plot_type in ['all', 'fbl']:
        # Plot feedback loop concentrations by size
        sizes = sorted(results['feedback_loops']['by_size'].keys())
        if not sizes:
            print("No feedback loop data available to plot.")
            return
        fig, axes = plt.subplots(1, len(sizes), figsize=(5*len(sizes), 5))
        if len(sizes) == 1:
            axes = [axes]
        
        for ax, size in zip(axes, sizes):
            conc = results['feedback_loops']['by_size'][size]['concentrations']
            ax.bar(conc.keys(), conc.values())
            ax.set_title(f'{size}-Node Feedback Loops')
            ax.tick_params(axis='x', rotation=45)
            ax.set_ylabel('Concentration')
        
        plt.tight_layout()
        plt.show()

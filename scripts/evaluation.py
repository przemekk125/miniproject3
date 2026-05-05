import os
import torch
import numpy as np
from torch_geometric.datasets import TUDataset
import networkx as nx
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt

def load_samples():
    dataset = TUDataset(root='./data/', name='MUTAG')
    
    er_path = "output/generated_er_graphs.pt"
    vae_path = "output/generated_vae_graphs.pt"
    
    er_data = torch.load(er_path, map_location="cpu", weights_only=False)
    vae_data = torch.load(vae_path, map_location="cpu", weights_only=False)
    
    print(f"Successfully loaded {len(er_data)} Baseline graphs.")
    print(f"Successfully loaded {len(vae_data)} VAE graphs.")
    
    return dataset, er_data, vae_data

def get_graph_hashes(graph_list):
    """Converts PyG Data objects to WL hashes for fast comparison."""
    hashes = []
    for data in graph_list:
        g = to_networkx(data, to_undirected=True)
        h = nx.weisfeiler_lehman_graph_hash(g)
        hashes.append(h)
    return hashes

def evaluate_novelty_uniqueness(gen_graphs, train_graphs):
    gen_hashes = get_graph_hashes(gen_graphs)
    train_hashes = set(get_graph_hashes(train_graphs))
    
    total = len(gen_hashes)
    unique_hashes = set(gen_hashes)
    
    novel_count = sum(1 for h in gen_hashes if h not in train_hashes)
    unique_count = len(unique_hashes)
    novel_unique_count = len(unique_hashes - train_hashes)
    
    return {
        "Novel (%)": (novel_count / total) * 100,
        "Unique (%)": (unique_count / total) * 100,
        "Novel+Unique (%)": (novel_unique_count / total) * 100
    }

def get_stats(graph_list):
    degrees = []
    clusterings = []
    centralities = []
    
    for g in graph_list:
        degrees.extend([d for _, d in g.degree()])
        
        clusterings.extend(list(nx.clustering(g).values()))
        try:
            c = nx.eigenvector_centrality(g, max_iter=1000)
            centralities.extend(list(c.values()))
        except nx.PowerIterationFailedConvergence:
            continue
            
    return degrees, clusterings, centralities

if __name__ == "__main__":
    train_data, er_data, vae_data = load_samples()

    er_results = evaluate_novelty_uniqueness(er_data, train_data)
    vae_results = evaluate_novelty_uniqueness(vae_data, train_data)

    print(f"{'Model':<25} | {'Novel':<10} | {'Unique':<10} | {'Novel+Unique':<10}")
    print("-" * 65)
    print(f"{'Baseline':<25} | {er_results['Novel (%)']:>9.1f}% | {er_results['Unique (%)']:>9.1f}% | {er_results['Novel+Unique (%)']:>9.1f}%")
    print(f"{'Deep Generative Model':<25} | {vae_results['Novel (%)']:>9.1f}% | {vae_results['Unique (%)']:>9.1f}% | {vae_results['Novel+Unique (%)']:>9.1f}%")

    train_nx = [to_networkx(d, to_undirected=True) for d in train_data]
    er_nx = [to_networkx(d, to_undirected=True) for d in er_data]
    vae_nx = [to_networkx(d, to_undirected=True) for d in vae_data]

    stats_train = get_stats(train_nx)
    stats_er = get_stats(er_nx)
    stats_vae = get_stats(vae_nx)

    metrics = ["Node Degree", "Clustering Coefficient", "Eigenvector Centrality"]
    models = ["Empirical", "Baseline (ER)", "Deep Model (VAE)"]
    data_matrix = [stats_train, stats_er, stats_vae]

    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharey='row')

    for i, metric_name in enumerate(metrics):
        # Determine common bins for the row
        all_values = data_matrix[0][i] + data_matrix[1][i] + data_matrix[2][i]
        if i == 0: # Degree (discrete)
            bins = np.arange(min(all_values), max(all_values) + 2) - 0.5
        else: # Clustering/Centrality (0 to 1)
            bins = np.linspace(0, 1, 25)

        for j, model_name in enumerate(models):
            ax = axes[i, j]
            ax.hist(data_matrix[j][i], bins=bins, color='skyblue', edgecolor='black', density=True)
            
            if i == 0: ax.set_title(model_name)
            if j == 0: ax.set_ylabel(metric_name)
            ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig("output/statistics_grid.png")
    plt.show()

import numpy as np
import pandas as pd
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
import torch
import argparse

def sample_er_graph(N, r, device="cpu"):
    # Sample undirected edges once on the upper triangle and mirror them.
    triu = torch.triu_indices(N, N, offset=1, device=device)
    mask = torch.rand(triu.size(1), device=device) < r
    src = triu[0, mask]
    dst = triu[1, mask]

    edge_index = torch.cat(
        [torch.stack([src, dst], dim=0), torch.stack([dst, src], dim=0)],
        dim=1,
    )
    x = torch.ones((N, 1), dtype=torch.float, device=device)
    return Data(x=x, edge_index=edge_index)

def sample(n, R, values, probs, device="cpu"):
    data_list = []
    for _ in range(n):
        samp = R[np.random.choice(values, size=1, p=probs)]
        N, r = samp.index[0], samp.values[0]

        data = sample_er_graph(N, r, device=device)
        data_list.append(data)
    return data_list


def main(device, n, output_path):
    
    dataset = TUDataset(root='./data/', name='MUTAG').to(device)

    nodes,n_edges = [], []
    for graph in dataset:
        nodes.append(graph.x.shape[0])
        # MUTAG in PyG stores undirected edges in both directions (u,v) and (v,u).
        n_edges.append(graph.edge_index.shape[1] // 2)
    nodes = np.array(nodes)
    n_edges = np.array(n_edges)
    df = pd.DataFrame({'nodes': nodes, 'n_edges': n_edges})

    # make empirical distribution of number of nodes
    n_nodes = df.groupby('nodes').size()
    values = n_nodes.index.values
    probs = n_nodes.values/n_nodes.values.sum()

    # compute average graph density for each number of nodes
    avg_nodes = df.groupby('nodes').mean()
    R = avg_nodes['n_edges'] / (avg_nodes.index * (avg_nodes.index - 1) / 2)
    if ((R < 0) | (R > 1)).any():
        raise ValueError("Invalid density value computed. Check the data and calculations.")

    data_list = sample(n, R, values, probs, device=device)
    torch.save(data_list, output_path)

    print(f"Saved {len(data_list)} PyG graphs to: {output_path}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the code on (e.g., 'cpu' or 'cuda')")
    parser.add_argument("--n", type=int, default=1000, help="Number of graphs to sample")
    parser.add_argument("--output", type=str, default="output/generated_er_graphs.pt", help="Output .pt path for generated PyG Data list")
    args = parser.parse_args()
    main(device=args.device, n=args.n, output_path=args.output)
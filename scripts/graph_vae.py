import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data


class GNNEncoder(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, latent_dim, num_rounds):
        super().__init__()
        self.input_net = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU()
        )

        self.message_nets = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            for _ in range(num_rounds)
        ])

        self.update_nets = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            for _ in range(num_rounds)
        ])

        self.mu_net = nn.Linear(hidden_dim, latent_dim)
        self.logvar_net = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, edge_index, batch):
        state = self.input_net(x)
        num_nodes = x.size(0)
        num_graphs = int(batch.max().item()) + 1

        for message_net, update_net in zip(self.message_nets, self.update_nets):
            messages = message_net(state)
            aggregated = x.new_zeros((num_nodes, messages.size(1)))
            aggregated = aggregated.index_add(0, edge_index[1], messages[edge_index[0]])
            state = state + update_net(aggregated)

        graph_state = x.new_zeros((num_graphs, state.size(1)))
        graph_state = graph_state.index_add(0, batch, state)

        mu = self.mu_net(graph_state)
        logvar = self.logvar_net(graph_state)
        return mu, logvar


class GraphVAE(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, latent_dim, num_rounds, max_nodes, min_nodes):
        super().__init__()
        self.max_nodes = max_nodes
        self.min_nodes = min_nodes
        self.n_node_classes = max_nodes - min_nodes + 1
        self.n_pairs = max_nodes * (max_nodes - 1) // 2

        self.encoder = GNNEncoder(node_feature_dim, hidden_dim, latent_dim, num_rounds)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + self.n_node_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_pairs)
        )

    def encode_node_count(self, n_nodes, device):
        idx = n_nodes - self.min_nodes
        return F.one_hot(idx, num_classes=self.n_node_classes).float().to(device)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, n_nodes):
        n_onehot = self.encode_node_count(n_nodes, z.device)
        decoder_input = torch.cat([z, n_onehot], dim=1)
        return self.decoder(decoder_input)

    def forward(self, data):
        mu, logvar = self.encoder(data.x, data.edge_index, data.batch)
        z = self.reparameterize(mu, logvar)
        n_nodes = torch.bincount(data.batch)
        logits = self.decode(z, n_nodes)
        return logits, mu, logvar, n_nodes


def dense_targets(data, max_nodes):
    batch_size = int(data.batch.max().item()) + 1
    targets = data.x.new_zeros((batch_size, max_nodes, max_nodes))

    for g in range(batch_size):
        node_mask = data.batch == g
        global_nodes = torch.where(node_mask)[0]
        local_index = {int(v): i for i, v in enumerate(global_nodes)}

        edges = data.edge_index.t()
        edges = edges[(data.batch[edges[:, 0]] == g) & (data.batch[edges[:, 1]] == g)]

        for src, dst in edges:
            i = local_index[int(src)]
            j = local_index[int(dst)]
            targets[g, i, j] = 1.0

    triu = torch.triu_indices(max_nodes, max_nodes, offset=1, device=targets.device)
    return targets[:, triu[0], triu[1]]


def pair_mask(n_nodes, max_nodes):
    masks = []
    triu = torch.triu_indices(max_nodes, max_nodes, offset=1, device=n_nodes.device)

    for n in n_nodes:
        valid = (triu[0] < n) & (triu[1] < n)
        masks.append(valid)

    return torch.stack(masks, dim=0)


def vae_loss(logits, targets, mask, mu, logvar, beta):
    bce = F.binary_cross_entropy_with_logits(
        logits[mask],
        targets[mask],
        reduction="mean"
    )

    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + beta * kl, bce, kl


def sample_graphs(model, n_graphs, train_node_counts, device):
    model.eval()
    sampled_graphs = []

    node_counts = np.random.choice(train_node_counts, size=n_graphs, replace=True)
    triu = torch.triu_indices(model.max_nodes, model.max_nodes, offset=1, device=device)

    with torch.no_grad():
        for n in node_counts:
            n_tensor = torch.tensor([n], dtype=torch.long, device=device)
            z = torch.randn((1, model.encoder.mu_net.out_features), device=device)
            logits = model.decode(z, n_tensor)
            probs = torch.sigmoid(logits[0])
            sampled_edges = torch.bernoulli(probs).bool()

            valid = (triu[0] < n) & (triu[1] < n) & sampled_edges
            src = triu[0, valid]
            dst = triu[1, valid]

            edge_index = torch.cat([
                torch.stack([src, dst], dim=0),
                torch.stack([dst, src], dim=0)
            ], dim=1)

            x = torch.ones((int(n), 1), dtype=torch.float, device=device)
            sampled_graphs.append(Data(x=x.cpu(), edge_index=edge_index.cpu()))

    return sampled_graphs


def main(args):
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    device = torch.device(args.device)
    dataset = TUDataset(root="./data/", name="MUTAG")

    rng = torch.Generator().manual_seed(args.seed)
    train_dataset, validation_dataset, test_dataset = random_split(
        dataset, (100, 44, 44), generator=rng
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size)

    train_node_counts = np.array([g.num_nodes for g in train_dataset])
    min_nodes = int(train_node_counts.min())
    max_nodes = int(train_node_counts.max())
    node_feature_dim = dataset.num_node_features

    model = GraphVAE(
        node_feature_dim=node_feature_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_rounds=args.message_passing_rounds,
        max_nodes=max_nodes,
        min_nodes=min_nodes
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for data in train_loader:
            data = data.to(device)
            logits, mu, logvar, n_nodes = model(data)

            targets = dense_targets(data, max_nodes)
            mask = pair_mask(n_nodes, max_nodes)

            loss, bce, kl = vae_loss(logits, targets, mask, mu, logvar, args.beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.num_graphs

        if epoch % args.print_every == 0:
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for data in validation_loader:
                    data = data.to(device)
                    logits, mu, logvar, n_nodes = model(data)
                    targets = dense_targets(data, max_nodes)
                    mask = pair_mask(n_nodes, max_nodes)
                    loss, _, _ = vae_loss(logits, targets, mask, mu, logvar, args.beta)
                    val_loss += loss.item() * data.num_graphs

            print(
                f"Epoch {epoch:04d} | "
                f"train loss {total_loss / len(train_dataset):.4f} | "
                f"val loss {val_loss / len(validation_dataset):.4f}"
            )

    generated_graphs = sample_graphs(
        model=model,
        n_graphs=args.n_samples,
        train_node_counts=train_node_counts,
        device=device
    )

    torch.save(generated_graphs, args.output)
    print(f"Saved {len(generated_graphs)} generated graphs to: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--message-passing-rounds", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="output/generated_graphvae_graphs.pt")
    args = parser.parse_args()

    main(args)
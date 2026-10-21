"""Heterogeneous GNN recommender: components and forward pass (Milestone 3, Part D).

    starting features --(per-type linear)--> 128-d
      --> HeteroLayer 1 --> ReLU --> HeteroLayer 2 --> L2-normalised article vectors
    user vector = mean of the GNN vectors of the user's 50 most recent articles
                  (learned default vector when there are none)
    score       = dot(user vector, candidate article vector)

HeteroLayer: one WeightedMeanConv per edge type, outputs summed per destination node type,
plus the node's own previous vector through a separate (per node type) weight matrix.

Layer choice (PyG 2.8.1): no built-in layer does a weighted mean and works between two
node types (bipartite) with a separate self term:
    GraphConv  edge_weight yes, but "mean" divides sum(w * x) by the neighbour count
               (not a weighted mean) and adds its own root weight for every edge type
    SAGEConv   mean + optional root weight, but no edge_weight
    GCNConv    edge_weight, but not bipartite, so unusable inside HeteroConv
So WeightedMeanConv implements sum(w * x_j) / sum(w) (a plain mean when the edge type has
no weights), with no self term; HeteroLayer adds the self term once per node type.
"""

import tomllib

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import HeteroConv, MessagePassing
from torch_geometric.utils import scatter

from recsys.paths import CONFIGS_DIR


def load_model_config() -> dict:
    with open(CONFIGS_DIR / "model.toml", "rb") as f:
        return tomllib.load(f)


class WeightedMeanConv(MessagePassing):
    """Neighbour aggregation for one edge type: sum(w * x_j) / sum(w), then a linear map.

    Without edge weights every w = 1, i.e. a plain mean. Nodes with no incoming edge of this
    type receive zeros. No self term (added by HeteroLayer).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="add")
        self.lin = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x: Tensor | tuple[Tensor, Tensor], edge_index: Tensor, edge_weight: Tensor | None = None) -> Tensor:
        x_src, x_dst = x if isinstance(x, tuple) else (x, x)
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
        # Normalise weights per destination node so they sum to 1 (on the sampled subgraph).
        total = scatter(edge_weight, edge_index[1], dim=0, dim_size=x_dst.size(0), reduce="sum")
        norm = edge_weight / total[edge_index[1]].clamp_min(1e-12)
        out = self.propagate(edge_index, x=(x_src, x_dst), norm=norm, size=(x_src.size(0), x_dst.size(0)))
        return self.lin(out)

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        return norm.view(-1, 1) * x_j


class HeteroLayer(nn.Module):
    """Sum of per-edge-type neighbour means + own previous vector through a separate weight."""

    def __init__(self, node_types: list[str], edge_types: list[tuple], dim: int):
        super().__init__()
        self.conv = HeteroConv({et: WeightedMeanConv(dim, dim) for et in edge_types}, aggr="sum")
        self.self_lin = nn.ModuleDict({nt: nn.Linear(dim, dim) for nt in node_types})

    def forward(self, x_dict: dict, edge_index_dict: dict, edge_weight_dict: dict) -> dict:
        messages = self.conv(x_dict, edge_index_dict, edge_weight_dict=edge_weight_dict)
        return {nt: self.self_lin[nt](x) + messages.get(nt, 0) for nt, x in x_dict.items()}


class GNNEncoder(nn.Module):
    """Per-type projection to `hidden_dim`, then `num_layers` HeteroLayers (ReLU between)."""

    def __init__(self, metadata: tuple, in_dims: dict[str, int], num_entity_types: int,
                 hidden_dim: int = 128, num_layers: int = 2, entity_type_dim: int = 64):
        super().__init__()
        node_types, edge_types = metadata
        self.entity_type_emb = nn.Embedding(num_entity_types, entity_type_dim)
        dims = {**in_dims, "entity_type": entity_type_dim}
        self.proj = nn.ModuleDict({nt: nn.Linear(dims[nt], hidden_dim) for nt in node_types})
        self.layers = nn.ModuleList(HeteroLayer(node_types, edge_types, hidden_dim) for _ in range(num_layers))

    @classmethod
    def from_data(cls, data: HeteroData, cfg: dict | None = None) -> "GNNEncoder":
        cfg = cfg or load_model_config()
        in_dims = {nt: data[nt].x.size(1) for nt in data.node_types if "x" in data[nt]}
        return cls(data.metadata(), in_dims, data["entity_type"].num_nodes,
                   hidden_dim=cfg["gnn"]["hidden_dim"], num_layers=cfg["gnn"]["num_layers"],
                   entity_type_dim=cfg["features"]["entity_type_dim"])

    def forward(self, data: HeteroData) -> dict[str, Tensor]:
        """Node vectors for every node in `data` (a full graph or a sampled subgraph)."""
        x_dict = {}
        for nt in data.node_types:
            if nt == "entity_type":
                # Sampled subgraphs carry the original node ids in n_id.
                ids = data[nt].n_id if "n_id" in data[nt] else torch.arange(data[nt].num_nodes)
                x = self.entity_type_emb(ids.to(self.entity_type_emb.weight.device))
            else:
                x = data[nt].x
            x_dict[nt] = self.proj[nt](x)

        edge_index_dict = {et: data[et].edge_index for et in data.edge_types}
        edge_weight_dict = {et: data[et].edge_weight for et in data.edge_types if "edge_weight" in data[et]}
        for i, layer in enumerate(self.layers):
            x_dict = layer(x_dict, edge_index_dict, edge_weight_dict)
            if i < len(self.layers) - 1:
                x_dict = {nt: F.relu(x) for nt, x in x_dict.items()}
        x_dict["article"] = F.normalize(x_dict["article"], dim=-1)
        return x_dict


def article_loader(data: HeteroData, article_index: Tensor, batch_size: int, fanout: list[int] | None = None,
                   shuffle: bool = False) -> NeighborLoader:
    """Sample up to fanout[k] neighbours per edge type at hop k around the given articles."""
    fanout = fanout or load_model_config()["gnn"]["fanout"]
    return NeighborLoader(data, num_neighbors={et: list(fanout) for et in data.edge_types},
                          input_nodes=("article", article_index), batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def encode_all_articles(encoder: GNNEncoder, data: HeteroData, batch_size: int = 2048,
                        fanout: list[int] | None = None) -> Tensor:
    """GNN vector for every article, computed batch by batch on sampled subgraphs."""
    out = torch.empty(data["article"].num_nodes, encoder.layers[0].self_lin["article"].out_features)
    for batch in article_loader(data, torch.arange(data["article"].num_nodes), batch_size, fanout):
        seeds = batch["article"].batch_size
        out[batch["article"].n_id[:seeds]] = encoder(batch)["article"][:seeds].cpu()
    return out


class UserEncoder(nn.Module):
    """Mean of the user's history article vectors; a learned default vector if there are none."""

    def __init__(self, dim: int):
        super().__init__()
        self.default = nn.Parameter(torch.randn(dim) * 0.01)

    def forward(self, article_vectors: Tensor, history: Tensor) -> Tensor:
        """history: [n_users, max_len] article row indices, padded with -1."""
        mask = history >= 0
        vectors = article_vectors[history.clamp_min(0)] * mask.unsqueeze(-1)
        count = mask.sum(dim=1, keepdim=True)
        mean = vectors.sum(dim=1) / count.clamp_min(1)
        return torch.where(count > 0, mean, self.default.expand_as(mean))


def score(user_vectors: Tensor, candidate_vectors: Tensor) -> Tensor:
    """Dot product per (user, candidate) pair: [n] x [n, d] -> [n]."""
    return (user_vectors * candidate_vectors).sum(dim=-1)

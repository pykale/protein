"""Self-contained PyTorch DrugBAN implementation.

Adapted from peizhenbai/DrugBAN (MIT, commit 9923f8c) without importing DGL or
the upstream repository at runtime. The architecture and tensor conventions
follow the original MolecularGCN, ProteinCNN, BANLayer, and MLPDecoder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from torch.nn import functional as F
try:
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:  # pragma: no cover - compatibility with older supported PyTorch
    from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader

from kale_protein.auto.weights import load_checkpoint_state_dict, resolve_pretrained_weight
from kale_protein.modalities.small_molecule.processors import RDKitGraphProcessor


PROTEIN_ALPHABET = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, "F": 7,
    "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, "O": 13, "N": 14,
    "Q": 15, "P": 16, "S": 17, "R": 18, "U": 19, "T": 20,
    "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25,
}


def encode_protein(sequence: str, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    sequence = str(sequence).upper()[:max_length]
    tokens = torch.zeros(max_length, dtype=torch.long)
    mask = torch.zeros(max_length, dtype=torch.bool)
    for index, residue in enumerate(sequence):
        tokens[index] = PROTEIN_ALPHABET.get(residue, 0)
        mask[index] = residue in PROTEIN_ALPHABET
    return tokens, mask


def _section(config, name: str) -> dict[str, Any]:
    value = config.get(name, {})
    return value if isinstance(value, dict) else {}


def _arch(config, key: str, default: Any) -> Any:
    return _section(config, "drugban").get(key, default)


def _training(config, key: str, default: Any) -> Any:
    return _section(config, "training").get(key, default)


def _device_of(module: nn.Module) -> torch.device:
    return next(module.parameters()).device


def _move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


class DrugBANCollator:
    """Collate raw or preprocessed DTI samples into tensor graph batches."""

    def __init__(self, config):
        self.config = config
        streams = config.get_streams()
        drug_kwargs = dict(streams["drug"].processor_kwargs)
        self.max_drug_nodes = int(drug_kwargs.get("max_nodes", 290))
        self.max_protein_length = int(
            streams["target"].processor_kwargs.get("max_length", 1000)
        )
        self.molecule_processor = RDKitGraphProcessor(
            input_key=streams["drug"].input_key,
            max_nodes=self.max_drug_nodes,
        )
        kernels = (
            int(_arch(config, "protein_kernel_1", 3)),
            int(_arch(config, "protein_kernel_2", 6)),
            int(_arch(config, "protein_kernel_3", 9)),
        )
        self.minimum_protein_length = sum(kernel - 1 for kernel in kernels) + 1

    def __call__(self, samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
        samples = list(samples)
        if not samples:
            raise ValueError("DrugBAN cannot collate an empty batch")

        drug_items = []
        target_items = []
        labels = []
        ids = []
        for sample in samples:
            if not isinstance(sample, dict):
                raise TypeError(f"DrugBAN samples must be dictionaries, got {type(sample).__name__}")
            drug = sample.get("drug")
            if not (isinstance(drug, dict) and "node_features" in drug):
                drug = self.molecule_processor.transform(sample)
            target = sample.get("target")
            if not isinstance(target, dict):
                target = {"sequence": sample.get("sequence", "")}
            drug_items.append(drug)
            target_items.append(target)
            if "label" in sample:
                labels.append(float(sample["label"]))
            ids.append(sample.get("id"))

        batch = {
            "drug": self.collate_drugs(drug_items),
            "target": self.collate_proteins(target_items),
            "ids": ids,
        }
        if labels:
            if len(labels) != len(samples):
                raise ValueError("Either every sample in a DrugBAN batch must have a label or none may")
            batch["label"] = torch.tensor(labels, dtype=torch.float32)
        return batch

    def collate_drugs(self, items: Iterable[dict[str, Any]]) -> dict[str, Any]:
        graphs = list(items)
        if not graphs:
            raise ValueError("Cannot collate an empty molecule batch")
        max_nodes = max(int(torch.as_tensor(graph["node_features"]).shape[0]) for graph in graphs)
        feature_dim = int(torch.as_tensor(graphs[0]["node_features"]).shape[-1])
        if feature_dim != 75:
            raise ValueError(f"DrugBAN expects 75 molecule node features, received {feature_dim}")

        features = torch.zeros((len(graphs), max_nodes, 75), dtype=torch.float32)
        features[:, :, 74] = 1.0
        adjacency = torch.eye(max_nodes, dtype=torch.float32).expand(len(graphs), -1, -1).clone()
        masks = torch.zeros((len(graphs), max_nodes), dtype=torch.bool)
        smiles = []
        atom_symbols = []
        for index, graph in enumerate(graphs):
            graph_features = torch.as_tensor(graph["node_features"], dtype=torch.float32)
            count = graph_features.shape[0]
            graph_mask = torch.as_tensor(
                graph.get("node_mask", torch.ones(count, dtype=torch.bool)), dtype=torch.bool
            )
            graph_adjacency = graph.get("adjacency")
            if graph_adjacency is None:
                graph_adjacency = _adjacency_from_edge_index(graph.get("edge_index"), count)
            graph_adjacency = torch.as_tensor(graph_adjacency, dtype=torch.float32)
            features[index, :count] = graph_features
            adjacency[index, :count, :count] = graph_adjacency
            masks[index, :count] = graph_mask
            smiles.append(graph.get("smiles"))
            atom_symbols.append(list(graph.get("atom_symbols", [])))
        return {
            "node_features": features,
            "adjacency": adjacency,
            "node_mask": masks,
            "smiles": smiles,
            "atom_symbols": atom_symbols,
        }

    def collate_proteins(self, items: Iterable[dict[str, Any]]) -> dict[str, Any]:
        proteins = list(items)
        if not proteins:
            raise ValueError("Cannot collate an empty protein batch")

        sequences = [protein.get("sequence") for protein in proteins]
        if all(sequence is not None for sequence in sequences):
            encoded = [encode_protein(sequence, self.max_protein_length) for sequence in sequences]
            return {
                "tokens": torch.stack([item[0] for item in encoded]),
                "attention_mask": torch.stack([item[1] for item in encoded]),
                "sequences": [str(sequence)[:self.max_protein_length] for sequence in sequences],
            }

        raw_tokens = [torch.as_tensor(protein["tokens"], dtype=torch.long).flatten() for protein in proteins]
        length = max(self.minimum_protein_length, max(tokens.numel() for tokens in raw_tokens))
        tokens = torch.zeros((len(proteins), length), dtype=torch.long)
        mask = torch.zeros((len(proteins), length), dtype=torch.bool)
        for index, item in enumerate(proteins):
            item_tokens = raw_tokens[index][:length]
            item_mask = torch.as_tensor(
                item.get("attention_mask", torch.ones(item_tokens.numel())), dtype=torch.bool
            ).flatten()[:item_tokens.numel()]
            tokens[index, :item_tokens.numel()] = item_tokens
            mask[index, :item_mask.numel()] = item_mask
        return {"tokens": tokens, "attention_mask": mask, "sequences": sequences}


def _adjacency_from_edge_index(edge_index, num_nodes):
    adjacency = torch.eye(num_nodes, dtype=torch.float32)
    if edge_index is None:
        return adjacency
    edges = torch.as_tensor(edge_index, dtype=torch.long)
    if edges.numel():
        adjacency[edges[0], edges[1]] = 1.0
    return adjacency


class DenseGraphConv(nn.Module):
    """DGL GraphConv-equivalent normalized message passing for dense batches."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, features, adjacency):
        degree = adjacency.sum(dim=-1).clamp_min(1.0)
        inv_sqrt_degree = degree.rsqrt()
        normalized = adjacency * inv_sqrt_degree.unsqueeze(-1) * inv_sqrt_degree.unsqueeze(-2)
        return self.linear(torch.bmm(normalized, features))


class MolecularGCN(nn.Module):
    def __init__(self, in_feats=75, dim_embedding=128, hidden_feats=(128, 128, 128)):
        super().__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        dimensions = [dim_embedding, *hidden_feats]
        self.layers = nn.ModuleList(
            DenseGraphConv(dimensions[index], dimensions[index + 1])
            for index in range(len(dimensions) - 1)
        )
        self.output_feats = dimensions[-1]

    def forward(self, graph):
        features = graph["node_features"].float()
        adjacency = graph["adjacency"].float()
        mask = graph["node_mask"].bool()
        hidden = self.init_transform(features)
        hidden = hidden * mask.unsqueeze(-1)
        for layer in self.layers:
            hidden = F.relu(layer(hidden, adjacency))
            hidden = hidden * mask.unsqueeze(-1)
        return hidden


class ProteinCNN(nn.Module):
    def __init__(self, embedding_dim=128, num_filters=(128, 128, 128), kernel_size=(3, 6, 9)):
        super().__init__()
        self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        channels = [embedding_dim, *num_filters]
        self.convolutions = nn.ModuleList(
            nn.Conv1d(channels[index], channels[index + 1], kernel_size[index])
            for index in range(3)
        )
        self.batch_norms = nn.ModuleList(nn.BatchNorm1d(channel) for channel in num_filters)
        self.output_feats = channels[-1]
        self.output_reduction = sum(size - 1 for size in kernel_size)

    def forward(self, protein):
        tokens = protein["tokens"].long()
        if tokens.shape[1] <= self.output_reduction:
            raise ValueError(
                f"Protein token length must exceed {self.output_reduction} for DrugBAN's convolutions"
            )
        hidden = self.embedding(tokens).transpose(1, 2)
        for convolution, batch_norm in zip(self.convolutions, self.batch_norms):
            hidden = batch_norm(F.relu(convolution(hidden)))
        return hidden.transpose(1, 2)


class FCNet(nn.Module):
    def __init__(self, dimensions, activation="ReLU", dropout=0.0):
        super().__init__()
        layers = []
        for input_dim, output_dim in zip(dimensions[:-1], dimensions[1:]):
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(input_dim, output_dim), dim=None))
            if activation:
                layers.append(getattr(nn, activation)())
        self.main = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.main(inputs)


class SafeBatchNorm1d(nn.BatchNorm1d):
    """Use running statistics for singleton vectors while retaining trainability."""

    def forward(self, inputs):
        if self.training and inputs.ndim == 2 and inputs.shape[0] == 1:
            return F.batch_norm(
                inputs, self.running_mean, self.running_var, self.weight, self.bias,
                False, self.momentum, self.eps,
            )
        return super().forward(inputs)


class BANLayer(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out, activation="ReLU", dropout=0.2, k=3):
        super().__init__()
        self.k = k
        self.h_dim = h_dim
        self.h_out = h_out
        self.v_net = FCNet([v_dim, h_dim * k], activation=activation, dropout=dropout)
        self.q_net = FCNet([q_dim, h_dim * k], activation=activation, dropout=dropout)
        self.p_net = nn.AvgPool1d(k, stride=k) if k > 1 else None
        self.h_mat = nn.Parameter(torch.empty(1, h_out, 1, h_dim * k).normal_())
        self.h_bias = nn.Parameter(torch.empty(1, h_out, 1, 1).normal_())
        self.batch_norm = SafeBatchNorm1d(h_dim)

    def attention_pooling(self, visual, question, attention):
        fused = torch.einsum("bvk,bvq,bqk->bk", visual, attention, question)
        if self.p_net is not None:
            fused = self.p_net(fused.unsqueeze(1)).squeeze(1) * self.k
        return fused

    def forward(self, visual, question, visual_mask=None, question_mask=None, softmax=False):
        projected_visual = self.v_net(visual)
        projected_question = self.q_net(question)
        attention = torch.einsum(
            "xhyk,bvk,bqk->bhvq", self.h_mat, projected_visual, projected_question
        ) + self.h_bias

        pair_mask = None
        if visual_mask is not None and question_mask is not None:
            pair_mask = visual_mask[:, None, :, None] & question_mask[:, None, None, :]
            attention = attention.masked_fill(~pair_mask, float("-inf") if softmax else 0.0)
        if softmax:
            flat = attention.flatten(2)
            flat = torch.softmax(flat, dim=-1)
            attention = torch.nan_to_num(flat).view_as(attention)
            if pair_mask is not None:
                attention = attention * pair_mask

        fused = sum(
            self.attention_pooling(projected_visual, projected_question, attention[:, head])
            for head in range(self.h_out)
        )
        return self.batch_norm(fused), attention


class MLPDecoder(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=512, out_dim=128, binary=1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = SafeBatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = SafeBatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = SafeBatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, inputs):
        hidden = self.bn1(F.relu(self.fc1(inputs)))
        hidden = self.bn2(F.relu(self.fc2(hidden)))
        hidden = self.bn3(F.relu(self.fc3(hidden)))
        return self.fc4(hidden)


class DrugBANNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        molecule_dim = int(_arch(config, "molecule_hidden_dim", 128))
        molecule_layers = int(_arch(config, "molecule_layers", 3))
        protein_dim = int(_arch(config, "protein_filter_dim", 128))
        ban_dim = int(_arch(config, "ban_hidden_dim", 256))
        self.molecule_encoder = MolecularGCN(
            in_feats=int(_arch(config, "atom_feature_dim", 75)),
            dim_embedding=int(_arch(config, "molecule_embedding_dim", 128)),
            hidden_feats=tuple(molecule_dim for _ in range(molecule_layers)),
        )
        self.protein_encoder = ProteinCNN(
            embedding_dim=int(_arch(config, "protein_embedding_dim", 128)),
            num_filters=(protein_dim, protein_dim, protein_dim),
            kernel_size=(
                int(_arch(config, "protein_kernel_1", 3)),
                int(_arch(config, "protein_kernel_2", 6)),
                int(_arch(config, "protein_kernel_3", 9)),
            ),
        )
        self.ban = weight_norm(
            BANLayer(
                v_dim=molecule_dim,
                q_dim=protein_dim,
                h_dim=ban_dim,
                h_out=int(_arch(config, "ban_heads", 2)),
                dropout=float(_arch(config, "ban_dropout", 0.2)),
                k=int(_arch(config, "ban_k", 3)),
            ),
            name="h_mat",
            dim=None,
        )
        self.decoder = MLPDecoder(
            in_dim=ban_dim,
            hidden_dim=int(_arch(config, "decoder_hidden_dim", 512)),
            out_dim=int(_arch(config, "decoder_output_dim", 128)),
            binary=int(_arch(config, "decoder_binary_dim", 1)),
        )

    def encode_drug(self, drug):
        return self.molecule_encoder(drug), drug["node_mask"].bool()

    def encode_target(self, target):
        embedding = self.protein_encoder(target)
        mask = target["attention_mask"].bool()[:, :embedding.shape[1]]
        return embedding, mask

    def fuse(self, drug_embedding, protein_embedding, drug_mask, protein_mask):
        return self.ban(drug_embedding, protein_embedding, drug_mask, protein_mask)

    def forward(self, batch):
        drug_embedding, drug_mask = self.encode_drug(batch["drug"])
        protein_embedding, protein_mask = self.encode_target(batch["target"])
        fused, attention = self.fuse(drug_embedding, protein_embedding, drug_mask, protein_mask)
        logits = self.decoder(fused).squeeze(-1)
        return {
            "logits": logits,
            "probabilities": torch.sigmoid(logits),
            "attention": attention,
            "drug_embedding": drug_embedding,
            "protein_embedding": protein_embedding,
            "drug_mask": drug_mask,
            "protein_mask": protein_mask,
            "fused_embedding": fused,
            "smiles": batch["drug"].get("smiles"),
            "atom_symbols": batch["drug"].get("atom_symbols"),
            "sequences": batch["target"].get("sequences"),
        }


class _MoleculeComponent(nn.Module):
    def __init__(self, network, collator):
        super().__init__()
        self.encoder = network.molecule_encoder
        self.collator = collator

    def embed(self, data):
        drug = _prepare_drug(data, self.collator)
        drug = _move_to_device(drug, _device_of(self.encoder))
        embedding = self.encoder(drug)
        mask = drug["node_mask"].bool()
        return {"embedding": embedding, "mask": mask, "smiles": drug.get("smiles"),
                "atom_symbols": drug.get("atom_symbols")}

    forward = embed


class _ProteinComponent(nn.Module):
    def __init__(self, network, collator):
        super().__init__()
        self.encoder = network.protein_encoder
        self.collator = collator

    def embed(self, data):
        target = _prepare_target(data, self.collator)
        target = _move_to_device(target, _device_of(self.encoder))
        embedding = self.encoder(target)
        mask = target["attention_mask"].bool()[:, :embedding.shape[1]]
        return {"embedding": embedding, "mask": mask, "sequences": target.get("sequences")}

    forward = embed


def _prepare_drug(data, collator):
    if isinstance(data, dict) and "node_features" in data:
        if torch.as_tensor(data["node_features"]).ndim == 3:
            return data
        return collator.collate_drugs([data])
    if isinstance(data, dict) and "drug" in data:
        return _prepare_drug(data["drug"], collator)
    items = data if isinstance(data, (list, tuple)) else [data]
    graphs = []
    for item in items:
        if isinstance(item, dict) and "node_features" in item:
            graphs.append(item)
        else:
            graphs.append(collator.molecule_processor.transform(item))
    return collator.collate_drugs(graphs)


def _prepare_target(data, collator):
    if isinstance(data, dict) and "tokens" in data:
        if torch.as_tensor(data["tokens"]).ndim == 2:
            return data
        return collator.collate_proteins([data])
    if isinstance(data, dict) and "target" in data:
        return _prepare_target(data["target"], collator)
    items = data if isinstance(data, (list, tuple)) else [data]
    proteins = [item if "sequence" in item else item.get("target", item) for item in items]
    return collator.collate_proteins(proteins)


class DrugBANModel(nn.Module):
    """AutoProteinModel entrypoint exposing direct protein/molecule components."""

    config_class = None

    def __init__(self, config, pretrain=False):
        super().__init__()
        self.config = config
        self.pretrain = pretrain
        self.collator = DrugBANCollator(config)
        self.network = DrugBANNetwork(config)
        self.network.eval()
        self.protein_model = _ProteinComponent(self.network, self.collator)
        self.molecule_model = _MoleculeComponent(self.network, self.collator)
        self.weight_path = None
        if pretrain:
            self.weight_path = resolve_pretrained_weight(config)
            _load_resolved_checkpoint(self.network, self.weight_path)

    def __iter__(self):
        yield self.protein_model
        yield self.molecule_model

    def component(self, stream_name):
        aliases = {
            "target": self.protein_model, "protein": self.protein_model,
            "drug": self.molecule_model, "molecule": self.molecule_model,
        }
        try:
            return aliases[stream_name]
        except KeyError as exc:
            raise KeyError(f"Unknown DrugBAN component {stream_name!r}; use target/protein or drug/molecule") from exc

    def embed(self, stream_data):
        if "drug" in stream_data and "target" in stream_data:
            return self.embed_components(stream_data)
        if "node_features" in stream_data or "smiles" in stream_data:
            return self.molecule_model.embed(stream_data)
        return self.protein_model.embed(stream_data)

    def embed_components(self, batch):
        return {
            "target": self.protein_model.embed(batch["target"]),
            "drug": self.molecule_model.embed(batch["drug"]),
        }

    def forward(self, batch):
        batch = _move_to_device(_prepare_batch(batch, self.collator), _device_of(self.network))
        return self.network(batch)


class DrugBANInteractionPredictor(nn.Module):
    """Trainable DrugBAN predictor with preprocessing, metrics, and checkpoint APIs."""

    config_class = None

    def __init__(self, config, pretrain=False):
        super().__init__()
        self.config = config
        self.pretrain = pretrain
        self.collator = DrugBANCollator(config)
        self.model = DrugBANNetwork(config)
        self.model.eval()
        self.protein_model = _ProteinComponent(self.model, self.collator)
        self.molecule_model = _MoleculeComponent(self.model, self.collator)
        self.weight_path = None
        if pretrain:
            self.weight_path = resolve_pretrained_weight(config)
            _load_resolved_checkpoint(self.model, self.weight_path)

    def component(self, stream_name):
        return {"target": self.protein_model, "protein": self.protein_model,
                "drug": self.molecule_model, "molecule": self.molecule_model}[stream_name]

    def embed_components(self, batch):
        prepared = _move_to_device(_prepare_batch(batch, self.collator), _device_of(self.model))
        drug_embedding, drug_mask = self.model.encode_drug(prepared["drug"])
        protein_embedding, protein_mask = self.model.encode_target(prepared["target"])
        return {
            "target": {"embedding": protein_embedding, "mask": protein_mask,
                       "sequences": prepared["target"].get("sequences")},
            "drug": {"embedding": drug_embedding, "mask": drug_mask,
                     "smiles": prepared["drug"].get("smiles"),
                     "atom_symbols": prepared["drug"].get("atom_symbols")},
        }

    def forward(self, batch_or_protein_embedding, molecule_embedding=None):
        if molecule_embedding is not None:
            protein = _embedding_parts(batch_or_protein_embedding)
            molecule = _embedding_parts(molecule_embedding)
            fused, attention = self.model.fuse(
                molecule[0], protein[0], molecule[1], protein[1]
            )
            logits = self.model.decoder(fused).squeeze(-1)
            return {
                "logits": logits,
                "probabilities": torch.sigmoid(logits),
                "attention": attention,
                "drug_mask": molecule[1],
                "protein_mask": protein[1],
                "atom_symbols": molecule_embedding.get("atom_symbols") if isinstance(molecule_embedding, dict) else None,
                "sequences": batch_or_protein_embedding.get("sequences") if isinstance(batch_or_protein_embedding, dict) else None,
            }
        batch = _move_to_device(
            _prepare_batch(batch_or_protein_embedding, self.collator), _device_of(self.model)
        )
        return self.model(batch)

    def make_dataloader(self, data, batch_size=None, shuffle=False, num_workers=0):
        if isinstance(data, DataLoader):
            return data
        if isinstance(data, dict):
            data = [data]
        return DataLoader(
            data,
            batch_size=batch_size or int(_training(self.config, "batch_size", 64)),
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collator,
        )

    def predict(self, batch_or_dataset, batch_size=None):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            if _looks_like_single_or_batched_dict(batch_or_dataset):
                output = self(batch_or_dataset)
                result = _detach_output(output)
            else:
                outputs = [
                    _detach_output(self(_move_to_device(batch, _device_of(self.model))))
                    for batch in self.make_dataloader(batch_or_dataset, batch_size=batch_size)
                ]
                result = _merge_outputs(outputs)
        self.train(was_training)
        return result

    def fit(
        self,
        train_data,
        valid_data=None,
        epochs=None,
        learning_rate=None,
        batch_size=None,
        optimizer=None,
    ):
        epochs = int(epochs or _training(self.config, "epochs", 100))
        learning_rate = float(learning_rate or _training(self.config, "learning_rate", 5e-5))
        loader = self.make_dataloader(train_data, batch_size=batch_size, shuffle=True)
        optimizer = optimizer or torch.optim.Adam(self.parameters(), lr=learning_rate)
        history = []
        self.train()
        for _ in range(epochs):
            total_loss = 0.0
            examples = 0
            for batch in loader:
                if "label" not in batch:
                    raise ValueError("DrugBAN fit() requires a label for every training sample")
                batch = _move_to_device(batch, _device_of(self.model))
                optimizer.zero_grad()
                output = self.model(batch)
                labels = batch["label"].float().view_as(output["logits"])
                loss = F.binary_cross_entropy_with_logits(output["logits"], labels)
                loss.backward()
                optimizer.step()
                count = labels.numel()
                total_loss += float(loss.detach()) * count
                examples += count
            history.append(total_loss / max(examples, 1))
        result = {"loss": history[-1] if history else None, "history": history, "epochs": epochs}
        if valid_data is not None:
            result["validation"] = self.evaluate(valid_data, batch_size=batch_size)
        return result

    def evaluate(self, data, batch_size=None, threshold=0.5):
        from kale_protein.tasks.drug_target_interaction.metrics import compute_metrics

        output = self.predict(data, batch_size=batch_size)
        labels = _labels_from_data(data, self.collator, batch_size)
        return compute_metrics(labels, output["probabilities"], threshold=threshold)

    def extract_attention(self, batch_or_dataset, batch_size=None):
        output = self.predict(batch_or_dataset, batch_size=batch_size)
        return {
            key: output.get(key)
            for key in ("attention", "drug_mask", "protein_mask", "atom_symbols", "sequences", "smiles")
        }

    def save_checkpoint(self, path, optimizer=None, extra=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": self.model.state_dict(),
            "config": self.config.to_dict(),
        }
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        if extra is not None:
            payload["extra"] = extra
        torch.save(payload, path)
        return path

    def load_checkpoint(self, path, strict=True, map_location="cpu"):
        return _load_resolved_checkpoint(self.model, path, strict=strict, map_location=map_location)


def _prepare_batch(data, collator):
    if isinstance(data, dict) and "drug" in data and "target" in data:
        drug = data["drug"]
        target = data["target"]
        if (
            isinstance(drug, dict) and "node_features" in drug
            and torch.as_tensor(drug["node_features"]).ndim == 3
            and isinstance(target, dict) and "tokens" in target
            and torch.as_tensor(target["tokens"]).ndim == 2
        ):
            return data
    if isinstance(data, dict):
        return collator([data])
    return collator(data)


def _embedding_parts(value):
    if isinstance(value, dict):
        embedding = value["embedding"]
        mask = value.get("mask")
    else:
        embedding = value
        mask = None
    if mask is None:
        mask = torch.ones(embedding.shape[:2], dtype=torch.bool, device=embedding.device)
    return embedding, mask.bool()


def _looks_like_single_or_batched_dict(value):
    return isinstance(value, dict)


def _detach_output(output):
    return {
        key: value.detach().cpu() if torch.is_tensor(value) else value
        for key, value in output.items()
    }


def _merge_outputs(outputs):
    if not outputs:
        return {"logits": torch.empty(0), "probabilities": torch.empty(0)}
    merged = {}
    for key in outputs[0]:
        values = [output.get(key) for output in outputs]
        if all(torch.is_tensor(value) for value in values):
            try:
                merged[key] = torch.cat(values, dim=0)
            except RuntimeError:
                merged[key] = values
        elif all(isinstance(value, list) for value in values):
            merged[key] = [item for value in values for item in value]
        else:
            merged[key] = values[0] if len(values) == 1 else values
    return merged


def _labels_from_data(data, collator, batch_size):
    if isinstance(data, dict):
        if "label" in data:
            return [float(data["label"])] if not torch.is_tensor(data["label"]) else data["label"]
        raise ValueError("DrugBAN evaluate() requires labels")
    labels = []
    loader = data if isinstance(data, DataLoader) else DataLoader(
        data, batch_size=batch_size or 64, shuffle=False, collate_fn=collator
    )
    for batch in loader:
        if "label" not in batch:
            raise ValueError("DrugBAN evaluate() requires labels")
        labels.append(batch["label"])
    return torch.cat(labels) if labels else torch.empty(0)


def _load_resolved_checkpoint(module, path, strict=True, map_location="cpu"):
    """Load via the generic weights helper, keeping key adaptation card-local."""

    state = load_checkpoint_state_dict(path, map_location=map_location)
    if state and all(key.startswith("model.") for key in state):
        state = {key[len("model."):]: value for key, value in state.items()}
    return module.load_state_dict(state, strict=strict)

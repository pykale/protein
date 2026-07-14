"""Small-molecule preprocessing utilities."""

from __future__ import annotations

import re

from kale_protein.registry import MODALITY_PROCESSOR_REGISTRY


SMILES_TOKEN_PATTERN = re.compile(
    r"(\[[^\]]+\]|Br?|Cl?|Si|Se|Na|Li|Mg|Ca|Al|@@?|%=\d{2}|%\d{2}|"
    r"\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|\d|[A-Za-z])"
)


@MODALITY_PROCESSOR_REGISTRY.register(("small_molecule", "smiles_tokenizer"))
class SMILESTokenizer:
    def __init__(self, input_key="smiles", vocab=None, unknown_token="<unk>", **kwargs):
        self.input_key = input_key
        self.vocab = vocab
        self.unknown_token = unknown_token

    def transform(self, sample):
        smiles = sample[self.input_key]
        tokens = SMILES_TOKEN_PATTERN.findall(smiles)
        if "".join(tokens) != smiles:
            raise ValueError(f"Could not tokenize complete SMILES string: {smiles!r}")
        output = {
            "smiles": smiles,
            "tokens": tokens,
            "attention_mask": [1] * len(tokens),
        }
        if self.vocab is not None:
            unknown_id = self.vocab.get(self.unknown_token)
            if unknown_id is None:
                raise ValueError(
                    f"SMILES vocabulary does not define unknown token {self.unknown_token!r}."
                )
            output["token_ids"] = [self.vocab.get(token, unknown_id) for token in tokens]
        return output


# This is the element ordering used by DGL-LifeSci's CanonicalAtomFeaturizer.
# Together with the remaining fields below it produces 74 atom features.
CANONICAL_ATOM_TYPES = (
    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na",
    "Ca", "Fe", "As", "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb",
    "Sn", "Ag", "Pd", "Co", "Se", "Ti", "Zn", "H", "Li", "Ge", "Cu",
    "Au", "Ni", "Cd", "In", "Mn", "Zr", "Cr", "Pt", "Hg", "Pb",
)


def _load_rdkit():
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError(
            "RDKit is required for the 'rdkit_graph' molecule processor. "
            "Install RDKit (for example, `pip install rdkit`) or pass an already "
            "preprocessed DrugBAN graph batch."
        ) from exc
    return Chem


def _one_hot(value, allowable_values):
    return [float(value == allowed) for allowed in allowable_values]


def canonical_atom_features(atom, Chem):
    """Return DGL-LifeSci-compatible canonical atom features (length 74)."""

    hybridization = Chem.rdchem.HybridizationType
    features = []
    features.extend(_one_hot(atom.GetSymbol(), CANONICAL_ATOM_TYPES))
    features.extend(_one_hot(atom.GetDegree(), range(11)))
    features.extend(_one_hot(atom.GetImplicitValence(), range(7)))
    features.append(float(atom.GetFormalCharge()))
    features.append(float(atom.GetNumRadicalElectrons()))
    features.extend(
        _one_hot(
            atom.GetHybridization(),
            (hybridization.SP, hybridization.SP2, hybridization.SP3,
             hybridization.SP3D, hybridization.SP3D2),
        )
    )
    features.append(float(atom.GetIsAromatic()))
    features.extend(_one_hot(atom.GetTotalNumHs(), range(5)))
    if len(features) != 74:
        raise RuntimeError(f"Canonical atom featurization produced {len(features)} values, expected 74")
    return features


@MODALITY_PROCESSOR_REGISTRY.register(
    ("small_molecule", "rdkit_graph"),
    aliases=(
        ("molecule/smile", "smiles"),
        ("molecule/smiles", "smiles"),
    ),
)
class RDKitGraphProcessor:
    """Convert SMILES into DrugBAN's canonical, DGL-free tensor graph."""

    def __init__(self, input_key="smiles", max_nodes=None, add_self_loops=True, **kwargs):
        self.input_key = input_key
        self.max_nodes = max_nodes
        self.add_self_loops = add_self_loops

    def transform(self, sample):
        import torch

        Chem = _load_rdkit()
        smiles = sample[self.input_key]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")

        atoms = list(mol.GetAtoms())
        num_atoms = len(atoms)
        if num_atoms == 0:
            raise ValueError(f"SMILES contains no atoms: {smiles!r}")
        if self.max_nodes is not None and num_atoms > self.max_nodes:
            raise ValueError(
                f"Molecule has {num_atoms} atoms, exceeding DrugBAN max_nodes={self.max_nodes}"
            )

        total_nodes = self.max_nodes or num_atoms
        node_features = torch.zeros((total_nodes, 75), dtype=torch.float32)
        node_features[:num_atoms, :74] = torch.tensor(
            [canonical_atom_features(atom, Chem) for atom in atoms], dtype=torch.float32
        )
        if total_nodes > num_atoms:
            node_features[num_atoms:, 74] = 1.0

        adjacency = torch.zeros((total_nodes, total_nodes), dtype=torch.float32)
        edge_sources = []
        edge_targets = []
        for bond in mol.GetBonds():
            source = int(bond.GetBeginAtomIdx())
            target = int(bond.GetEndAtomIdx())
            adjacency[source, target] = 1.0
            adjacency[target, source] = 1.0
            edge_sources.extend((source, target))
            edge_targets.extend((target, source))
        if self.add_self_loops:
            adjacency.fill_diagonal_(1.0)

        node_mask = torch.zeros(total_nodes, dtype=torch.bool)
        node_mask[:num_atoms] = True
        return {
            "smiles": smiles,
            "node_features": node_features,
            "adjacency": adjacency,
            "node_mask": node_mask,
            "edge_index": torch.tensor([edge_sources, edge_targets], dtype=torch.long),
            "atom_symbols": [atom.GetSymbol() for atom in atoms],
            "num_atoms": num_atoms,
        }

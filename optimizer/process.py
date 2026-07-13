import torch
from functools import partial
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer

def process_smiles_graph(target_smiles):
    atom_featurizer = CanonicalAtomFeaturizer()
    bond_featurizer = CanonicalBondFeaturizer(self_loop=True)

    fc = partial(smiles_to_bigraph, add_self_loop=True)
    
    molecule_graph = fc(
        smiles=target_smiles,
        node_featurizer=atom_featurizer,
        edge_featurizer=bond_featurizer
    )
    # Key: add a virtual node bit, going from 74 dims to 75 dims
    actual_node_feats = molecule_graph.ndata.pop('h')  # (num_nodes, 74)
    num_actual_nodes = actual_node_feats.shape[0]
    virtual_node_bit = torch.zeros([num_actual_nodes, 1])  # virtual node marker
    actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)  # (num_nodes, 75)
    molecule_graph.ndata['h'] = actual_node_feats
    molecule_graph = molecule_graph.add_self_loop()
    print(f"molecule graph node feature dim: {molecule_graph.ndata['h'].shape}")  # should be (num_nodes, 75)
    # Note: molecule_graph is moved to the GPU on demand inside SequenceOptimizer
    return molecule_graph

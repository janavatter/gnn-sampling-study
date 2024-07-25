from dgl.dataloading import BlockSampler
import dgl
from dgl import NID, EID
import numpy as np
import torch
from scipy import sparse
from dgl.random import choice
from dgl import backend as F


def get_laplacian(g):
    num_nodes = g.number_of_nodes()
    #adj = g.adjacency_matrix(scipy_fmt='coo')
    adj = g.adj_external(scipy_fmt='coo')
    laplacian = sparse.eye(num_nodes) - adj
    return laplacian


class FastGCNSampler(BlockSampler):
    def __init__(self, g, fanouts, prefetch_node_feats=None, prefetch_labels=None, prefetch_edge_feats=None, output_device=None, ):
        self.fanouts = fanouts
        self.laplacian = get_laplacian(g)
        self.prefetch_labels = prefetch_labels
        self.prefetch_edge_feats = prefetch_edge_feats
        self.prefetch_node_feats = prefetch_node_feats
        self.set_seed()

    def set_seed(self, random_seed=None):
        if random_seed is None:
            self.random_seed = choice(1e18, 1)
        else:
            self.random_seed = F.tensor(random_seed, F.int64)

    def get_layer_nodes(self, g, seed_nodes, fanout, p):
        #U = self.laplacian[seed_nodes, :]
        #sample the next layer's nodes based on the pre-computed probability (p).
        s_num = np.min([np.sum(p > 0), fanout]) #s_num = np.min([np.sum(p > 0), samp_num_list[d]])
        all_nodes = np.random.choice(g.num_nodes(), s_num, p=p, replace=False) #after_nodes = np.random.choice(num_nodes, s_num, p = p, replace = False)
        #     col-select the lap_matrix (U), and then divided by the sampled probability for
        #         #     unbiased-sampling. Finally, conduct row-normalization to avoid value explosion.
        #         adj = row_norm(U[: , after_nodes].multiply(1/p[after_nodes]))
        #all_nodes = U[:, selected_nodes].multiply(1/p[selected_nodes])
        #all_nodes = torch.concat((torch.tensor(selected_nodes), seed_nodes))
        #adjs += [sparse_mx_to_torch_sparse_tensor(row_normalize(adj))]
        #all_nodes = selected_nodes
        return all_nodes

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes #previous_nodes = batch_nodes
        blocks = []

        #pre-compute the sampling probability (importance) based on the global degree (lap_matrix)
        pi = np.array(np.sum(self.laplacian.multiply(self.laplacian), axis=0))[0] #pi = np.array(np.sum(lap_matrix.multiply(lap_matrix), axis=0))[0]
        p = pi / np.sum(pi) #p = pi / np.sum(pi)

        #nodes = []

        for i, fanout in enumerate(reversed(self.fanouts)): #for d in range(depth):
            all_nodes = self.get_layer_nodes(g, torch.tensor(seed_nodes), int(fanout), p)
            frontier = dgl.in_subgraph(g, seed_nodes)
            frontier = dgl.out_subgraph(frontier, all_nodes)
            eid = frontier.edata[EID]
            block = dgl.to_block(frontier, seed_nodes, include_dst_in_src=True, src_nodes=None)
            block.edata[EID] = eid
            seed_nodes = block.srcdata[NID]  # previous nodes = after nodes
            # seed_nodes = all_nodes
            blocks.insert(0, block)
            #nodes.extend(x for x in seed_nodes if x not in nodes)

        #subg = dgl.node_subgraph(g, nodes)
        #print(subg)

        self.set_seed()
        return seed_nodes, output_nodes, blocks


        # https://github.com/UCLA-DM/LADIES/blob/master/pytorch_ladies.py

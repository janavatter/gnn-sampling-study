import torch
import dgl
from dgl import NID, EID
import numpy as np
from scipy import sparse
from dgl.random import choice
from dgl import backend as F


def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sparse.diags(r_inv)

    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    if len(sparse_mx.row) == 0 and len(sparse_mx.col) == 0:
        indices = torch.LongTensor([[], []])
    else:
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return indices, values, shape


class LADIESSampler(dgl.dataloading.BlockSampler):
    def __init__(self, g, fanouts, prefetch_node_feats=None, prefetch_labels=None, prefetch_edge_feats=None, output_device=None):
        self.fanouts = fanouts
        self.laplacian = self.get_laplacian(g)
        self.prefetch_labels = prefetch_labels
        self.prefetch_edge_feats = prefetch_edge_feats
        self.prefetch_node_feats = prefetch_node_feats
        self.set_seed()

    def set_seed(self, random_seed=None):
        if random_seed is None:
            self.random_seed = choice(1e18, 1)
        else:
            self.random_seed = F.tensor(random_seed, F.int64)

    def get_laplacian(self, g):
        num_nodes = g.number_of_nodes()
        #adj = g.adjacency_matrix(scipy_fmt='coo')
        adj = g.adj_external(scipy_fmt='coo')
        laplacian = sparse.eye(num_nodes) - adj
        return laplacian

    def get_layer_nodes(self, g, seed_nodes, fanout):
        U = self.laplacian[seed_nodes, :] # U = lap_matrix[previous_nodes , :]
        pi = np.array(np.sum(U.multiply(U), axis=0))[0] # pi = np.array(np.sum(U.multiply(U), axis=0))[0]
        p = pi / np.sum(pi) #p = pi / np.sum(pi)
        s_num = np.min([np.sum(p > 0), fanout]) # s_num = np.min([np.sum(p > 0), samp_num_list[d]])
        selected_nodes = np.random.choice(g.num_nodes(), s_num, p=p, replace=False) # after_nodes = np.random.choice(num_nodes, s_num, p = p, replace = False) # num nodes == len(feat_data)
        all_nodes = torch.concat((torch.tensor(selected_nodes), seed_nodes)) # after_nodes = np.unique(np.concatenate((after_nodes, batch_nodes)))
        #adj = U[:, all_nodes].multiply(1 / p[all_nodes])

        return all_nodes #, adj

    # normlaized laplacian matrix P, batch size b, sample number n
    # randomly sample a batch of b output nodes as QL --> row selection matrix
    # for l=L to 1 do
    #       get layer-dependent laplacian matrix Q(l)P. Calculate sampling probability for each node using pi(l-1) and organize them into a random diagonal matrix S(l-1)
    #       sample n nodes in l-1 layer using p(l-1) The sampled nodes formulate Q(l-1)
    #       Reconstruct sampled laplacian matrix between sampled nodes in layer l-1 and l by ..., then normalize it ...
    # end for
    # return modified laplacian matrices and sampled node at input layer Q0

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes # previous_nodes = batch_nodes (batch nodes 1024? Len?)
        blocks = []
        for i, fanout in enumerate(reversed(self.fanouts)): # for d in range(depth) # depth==num_layers
            all_nodes = self.get_layer_nodes(g, torch.tensor(seed_nodes), int(fanout)) # all_nodes, adj
            #adjs += [sparse_mx_to_torch_sparse_tensor(row_normalize(adj))]
            frontier = dgl.in_subgraph(g, seed_nodes)
            frontier = dgl.out_subgraph(frontier, all_nodes)
            eid = frontier.edata[EID]
            block = dgl.to_block(frontier, seed_nodes, include_dst_in_src=True, src_nodes=None)
            block.edata[EID] = eid
            seed_nodes = block.srcdata[NID] # previous nodes = after nodes
            #seed_nodes = all_nodes
            blocks.insert(0, block)
            #nodes.extend(x for x in seed_nodes if x not in nodes)

        self.set_seed()
        return seed_nodes, output_nodes, blocks #, adjs


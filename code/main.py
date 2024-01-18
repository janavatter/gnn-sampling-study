import torch
import argparse
from time import perf_counter

import wandb

import dgl
from dgl.data import AsNodePredDataset
from dgl.data import CoraGraphDataset
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import FlickrDataset
from dgl.data import YelpDataset
from dgl.data import RedditDataset

from models.blocks.gcn import GCN
from models.blocks.gat import GAT
from models.subgraph.gcn_subg import GCN_subg
from models.subgraph.gat_subg import GAT_subg
from train import train
from layerwise_inference import layerwise_infer

import psutil


def train_gnn(args):

    start_time = perf_counter()
    dataset = args.data
    start_dataloading_time = perf_counter()
    if args.data in ["ogbn-products", "ogbn-arxiv", "ogbn-papers100M"]:
        dgldataset = DglNodePropPredDataset(dataset)
        graphdata = AsNodePredDataset(dgldataset)
    elif args.data == "flickr":
        dgldataset = FlickrDataset()
        graphdata = AsNodePredDataset(dgldataset)
    elif args.data == "yelp":
        dgldataset = YelpDataset()
        graphdata = AsNodePredDataset(dgldataset)
    elif args.data == "cora":
        graphdata = CoraGraphDataset()
        graphdata = AsNodePredDataset(graphdata)
    elif args.data == "reddit":
        graphdata = RedditDataset()
        graphdata = AsNodePredDataset(graphdata)
    
    g = graphdata[0]
    if args.data in ["ogbn-products", "ogbn-arxiv"]:
        g = dgl.add_self_loop(g)

    g = g.to('cuda' if args.mode == 'puregpu' else 'cpu')
    device = torch.device('cpu' if args.mode == 'cpu' else 'cuda')
    end_dataloading_time = perf_counter()

    in_size = g.ndata['feat'].shape[1]
    out_size = graphdata.num_classes
    hidden_size = args.hidden
    num_layer = args.layer
    
    gnn = {('gcn', 'neighbor'): GCN,
           ('gcn', 'labor'): GCN,
           ('gcn', 'ladies'): GCN,
           ('gcn', 'fastgcn'): GCN,
           ('gcn', 'cluster'): GCN_subg,
           ('gcn', 'saint'): GCN_subg,
           ('gcn', 'shadow'): GCN_subg,
           ('gcn', 'none'): GCN,
           ('gat', 'neighbor'): GAT,
           ('gat', 'labor'): GAT,
           ('gat', 'ladies'): GAT,
           ('gat', 'fastgcn'): GAT,
           ('gat', 'cluster'): GAT_subg,
           ('gat', 'saint'): GAT_subg,
           ('gat', 'shadow'): GAT_subg}[(args.model, args.sampler)]

    model = gnn(in_size, hidden_size, out_size, num_layer, heads=[8, 1]).to(device)

    new_test_idx = graphdata.test_idx

    train(args, device, g, graphdata, model, args.sampler)

    start_inference_time = perf_counter()

    acc, f1 = layerwise_infer(device, g, new_test_idx, model, args.data, batch_size=args.testbatchsize, out_size=out_size)

    end_time = perf_counter()

    memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)
    memory_usage_full = psutil.Process().memory_full_info()
    process_cpu_times = psutil.Process().cpu_times()
    system_cpu_times = psutil.cpu_times()
    system_cpu_stats = psutil.cpu_stats()

    wandb.log({'test-acc': acc.item(), 'test-f1': f1.item()})

    wandb.log(
        {"end-to-end time": end_time - start_time, "dataloading time": end_dataloading_time - start_dataloading_time,
         "inference time": end_time - start_inference_time, "used memory": memory_usage, "used memory full": memory_usage_full, 'process cpu times': process_cpu_times,
         'system cpu times': system_cpu_times, 'system cpu stats': system_cpu_stats})


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='mixed', choices=['cpu', 'mixed', 'puregpu'],
                        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
                             "'puregpu' for pure-GPU training.")
    parser.add_argument("--model", default="gcn", choices=["gcn", "gat"])
    parser.add_argument("--data", default="ogbn-products", choices=["ogbn-products", "ogbn-arxiv", "ogbn-papers100M", "ogbn-mag", "cora", "flickr", "yelp", "cora", "reddit"])
    parser.add_argument("--sampler", default="neighbor", choices=["neighbor", "cluster", "saint", "labor",
                                                                  "shadow", "ladies", "fastgcn"])
    parser.add_argument("--fanout", type=int, nargs='+')
    parser.add_argument("--num_parts", default=100, type=int)
    parser.add_argument("--budget", type=int, default=6000)
    parser.add_argument("--fastladies", type=int, nargs='+')
    parser.add_argument("--epoch", default=10, type=int)
    parser.add_argument("--hidden", default=16, type=int)
    parser.add_argument("--layer", default=2, type=int)
    parser.add_argument("--wandb", default="sampling-study-test")
    parser.add_argument("--cache_path", default="cluster_gcn.pkl")
    parser.add_argument("--batchsize", default=1024, type=int)
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--testbatchsize", type=int, default=4096)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        args.mode = 'cpu'

    print("the number of cpu threads: {}".format(torch.get_num_threads()))
    num_threads = torch.get_num_threads()

    config = {"architecture": args.model, "dataset": args.data, "sampler":args.sampler, "mode":args.mode, "epochs":args.epoch, "layer":args.layer, "hidden":args.hidden, "num_parts":args.num_parts, "num_threads":num_threads, "batch_size":args.batchsize, "fanout":args.fanout, "budget":args.budget, "run":args.run}
    wandb.login()
    run_name = args.model + '-' + args.data + '-' + args.sampler + '-' + args.mode + '-' + str(num_threads) + '-' + str(args.epoch) + '-' + str(args.layer) + '-' + str(args.hidden) + '-' + str(args.num_parts) + '-' + str(args.batchsize) + '-' + str(args.fanout) + '-' + str(args.budget)
    wandb.init(project=args.wandb, entity="user-name", name=run_name, config=config)

    train_gnn(args)




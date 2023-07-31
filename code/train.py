import torch
import torch.nn.functional as F
from dgl.dataloading import DataLoader, NeighborSampler, ClusterGCNSampler, SAINTSampler, ShaDowKHopSampler, MultiLayerFullNeighborSampler
from dgl.dataloading import LaborSampler
from dgl.sampling.randomwalks import random_walk
from sampler.LADIESSampler import LADIESSampler
from sampler.FastGCNSampler import FastGCNSampler
from evaluate import evaluate_blocks, evaluate_subgraph_shadow, evaluate_subgraph
import numpy as np

from time import perf_counter

import wandb


def train_blocks(model, train_dataloader, opt, sampling, args):
    block_table = wandb.Table(columns=['sampler', 'block'])
    total_loss = 0
    for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
        block_table.add_data(sampling, str(blocks))
        x = blocks[0].srcdata['feat']
        y = blocks[-1].dstdata['label']
        y_hat = model(blocks, x)
        if args.data == "yelp":
            y = y.type(torch.float64)
        else:
            y = y.type(torch.LongTensor)
        loss = F.cross_entropy(y_hat, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return it, total_loss, block_table


def train_subgraph(model, train_dataloader, opt, sampling, args):
    subgraph_table = wandb.Table(columns=['sampler', 'block'])
    total_loss = 0
    for it, subgraph in enumerate(train_dataloader):
        subgraph_table.add_data(sampling, str(subgraph))
        x = subgraph.ndata['feat']
        y = subgraph.ndata['label']
        y_hat = model(subgraph, x)
        if args.data == "yelp":
            y = y.type(torch.float64)
        else:
            y = y.type(torch.LongTensor)

        loss = F.cross_entropy(y_hat, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return it, total_loss, subgraph_table


def train_subgraph_shadow(model, train_dataloader, opt, sampling, args):
    subgraph_table = wandb.Table(columns=['sampler', 'block'])
    total_loss = 0
    for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
        subgraph_table.add_data(sampling, str(blocks))
        x = blocks.ndata['feat']
        y = blocks.ndata['label']
        y_hat = model(blocks, x)
        if args.data == "yelp":
            y = y.type(torch.float64)
        else:
            y = y.type(torch.LongTensor)
        loss = F.cross_entropy(y_hat, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return it, total_loss, subgraph_table


def train(args, device, g, dataset, model, sampling):

    train_idx = dataset.train_idx.to(device)
    val_idx = dataset.val_idx.to(device)

    start_sampling_time = perf_counter()
    fanout = args.fanout
    if sampling in ['neighbor', 'labor', 'shadow', 'ladies', 'fastgcn'] and len(args.fanout) != args.layer:
        if len(args.fanout) == 1:
            fanout = []
            for i in range(args.layer):
                fanout.append(str(args.fanout[0]))
        else:
            print('Fanout {0} and number of layers {1} do not match.'.format(args.fanout, args.layer))
            exit()
            
    if sampling == 'neighbor':
        sampler = NeighborSampler(fanout, # fanout for [layer-0, layer-1]
                                  prefetch_node_feats=['feat'],
                                  prefetch_labels=['label'])
    elif sampling == 'labor':
        sampler = LaborSampler(fanout, prefetch_node_feats=['feat'], prefetch_labels=['label'])
    elif sampling == 'cluster':
        sampler = ClusterGCNSampler(g, args.num_parts, prefetch_ndata=['feat', 'label'], cache_path=args.cache_path)
        train_idx = torch.arange(args.num_parts)
        val_idx = torch.arange(args.num_parts)
    elif sampling == 'saint':
        sampler = SAINTSampler('node', args.budget, prefetch_ndata=['feat', 'label'])
    elif sampling == 'shadow':
        sampler = ShaDowKHopSampler(fanout, prefetch_node_feats=['feat'])
    elif sampling == 'ladies':
        sampler = LADIESSampler(g, fanout, prefetch_node_feats=['feat'], prefetch_labels=['label'])
    elif sampling == 'fastgcn':
        sampler = FastGCNSampler(g, fanout, prefetch_node_feats=['feat'], prefetch_labels=['label'])
    elif sampling == "randomWalk":
        sampler = random_walk(g, [0, 1], length=10)
    elif sampling == 'none':
        sampler = MultiLayerFullNeighborSampler(2, prefetch_node_feats=['feat'], prefetch_labels=['label'])
    else:
        sampler = NeighborSampler(fanout, prefetch_node_feats=['feat'], prefetch_labels=['label'])

    eval_func = {'neighbor': evaluate_blocks,
                'labor': evaluate_blocks,
                'ladies': evaluate_blocks,
                'fastgcn': evaluate_blocks,
                'shadow': evaluate_subgraph_shadow,
                'saint': evaluate_subgraph,
                'cluster': evaluate_subgraph}[sampling]

    use_uva = (args.mode == 'mixed')

    train_dataloader = DataLoader(g, train_idx, sampler, device=device,
                                  batch_size=args.batchsize, shuffle=True,
                                  drop_last=False, num_workers=0,
                                  use_uva=use_uva)

    val_dataloader = DataLoader(g, val_idx, sampler, device=device,
                                batch_size=args.batchsize, shuffle=False,
                                drop_last=False, num_workers=0,
                                use_uva=use_uva)

    end_sampling_time = perf_counter()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    out_size = dataset.num_classes

    train_func = {'neighbor': train_blocks,
                'labor': train_blocks,
                'ladies': train_blocks,
                'fastgcn': train_blocks,
                'shadow': train_subgraph_shadow,
                'saint': train_subgraph,
                'cluster': train_subgraph}[sampling]

    start_time = perf_counter()

    current_epoch_time = 0

    # training loop
    for epoch in range(args.epoch):
        start_epoch = perf_counter()
        model.train()
        it, total_loss, table = train_func(model, train_dataloader, opt, sampling, args)
        acc = eval_func(model, g, val_dataloader, out_size)
        wandb.log({'accuracy': acc, 'loss': total_loss / (it+1)})
        end_epoch = perf_counter()

        current_epoch_time += (end_epoch-start_epoch)

    end_time = perf_counter()

    table_name = args.model + '-' + args.data + '-' + args.sampler + '-' + args.mode + '-' + str(torch.get_num_threads()) + '-' + str(args.epoch) + '-' + str(args.layer) + '-' + str(args.hidden) + '-' + str(args.num_parts) + '-' + str(args.batchsize) + '-' + str(args.fanout) + '-' + str(args.budget)
    wandb.log({"sampling time": end_sampling_time-start_sampling_time, "overall training time": end_time-start_time, "time per epoch": current_epoch_time/args.epoch, "sampling size " + table_name : table})

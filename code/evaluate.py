import torch
import torchmetrics.functional as MF


def evaluate_blocks(model, graph, dataloader, out_size):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata['feat']
            ys.append(blocks[-1].dstdata['label'])
            y_hats.append(model(blocks, x))
    return MF.accuracy(torch.cat(y_hats), torch.cat(ys), task='multiclass', num_classes=int(out_size))


def evaluate_subgraph_shadow(model, graph, dataloader, out_size):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = blocks.ndata['feat']
            ys.append(blocks.ndata['label'])
            y_hats.append(model(blocks, x))
    return MF.accuracy(torch.cat(y_hats), torch.cat(ys), task='multiclass', num_classes=int(out_size))


def evaluate_subgraph(model, graph, dataloader, out_size):
    model.eval()
    ys = []
    y_hats = []
    for it, subgraph in enumerate(dataloader):
        with torch.no_grad():
            x = subgraph.ndata['feat']
            ys.append(subgraph.ndata['label'])
            y_hats.append(model(subgraph, x))
    return MF.accuracy(torch.cat(y_hats), torch.cat(ys), task='multiclass', num_classes=int(out_size))


def evaluate_full(g, features, labels, mask, model, out_size):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        return MF.accuracy(logits, labels, task='multiclass', num_classes=int(out_size))

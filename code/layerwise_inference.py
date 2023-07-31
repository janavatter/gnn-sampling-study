import torch
import torchmetrics.functional as MF


def layerwise_infer(device, graph, nid, model, out_size, batch_size):
    model.eval()
    with torch.no_grad():
        pred = model.inference(graph, device, batch_size)
        pred = pred[nid]
        label = graph.ndata['label'][nid].to(pred.device)

        return MF.accuracy(pred, label, task='multiclass', num_classes=int(out_size))

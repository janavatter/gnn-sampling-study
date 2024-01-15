import torch
import torchmetrics.functional as MF


def layerwise_infer(device, graph, nid, model, dataset, batch_size, out_size):
    model.eval()
    with torch.no_grad():
        pred = model.inference(graph, device, batch_size) # pred in buffer_device
        pred = pred[nid]
        label = graph.ndata['label'][nid].to(pred.device)
        #return MF.accuracy(pred, label)

        if dataset == 'yelp':
            acc = MF.classification.multilabel_accuracy(pred, label, num_labels=int(out_size))
            f1 = MF.classification.multilabel_f1_score(pred, label, num_labels=int(out_size))
        else:
            acc = MF.accuracy(pred, label, task='multiclass', num_classes=int(out_size))
            f1 = MF.f1_score(pred, label, task='multiclass', num_classes=int(out_size))

    return acc, f1

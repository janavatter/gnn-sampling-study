# Artifacts and Reproducibility
## Sampling for Graph Neural Networks: Analyzing Performance Across Different Datasets 
Currently in submission.

### Artifacts
Each experiment is named `sampling-study-` followed by the experiment type and dataset. The options for type are `base` for the sampler-specific experiments, `layers` for the experiments where we vary the GNN depth, `threads` for using 1 to 32 threads and `gpu` for exploring CPU vs GPU sampling and training.
As datasets we use three datasets from the [OGB](https://ogb.stanford.edu/) collection, namely `arxiv`, `products` and `papers100M`. In addition, we use two datasets integrated in the [Deep Graph Library](https://www.dgl.ai/), [`Flickr`](https://docs.dgl.ai/generated/dgl.data.FlickrDataset.html) and [`Yelp`](https://docs.dgl.ai/generated/dgl.data.YelpDataset.html). 

All experiments were logged via [Weights & Biases (W&B)](https://wandb.ai/site). The experiment configurations are described in a [Google Docs spreadsheet](https://docs.google.com/spreadsheets/d/1EQToIxNeqmFrgdvoOTpQ9AbOb7x85ddBbNRhZPYm-nE/edit?usp=sharing). The logs from W&B are downloaded and can be found in the artifacts/wandb subdirectory.

### Reproducibility
All experiments can be reproduced by using the parameters given by the [spreadsheet](https://docs.google.com/spreadsheets/d/1EQToIxNeqmFrgdvoOTpQ9AbOb7x85ddBbNRhZPYm-nE/edit?usp=sharing). In order to log the results, one needs a Weights & Biases account. When first using W&B, you`ll need to authorize your device.

#### Preliminaries:
- You have a W&B account.
- You have installed all requirements given in `requirements.txt`.
- CUDA is installed for the GPU experiments.

##
A sample command to run an experiment is:

`python3 main.py --data ogbn-arxiv --epoch 20 --hidden 256 --sampler neighbor --layer 2 fanout 10 --wandb sampling-study-base-experiments-arxiv --batchsize 1024 --mode cpu`

This will automatically download the needed dataset (might take some time), convert it to a DGL graph and start the training process.


##
Please contact jana.vatter@tum.de for any questions regarding reproducibility.

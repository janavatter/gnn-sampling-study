# Artifacts and Reproducibility
## From Large to Small Graphs: How to Sample in Graph Neural Network Training 
Currently in submission.

### Artifacts
Each experiment is named `sampling-study-` followed by the experiment type and dataset. The options for type are `base` for the sampler-specific experiments, `layers` for the experiments where we vary the GNN depth, `gpu` for exploring CPU vs GPU sampling and training, and `bias` for investigating the bias.
As datasets we use three datasets from the [OGB](https://ogb.stanford.edu/) collection, namely `arxiv`, `products` and `papers100M`. In addition, we use ten datasets integrated in the [Deep Graph Library](https://www.dgl.ai/), [`AmazonCoBuyComputer`](https://docs.dgl.ai/generated/dgl.data.AmazonCoBuyComputerDataset.html#dgl.data.AmazonCoBuyComputerDataset), [`AmazonCoBuyPhoto`](https://docs.dgl.ai/generated/dgl.data.AmazonCoBuyPhotoDataset.html#dgl.data.AmazonCoBuyPhotoDataset), [`Citeseer`](https://docs.dgl.ai/generated/dgl.data.CiteseerGraphDataset.html#dgl.data.CiteseerGraphDataset), [`Cora`](https://docs.dgl.ai/generated/dgl.data.CoraGraphDataset.html), [`Flickr`](https://docs.dgl.ai/generated/dgl.data.FlickrDataset.html), [`CoauthorCS`](https://docs.dgl.ai/generated/dgl.data.CoauthorCSDataset.html#dgl.data.CoauthorCSDataset), [`CoauthorPhysics`](https://docs.dgl.ai/generated/dgl.data.CoauthorPhysicsDataset.html#dgl.data.CoauthorPhysicsDataset), [`Reddit`](https://docs.dgl.ai/generated/dgl.data.RedditDataset.html), [`Pubmed`](https://docs.dgl.ai/generated/dgl.data.PubmedGraphDataset.html#dgl.data.PubmedGraphDataset), and [`Yelp`](https://docs.dgl.ai/generated/dgl.data.YelpDataset.html). 
For the experiments with bias we use the `German Credit`, `Recidivism`, `Credit Defaulter`, `Pokec-n`, and `Pokec-z` datasets. They can be found in our repository under `code/dataset/`.

All experiments were logged via [Weights & Biases (W&B)](https://wandb.ai/site). The experiment configurations are described in a spreadsheet in the directory `artifacts`. The logs from W&B are downloaded and can be found in the artifacts/wandb subdirectory.

### Reproducibility
All experiments can be reproduced by using the parameters given by the spreadsheet in the directory `artifacts`. The requirements are listed in `requirements.txt`. Make sure you log in to WandB via ```wandb login``` before starting to run experiments. 
When first using W&B, you`ll need to authorize your device. For the GPU experiments, CUDA needs to be installed.

##
A sample command to run an experiment is:

`python3 main.py --data ogbn-arxiv --epoch 20 --hidden 256 --sampler neighbor --layer 2 fanout 10 --wandb sampling-study-base-experiments-arxiv --batchsize 1024 --mode cpu`

This will automatically download the needed dataset (might take some time), convert it to a DGL (Deep Graph Library) graph and start the training process.

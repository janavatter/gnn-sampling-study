# Artifacts and Reproducibility
## From Large to Small Graphs: How to Sample in Graph Neural Network Training 
Currently in submission. 

Please do not hesitate to write us (jana.vatter@tum.de) if you need any help to run or reproduce the results.

This repository contains the code, data, and artifacts of our experiments.

## Artifacts
Each experiment is named `sampling-study-` followed by the experiment type and dataset. The options for type are `base` for the sampler-specific experiments, `layers` for the experiments where we vary the GNN depth, `gpu` for exploring CPU vs GPU sampling and training, and `bias` for investigating the bias.
As datasets we use three datasets from the [OGB](https://ogb.stanford.edu/) collection, namely `arxiv`, `products` and `papers100M`. In addition, we use ten datasets integrated in the [Deep Graph Library](https://www.dgl.ai/), [`AmazonCoBuyComputer`](https://docs.dgl.ai/generated/dgl.data.AmazonCoBuyComputerDataset.html#dgl.data.AmazonCoBuyComputerDataset), [`AmazonCoBuyPhoto`](https://docs.dgl.ai/generated/dgl.data.AmazonCoBuyPhotoDataset.html#dgl.data.AmazonCoBuyPhotoDataset), [`Citeseer`](https://docs.dgl.ai/generated/dgl.data.CiteseerGraphDataset.html#dgl.data.CiteseerGraphDataset), [`Cora`](https://docs.dgl.ai/generated/dgl.data.CoraGraphDataset.html), [`Flickr`](https://docs.dgl.ai/generated/dgl.data.FlickrDataset.html), [`CoauthorCS`](https://docs.dgl.ai/generated/dgl.data.CoauthorCSDataset.html#dgl.data.CoauthorCSDataset), [`CoauthorPhysics`](https://docs.dgl.ai/generated/dgl.data.CoauthorPhysicsDataset.html#dgl.data.CoauthorPhysicsDataset), [`Reddit`](https://docs.dgl.ai/generated/dgl.data.RedditDataset.html), [`Pubmed`](https://docs.dgl.ai/generated/dgl.data.PubmedGraphDataset.html#dgl.data.PubmedGraphDataset), and [`Yelp`](https://docs.dgl.ai/generated/dgl.data.YelpDataset.html). 
For the experiments with bias we use the `German Credit`, `Recidivism`, `Credit Defaulter`, `Pokec-n`, and `Pokec-z` datasets. They can be found in our repository under `code/dataset/`.

All experiments were logged via [Weights & Biases (W&B)](https://wandb.ai/site). The experiment configurations are described in a spreadsheet in the directory `artifacts`. The logs from W&B are downloaded and can be found in the artifacts/wandb subdirectory.

## Reproducibility and Experiments
All experiments can be reproduced by using the parameters given by the spreadsheet in the directory `artifacts`. The requirements are listed in `requirements.txt`. Make sure you log in to WandB via ```wandb login``` before starting to run experiments. 
When first using W&B, you`ll need to authorize your device. For the GPU experiments, CUDA needs to be installed.

Start by cloning the repository and installing the requirements:

```bash
git clone https://github.com/janavatter/gnn-sampling-study.git
pip install -r requirements.txt
```

##
A sample command to run an experiment is:

```
python3 main.py --data ogbn-arxiv --epoch 20 --hidden 256 --sampler neighbor --layer 2 --fanout 10 --wandb sampling-study-base-experiments-arxiv --batchsize 1024 --mode cpu
```

This will automatically download the needed dataset (might take some time), convert it to a DGL (Deep Graph Library) graph and start the training process.


## GPU-based training
If using GPU-based training, make sure that CUDA is installed. You can use the following commands:
```bash
sudo apt update
sudo apt install -y build-essential
wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run
sudo sh cuda_11.7.0_515.43.04_linux.run

echo -e "\nexport PATH=/usr/local/cuda-11.7/bin:\$PATH\nexport LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc

pip install torch torchvision torchaudio

pip install  dgl -f https://data.dgl.ai/wheels/cu117/repo.html
pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html
```

## Experiment Parameters
For the training, one needs to set some parameters. We give an overview of the parameters by explaining them and describing the possible values.

1. *data*: the name of the dataset, choose from (*ogbn-arxiv*, *ogbn-products*, *ogbn-papers*, *amazoncomputer*, *amazonphoto*, *citeseer*, *cora*, *coauthorcs*, *coauthorph*, *flickr*, *pubmed*, *reddit*, *yelp*)
2. *sampler*: the sampler to use, choose from (*neighbor*, *vrgcn*, *fastgcn*, *ladies*, *labor*, *cluster*, *saint*, *shadow*)
3. *fanout*: the fanout of the sampling strategy, use with *neighbor*, *vrgcn*, *labor*, *shadow*.
4. *fastladies*: the nodes per layer, use with *fastgcn* and *ladies*.
5. *budget*: the budget for the *saint* sampler.
6. *num_parts*: the number of partitions for *cluster*
7. *epoch*: the number of epochs.
8. *hidden*: the size of the hidden dimensions.
9. *layer*: the number of layers.
10. *batchsize*: the batchsize.
11. *mode*: use *cpu* for CPU training, *mixed* for CPU-GPU training, *puregpu* for GPU training. 
12. *wandb*: the name of the wandb project.

The experiment parameters we used are described in a spreadsheet in the directory `artifacts` (`GNNSamplingStudyExperimentParameters.xlsx`).


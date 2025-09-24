# CEBRA_NLP

This project aims to use CEBRA to construct latent space with identifiability in a text embedding model's representation.

## Setting valiation
dateset=dair-ai,go_emotions,ag_news,imdb,
embedding=bert,roberta,sentence_bert,embeddinggemma,granite_embedding,jina_embedding,qwen3_embedding\

cebra : you can use any cebra models and criterions available.
visualiation setting : if the output_dim of cebra are above 3,plottings are produced usig UMAP and PCA to reduce dim to 2 for visualization.


### Defalut setting
```
defaults:
  - paths: default
  - dataset: dair-ai                  # dataset
  - embedding: bert                   # language embedding model
  - cebra: offset1-model-lr           # Cebra model 
  - consistency_check: default        #consistecy check default: consistecy check enabeled, check between 5runs
  - hpt: my_sweep
  - _self_

device: ${oc.env:DEVICE, cpu}

reproducibility:         
  seed: 7
  deterministic: false
  cudnn_benchmark: false

hydra:
  run:

    dir: results/${dataset.name}/${now:%Y-%m-%d_%H-%M-%S}

evaluation:
  test_size: 0.2
  random_state: 42
  knn_neighbors: 5
  enable_plots: false                 # defaults setting do not provide any visualization
```

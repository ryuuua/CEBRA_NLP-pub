# CEBRA_NLP

This project aims to use CEBRA to construct latent space with identifiability in a text embedding model's representation.

## Setting
dateset=dair-ai,go_emotions,ag_news,imdb,
embedding=bert,roberta,sentence_bert,embeddinggemma,granite_embedding,jina_embedding,qwen3_embedding

cebra 
cebra=
cebra.output_dim=
cebra
### Defalut setting
'''
defaults:
  - paths: default
  - dataset: dair-ai
  - embedding: bert
  - cebra: offset1-model-lr
  - consistency_check: default
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
  enable_plots: false

wandb:
  project: "CEBRA_NLP_Experiment-${dataset.name}"
  run_name: "default_run"
  entity: null

ddp:
  world_size: 2
  rank: 0
  local_rank: 0
'''

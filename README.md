# Heter-GCN
This repository provides a reference implementation of the proposed model in the following paper:

> Yongji Wu, Defu Lian, Shuowei Jin, Enhong Chen. Graph Convolutional Networks on User Mobility Heterogeneous Graphs for Social Relationship Inference. The 27th International Joint Conference on Artificial Intelligence (IJCAI 2019), Macao, China, August, 2019

Our implementation is based on: 
Thomas N. Kipf, Max Welling, [Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)](https://github.com/tkipf/gcn)

## Requirments
- TensorFlow >= 1.10

## Usage
```
python setup.py install
python heter_gcn/unsupervised_train.py
python heter_gcn/semi_sup_train.py
```
where `unsupervised_train.py` is for unsupervised training, while `semi_sup_train.py` is for semi-supervised training when part of the  ground truth social network is available.

The Austin dataset (Gowalla) is included in the `dataset` folder as an example. The provided partial social graph is a partial social network used for semi-supervised training which contains 30% of social pairs.

## Citing
Please cite our paper if you find it useful in your research:
```
@inproceedings{WLJC19,
author = {Yongji Wu and Defu Lian and Shuowei Jin and Enhong Chen},
title = {Graph Convolutional Networks on User Mobility Heterogeneous Graphsfor Social Relationship Inference},
booktitle = {Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence (IJCAI-19)},
year = {2019}
}
```
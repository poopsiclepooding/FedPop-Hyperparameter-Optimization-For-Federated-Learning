# FedPop-Hyperparameter-Optimization-For-Federated-Learning

## Introduction

This research project focuses on improving Federated Learning (FL) by optimizing hyperparameters across distributed clients. The goal is to adapt the hyperparameters dynamically on each client by applying a gradient descent technique to the hyperparameters, as suggested in the paper "Gradient Descent: The Ultimate Optimizer." The premise is that each client will learn the best hyperparameters suited for its own data distribution, and averaging the hyperparameters from all clients will lead to the best overall configuration.

## Overview

Federated Learning allows multiple decentralized clients to collaboratively train machine learning models without sharing data. One of the key challenges in FL is selecting the appropriate hyperparameters, as different clients may have different data distributions. In this project, we implement hyperparameter optimization through gradient descent on each client, with the aim of optimizing the hyperparameters individually for each client. By averaging the learned hyperparameters from all clients, we hope to find the optimal configuration for the federated system as a whole.

This project includes experiments on both IID (Independent and Identically Distributed) and non-IID datasets. The results from these experiments aim to demonstrate how this approach can improve the performance of federated learning in diverse scenarios.

## Installation

1. Git clone this repository
```bash
git clone https://github.com/yourusername/Hyperparameter-Optimization-in-Federated-Learning.git
cd Hyperparameter-Optimization-in-Federated-Learning
```

2. Create a python environment and install the dependencies from `requirments.txt`.
```bash
pip install -r requirements.txt
```

3. Run the jupyter notebooks

## FIles

The code is divided into 4 jupyter notebook as follows:

1. `iid_hyperparameter_optimization.ipynb`: This notebook demonstrates hyperparameter optimization using gradient descent on IID datasets, where data distributions across clients are the same.

2. `non_iid_hyperparameter_optimization.ipynb`: This notebook demonstrates hyperparameter optimization on non-IID datasets, where data distributions vary across clients.

3. `iid_no_hyperparameter_optimization.ipynb`: This notebook contains the federated learning setup without hyperparameter optimization, using IID datasets for comparison.

4. `non_iid_no_hyperparameter_optimization.ipynb`: This notebook trains a federated learning model without hyperparameter optimization, using non-IID datasets for comparison.

List of hyperparameters to test the model on should be put as list variables on the last code block in the ipynb files. 
Results for all these are stored in a folder called `results`.

# Basic neural network library supported auto-differentiation

## Motivation

Most deep learning courses aim to teach math behind the network, architecture and their applications.

However, seldom course talk about how to implement the deep learning library and start every from scratch.

Wish to implement this kind of library and learn how to design a better numerical software.

## Target

Based on `autograd` project, build a similar library that user simply define the function, and this lib can automatically calculate this differentiation form of given function.

Build the computational graph when function is called, calculate the backward propogation with respect to variable.

Provide a benchmark compared to tensorflow and pytorch.

If time allowed, provide a simple multi-layer perceptron (neural network) interface, criterion, optimizer, datasets and dataloader like pytorch.


## Goal Feature

- Auto-diff
- Multi-layer perceptron

- Optimizer:
   - Stochastic Gradient Descent
- Criterion:
  - Mean Square Error
  - Cross Entropy
- datasets
- dataloader


## TODO

- linear regression with more than one variable
- neural network
- classifier
- benchmark
- unit test
- cross entropy criterion
- documents

## Reference

- autograd
- pytorch
- tensorflow
- caffe
- caffe2

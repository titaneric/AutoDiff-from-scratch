# Basic neural network library supported auto-differentiation

## Motivation

Most deep learning courses aim to teach math behind the network, architecture and their applications.

However, seldom course talk about how to implement and design the deep learning library and start everything from scratch.

Wish to implement this kind of library and learn how and why the priors (TensorFlow and PyTorch etc.) design their work during the development of final project.

## Target

Based on [Autograd](https://github.com/HIPS/autograd) project, build a similar library that user simply define the function, and this lib can automatically calculate this differentiation form of given function.

Build the computational graph when function is called, calculate the backward propogation with respect to variable or placeholder (in tensorflow term).

Provide a benchmark compared to TensorFlow and PyTorch.

If time allowed, provide a simple multi-layer perceptron (neural network) interface, criterion, optimizer, datasets and dataloader like the priors.

## Project Link

[AutoDiff-from-scratch](https://github.com/titaneric/AutoDiff-from-scratch)

## TODO

- benchmark
- documents

## Reference

### Source code

- [Autograd](https://github.com/HIPS/autograd)
- [Autodidact](https://github.com/mattjj/autodidact)
- [PyTorch](https://github.com/pytorch/pytorch)
- [TensorFlow](https://github.com/tensorflow/tensorflow)
- [Caffe](https://github.com/BVLC/caffe)
- [Caffe2](https://github.com/pytorch/pytorch/tree/master/caffe2)

### Lecture

- [Backpropagation, Toronto CSC321](http://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/slides/lec06.pdf)
- [Automatic Differentiation, Toronto CSC321](http://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/slides/lec10.pdf)
- [Backpropagation, Stanford CS224N](https://www.youtube.com/watch?v=yLYHDSv-288&list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z&index=5&t=2177s)
- [Introduction to Neural Networks, Stanford CS231N](https://www.youtube.com/watch?v=d14TUNcbn1k&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=4)
- [Backpropagation: Find Partial Derivatives, MIT 18.065](https://www.youtube.com/watch?v=lZrIPRnoGQQ&list=PLUl4u3cNGP63oMNUHXqIUcrkS2PivhN3k&index=30&t=0s)

### Documents

- [Jax official](https://jax.readthedocs.io/en/latest/index.html)
- [Phd Thesis by Dougal Maclaurin (one of Autograd author)](https://dougalmaclaurin.com/phd-thesis.pdf)

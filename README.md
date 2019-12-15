# Basic neural network library supported auto-differentiation

## Introduction

This is the very simple neural network library that supporting auto-differtiation. After referenced works from priors (see *Reference*), I provide the implementation which is simple enough that can tackle the real-world problem (regression and classification).

## Documents

### Forward-prop and backward-prop

Source:

- `autodiff/autodiff/core.py`
- `autodiff/autodiff/diff.py`

We init the graph in the forward prop and call the wrapped operation (See *Wrapped operation and VJP*). I am too lazy to use the DiGraph in networkx as the computational graph but only use litte features (build graph and topological sort).

Note that unlike `autograd` and `jax`, I can only calculate the first order derivative of the given function.

### Wrapped operation and VJP

Source:

- `autodiff/autodiff/numpy_grad/wrapper.py`
- `autodiff/autodiff/numpy_grad/vjps.py`

In the forward-prop, we construct the computational graph (See *Computational graph*) with the wrapped numpy function in the wrapper module.

In the backward-prop, we need to reference the VJP form of the given operation, we register the VJPs in the very begining. I only support little function but it's enough to do the more higher implementation (See *High level usage*).

### Computational graph

Source: 

- `autodiff/autodiff/graph/tracer.py` (wrapped operation and graph building)
- `autodiff/autodiff/graph/node.py` (nodes classes)
- `autodiff/autodiff/graph/manager.py` (Context manager for graph)

Notice that we don't have to worry about the how the node has been built (order of execution), because the python interpreter know it, for example

```python
ad.add(ad.Variable(x), ad.Variable(y)
```

The interpreter know that we need to construct the Variable of x and then Variable of y first, after that we do the operation of add.

By fully leverage this feature, we can push variable nodes into stack and pop them from stack in the primitive operation and connect them with an edge.

Note that in the current implementation, I use `id(x)` to identify different nodes in the graph (which is bad but simple). Due to small integer may have the same address which can cause program no longer work as expected, there are lots of TODO in the future.

### High level usage

Source:

- `autodiff/nn/layer.py` (Module and Layer classes)
- `autodiff/nn/criterion.py` (Criterion for different problems)
- `autodiff/nn/optimizer.py` (Optimizers)

In the high-level implementation, Fully imitate the syntax from pytorch but it only support the multi-layer perceptron with little weird syntax in the current time, as shown below.

```python
class SimpleModel(Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.linear1 = Linear(num_features, 5)
        self.linear2 = Linear(5, num_classes)

    def forward(self, x):
        x = self.linear1(x, bridge=False)
        x = self.linear2(x)
        return x
```

This model want to solve the classification problem with two layers, notice that in the `forward` method, first layer MUST assign bridge to `False` to generate a new `Placeholder` for x.

### Example

Source:

- `examples\*`

We cover the real-world problem including regression and classification. I use MSE criterior for regression and cross-entropy loss for classification.

### Test

Source:

- `autodiff\tests\*`

So far, we cover the test for get value from function and it's derivatives.

## Reference

### Implementation

- Core idea from [Autograd](https://github.com/HIPS/autograd)
- Test procedure and test cases from [JAX](https://github.com/google/jax)
- Variable, Placeholder and Constant syntax from [TensorFlow](https://github.com/tensorflow/tensorflow)
- Layer syntax and cross entropy from [PyTorch](https://github.com/pytorch/pytorch)


### Source code

- [Autograd](https://github.com/HIPS/autograd)
- [Autodidact](https://github.com/mattjj/autodidact)
- [PyTorch](https://github.com/pytorch/pytorch)
- [TensorFlow](https://github.com/tensorflow/tensorflow)
- [Caffe](https://github.com/BVLC/caffe)
- [Caffe2](https://github.com/pytorch/pytorch/tree/master/caffe2)
- [JAX](https://github.com/google/jax)

### Lecture

- [Backpropagation, Toronto CSC321](http://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/slides/lec06.pdf)
- [Automatic Differentiation, Toronto CSC321](http://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/slides/lec10.pdf)
- [Backpropagation, Stanford CS224N](https://www.youtube.com/watch?v=yLYHDSv-288&list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z&index=5&t=2177s)
- [Introduction to Neural Networks, Stanford CS231N](https://www.youtube.com/watch?v=d14TUNcbn1k&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=4)
- [Backpropagation: Find Partial Derivatives, MIT 18.065](https://www.youtube.com/watch?v=lZrIPRnoGQQ&list=PLUl4u3cNGP63oMNUHXqIUcrkS2PivhN3k&index=30&t=0s)

### Documents

- [Jax official](https://jax.readthedocs.io/en/latest/index.html)
- [Phd Thesis by Dougal Maclaurin (one of Autograd author)](https://dougalmaclaurin.com/phd-thesis.pdf)

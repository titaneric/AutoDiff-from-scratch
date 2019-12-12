#pylint: disable=no-member
import numpy as _np
import networkx as nx

import autodiff as ad


class Module:
    def __init__(self):
        self._graph = nx.DiGraph()
        self._parameters = []

    def __call__(self, *args):
        #pylint: disable=assignment-from-none
        ad.register_graph(self._graph)
        result = self.forward(*args)
        # only set parameters at the first forward pass
        if not self._parameters:
            ad.set_parameters(self._parameters, self._graph)
            ad.set_forwarded(self._graph)

        return result

    def forward(self, *args):
        return

    def zero_grad(self):
        ad.zero_grad(self._graph)

    def backward(self, upstream):
        return ad.backward_prop(upstream)

    def parameters(self):
        return self._parameters


class Layer:
    def __init__(self):
        pass

    def __call__(self):
        pass


class Linear(Layer):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = _np.random.random((in_features, out_features))

    def __call__(self, features, bridge=True):
        features = features if bridge else ad.Placeholder(features)
        return ad.dot(features, ad.Variable(self.weight))

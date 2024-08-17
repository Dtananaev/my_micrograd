#
# Author: Denis Tananaev
# Date: 17.08.2024
#
import math
import numpy as np
import matplotlib.pyplot as plt
import random
from graphviz import Digraph


def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})
    
    for n in nodes:
        dot.node(name=str(id(n)), label = "{%s | data %.4f | grad %.4f }" % (n.label, n.data,  n.grad), shape='record')

        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))
    
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot


class Value:
    """Basic class for values."""

    def __init__(self, data,  _children=(), _op="",  label="",):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        """Print statement."""
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        """Internally call a.__add__(b)."""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            """For case a+a we have += gradients"""
            self.grad  += 1.0  * out.grad
            other.grad += 1.0  * out.grad
        out._backward = _backward
        return out


    def __radd__(self, other):
        """For case 1 + Value(1.0)"""
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data * other.data, (self, other), "*")
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad 
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        """For case 1  *  Value(1.0)"""
        return self * other

    def tanh(self):
        n = self.data
        t = (math.exp(2.0*n) -1.0 ) / (math.exp(2.0*n) + 1.0)
        out =  Value(t, (self, ), "tanh")

        def _backward():
            self.grad += (1.0 -  out.data ** 2) * out.grad

        out._backward = _backward
        return out
    

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), "exp")

        def _backward():
            self.grad  += out.data * out.grad
        out._backward = _backward
        return out


    def __truediv__(self, other):
        return self * other ** -1


    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int, float powers for now."
        out = Value(self.data ** other, (self, ), f"**{other}")

        def _backward():
            self.grad += other * self.data ** (other -1) * out.grad

        out._backward = _backward
        return out

    def __neg__(self,):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def backward(self):
        # Topological graph sort
        topo = []
        visited =set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


class Neuron:

    def __init__(self, nin):

        self.w  = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))


    def parameters(self):
        return self.w + [self.b]

    def __call__(self, x):


        act = sum([wi*xi for wi, xi in zip(self.w, x)], self.b)
        out = act.tanh()
        return out

class Layer:

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) ==1 else outs


class MLP:

    def __init__(self, nin, nouts):

        sz = [nin] + nouts

        self.layers  = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]


    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __call__(self, x):

        for layer in self.layers:
            x = layer(x)
        return x

if __name__ == "__main__":
    # First simple test
    a = Value(2.0, label="a")
    b = Value(-3.0, label="b")
    c = Value(10.0, label="c")
    e = a*b; e.label="e"
    d = e +c; d.label="d"
    f = Value(-2.0, label="f")
    L = d * f; L.label= "L"
    dot = draw_dot(L)
    dot.render("graph", format="png")
    
    # Second neuron
    x1 = Value(2.0, label="x1")
    x2 = Value(0.0, label="x2")
    w1 = Value(-3.0, label="w1")
    w2 = Value(1.0, label="w2")
    b= Value(6.7, label="b")

    x1w1 = x1*w1; x1w1.label="x1w1"
    x2w2 = x2*w2; x2w2.label="x2w2"
    x1w1x2w2 = x1w1+x2w2; x1w1x2w2.label="x1w1 + x2w2"
    n = x1w1x2w2+b; n.label = "n"
    o = n.tanh(); o.label="o"
    o.backward()
    dot = draw_dot(o)
    dot.render("neuron", format="png")


    # Third

    a = Value(2.0, label="a")
    b = a + a; b.label="b"
    b.backward()
    dot = draw_dot(b)
    dot.render("a_a", format="png")


    b = 1 + a

    # Forth neuron
    x1 = Value(2.0, label="x1")
    x2 = Value(0.0, label="x2")
    w1 = Value(-3.0, label="w1")
    w2 = Value(1.0, label="w2")
    b= Value(6.7, label="b")

    x1w1 = x1*w1; x1w1.label="x1w1"
    x2w2 = x2*w2; x2w2.label="x2w2"
    x1w1x2w2 = x1w1+x2w2; x1w1x2w2.label="x1w1 + x2w2"
    n = x1w1x2w2+b; n.label = "n"

    e = (2*n).exp()
    o = (e-1)/ (e+1); o.label="o"
    o.backward()
    dot = draw_dot(o)
    dot.render("neuron_exp", format="png")


    # Neuron
    x = [2.0, 3.0]
    n = Neuron(2)


    # Layer
    n = Layer(2, 3)

    # MLP

    x = [1.0, 2.0, -1.0]

    n = MLP(3, [4,4,1])
    f = n(x)
    print(f"MLP {n(x)}")
    dot = draw_dot(f)
    dot.render("MLP", format="png")


    # DATASET

    xs = [
        [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 0.5],
    [1.0, 1.0, -1.0]

    ]

    ys = [1.0, -1.0, -1.0, 1.0]

    # Training loop
    for k in range(20):
        ypred = [n(x) for x in xs]

        loss = sum([(ypd -ygt)**2 for ygt, ypd in zip(ys, ypred)])

        # Make zero grad
        for p in n.parameters():
            p.grad = 0.0
        # Compute backward
        loss.backward()

        # Update
        for p in n.parameters():
            p.data += -0.05 * p.grad
        
        print(f"k {k}, loss {loss.data}")
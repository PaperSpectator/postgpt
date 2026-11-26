import random
import math

class Val:
    """Scalar autograd node. Tracks computation graph for backpropagation."""
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = float(data)
        self.grad = 0.0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Val) else Val(other)
        return Val(self.data + other.data, (self, other), (1.0, 1.0))

    def __mul__(self, other):
        other = other if isinstance(other, Val) else Val(other)
        return Val(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        return Val(self.data ** other, (self,), (other * self.data ** (other - 1),))

    def log(self):
        d = max(self.data, 1e-12)
        return Val(math.log(d), (self,), (1.0 / d,))

    def exp(self):
        e = math.exp(min(self.data, 80))
        return Val(e, (self,), (e,))

    def relu(self):
        return Val(max(0, self.data), (self,), (float(self.data > 0),))

    def tanh(self):
        t = math.tanh(self.data)
        return Val(t, (self,), (1.0 - t * t,))

    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return (-self) + other
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * (other if isinstance(other, Val) else Val(other)) ** -1
    def __rtruediv__(self, other): return Val(other) * self ** -1

    def backward(self):
        topo = []
        visited = set()
        def build(v):
            if id(v) not in visited:
                visited.add(id(v))
                for c in v._children:
                    build(c)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            for child, lg in zip(v._children, v._local_grads):
                child.grad += lg * v.grad

def _randn(std=0.02):
    return random.gauss(0, std)

def _matrix(rows, cols, std=0.02):
    return [[Val(_randn(std)) for _ in range(cols)] for _ in range(rows)]

def _zeros(rows, cols):
    return [[Val(0.0) for _ in range(cols)] for _ in range(rows)]

def linear(x, w):
    """Matrix-vector multiply: w @ x. w is [out, in], x is [in]."""
    return [sum(wi * xi for wi, xi in zip(row, x)) for row in w]

def softmax(logits):
    """Numerically stable softmax over list of Val."""
    max_val = max(v.data for v in logits)
    exps = [(v - max_val).exp() for v in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_float(logits):
    """Softmax over plain floats."""
    max_val = max(logits)
    exps = [math.exp(min(v - max_val, 80)) for v in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    """RMS normalization."""
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + Val(1e-5)) ** -0.5
    return [xi * scale for xi in x]
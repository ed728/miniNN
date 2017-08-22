import numpy as np

class Layer:
    """
    This represents a single layer of our Network.
    """
    def __init__(self, size):
        self.size = size

    def forward(self, X):
        raise NotImplementedError()

    def backward(self):
        raise NotImplementedError()

    def getParams(self):
        raise NotImplementedError()

    def setParams(self, params):
        raise NotImplementedError()

class FullyConnectedLayer(Layer):
    """
    A fully connected layer.
    """

    def __init__(self, size, activation):
        """
        dim(weights): dim(INPUT) x HIDDEN LAYERS
        dim(bias): HIDDEN LAYERS
        activation: an object providing both an activation function and its derivative.
        """
        Layer.__init__(self, size)
        self._W = np.array(([]))
        self._b = np.array(([]))
        self._act = activation
        self._z = None

    def setWeight(self, w):
        """
        This is determined by the size of this layer and the one below it.
        """
        self._W = w

    def setBias(self, b):
        self._b = b

    def forward(self, X):
        """
        Does the forward pass for an input X.
        dim(X): BATCH SIZE x dim(INPUT)
        """
        self._X = X # For backprop later on.
        self._z = np.dot(X, self._W) + self._b
        a = self._act.a(self._z)
        return a


    def backward(self, delta_W_next):
        """
        delta(N) = delta(N+1) * W(N+1)^T * a'(z(N))
        dC/dW(N) = delta(N) * a(z(N-1))

        We provide the delta multiplied by the weight of the next layer in the network.
        """
        delta = delta_W_next * self._act.a_prime(self._z)
        delta_W = np.dot(delta, self._W.T)
        grad_w = np.dot(self._X.T, delta)
        grad_b = np.array(([np.sum(delta, axis=0)]))
        return grad_w, grad_b, delta_W

    def get_total_size(self):
        return self._W.size + self._b.size

    def get_params(self):
        return np.concatenate((self._W.ravel(), self._b.ravel()))

    def set_params(self, params):
        """
        Should be used to update parameters after they have been
        originally set.
        """
        self._W = np.reshape(params[0:self._W.size], self._W.shape)
        end = self._W.size + self._b.size
        self._b = np.reshape(params[self._W.size:end], self._b.shape)

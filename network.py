import numpy as np

class NeuralNetwork:
    def __init__(self, inputSize):
        self._layers = list()
        self.inputSize = inputSize

    def add_layer(self, l):
        # Compute weights size based on previous layer.
        if len(self._layers) > 0:
            size_x = self._layers[-1].size
        else:
            # If no previous layer, use the input size.
            size_x = self.inputSize
        size_y = l.size
        # Set random weights initially.
        l.setWeight(np.random.randn(size_x, size_y))
        l.setBias(np.random.randn(size_y))

        self._layers.append(l)

    def cost(self, y, yh):
        """
        Currently support only 1 cost function for simplicity.
        """
        return 0.5*(sum((y - yh)**2))

    def costX(self, X, Y):
        return self.cost(Y, self.run(X))

    def run(self, X):
        """
        Runs the network and results the final output as a numpy array.
        """
        a = X
        for l in self._layers:
            a = l.forward(a)
        return a

    def get_params(self):
        params = np.array(([]))
        for l in self._layers:
            params = np.concatenate((params, l.get_params()))
        return params

    def set_params(self, params):
        idx = 0
        # Let's do it the Pythonic way and let it blow up if the user
        # passes in something stupid.
        for l in self._layers:
            sz = l.get_total_size()
            l.set_params(params[idx:idx+sz])
            idx += sz

    def compute_gradients(self, X, Y):
        Y_hat = self.run(X)
        # We use Y_hat prime as our first delta_W
        delta_W = -(Y - Y_hat)
        grads = []
        for l in reversed(self._layers):
            grad_W, grad_b, delta_W = l.backward(delta_W)
            grads.append((grad_W, grad_b))
        return list(reversed(grads))

    def compute_gradients_unpacked(self, X, Y):
        grads = self.compute_gradients(X, Y)
        up = np.array(([]))
        for g in grads:
            up = np.concatenate((up, g[0].ravel(), g[1].ravel()))
        return up

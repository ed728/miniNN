import numpy as np

class Activation:
    def a(Z):
        """
        a
        """
        raise NotImplementedError()

    def a_prime(Z):
        """
        a'
        """
        raise NotImplementedError()

class Sigmoid(Activation):
    def a(self, Z):
        """
        s(x) = 1/(1 + e^(-x))
        """
        return 1/(1 + np.exp(-Z))

    def a_prime(self, Z):
        """
        s'(x) = s(x) * (1 - s(x))
        """
        az = self.a(Z)
        return az * (1 - az)

import numpy as np

from network import NeuralNetwork
from layer import FullyConnectedLayer
from activation import Sigmoid
from loader import load_NN

def compute_numerical_gradient(N, X, y):
    """
    Most important function...
    """
    paramsInitial = N.get_params()
    numgrad = np.zeros(paramsInitial.shape)
    perturb = np.zeros(paramsInitial.shape)
    e = 1e-4

    for p in range(len(paramsInitial)):
        perturb[p] = e
        N.set_params(paramsInitial+perturb)
        loss2 = N.costX(X, y)

        N.set_params(paramsInitial-perturb)
        loss1 = N.costX(X, y)

        numgrad[p] = (loss2-loss1)/(2*e)

        perturb[p] = 0

    N.set_params(paramsInitial)

    return numgrad

def sanity_check():
    NN = NeuralNetwork(2)
    NN.add_layer(FullyConnectedLayer(3, Sigmoid()))
    NN.add_layer(FullyConnectedLayer(1, Sigmoid()))
    X = np.array(([3, 5], [5, 1], [10, 2]), dtype=float)
    Y = np.array(([75], [82], [93]), dtype=float)
    X = X/np.amax(X, axis=0)
    Y = Y/100.
    print("Numeric check:")
    print(NN.compute_gradients_unpacked(X, Y))
    print(compute_numerical_gradient(NN, X, Y))

if __name__=="__main__":
    sanity_check()

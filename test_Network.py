# ---------------
# Xavier Robbins
# CS3450-031 
# Lab 9 : Training FMNIST
# Apr 25 2023
# ---------------

from unittest import TestCase
import layers
import network
import numpy as np
import torch
import unittest

class TestNetwork(TestCase):
    """
    Tests the Network class
    """
    def setUp(self):
        self.x = layers.Input((2, 1), train=False)
        # self.x.set()

        self.W = layers.Input((3, 2), train=True)
        self.W.set(torch.tensor([[-1, 7], [4, 8], [2, 3]], dtype=torch.float64))

        self.b = layers.Input((3, 1), train=True)
        self.b.set(torch.tensor([[2], [3], [3]], dtype=torch.float64))

        self.linear_1 = layers.Linear(self.x, self.W, self.b) # u

        self.relu = layers.ReLU(self.linear_1) # h

        self.M = layers.Input((2, 3), train=True)
        self.M.set(torch.tensor([[-2, 1, 0], [3, -1, -1]], dtype=torch.float64))

        self.c = layers.Input((2, 1), train=True)
        self.c.set(torch.tensor([[1], [0]], dtype=torch.float64))

        self.linear_2 = layers.Linear(self.relu, self.M, self.c) # v/o

        self.y = layers.Input((2, 1), train=False)
        self.y.set(torch.tensor([[2], [-1]], dtype=torch.float64))

        self.mse = layers.MSELoss(self.y, self.linear_2) # L

        self.regularize_1 = layers.Regularization(self.W, 1) # s1
        self.regularize_2 = layers.Regularization(self.M, 1) # s2

        self.sum_regularize = layers.Sum(self.regularize_1, self.regularize_2) # S

        self.output = layers.Sum(self.mse, self.sum_regularize) # J

        self.network = network.Network()
        self.network.set_input(self.x)
        self.network.add(self.W)
        self.network.add(self.b)
        self.network.add(self.linear_1)
        self.network.add(self.relu)
        self.network.add(self.M)
        self.network.add(self.c)
        self.network.add(self.linear_2)
        self.network.add(self.y)
        self.network.add(self.mse)
        self.network.add(self.regularize_1)
        self.network.add(self.regularize_2)
        self.network.add(self.sum_regularize)
        self.network.set_output(self.output)

    
    def test_forward(self):
        self.network.forward(torch.tensor([[1], [2]], dtype=torch.float64))
        np.testing.assert_allclose(self.network.output.numpy(), np.array([263]))

    
    def test_backward(self):
        self.network.forward(torch.tensor([[1], [2]], dtype=torch.float64))
        self.network.backward()
        np.testing.assert_allclose(self.output.grad.numpy(), np.array([1]))
        np.testing.assert_allclose(self.sum_regularize.grad.numpy(), np.array([1]))
        np.testing.assert_allclose(self.regularize_2.grad.numpy(), np.array([1])) 
        # np.testing.assert_allclose(self.M.grad.numpy(), np.array([[-4, 2, 0], [6, -2, -2]]))
        np.testing.assert_allclose(self.regularize_1.grad.numpy(), np.array([1]))
        # np.testing.assert_allclose(self.W.grad.numpy(), np.array([[-2, 14], [8, 16], [4, 6]]))
        np.testing.assert_allclose(self.mse.grad.numpy(), np.array([1]))
        np.testing.assert_allclose(self.linear_2.grad.numpy(), np.array([[8], [-12]]))
        # np.testing.assert_allclose(self.M.grad.numpy(), np.array([[120, 184, 88], [-180, -276, -132]]))
        np.testing.assert_allclose(self.relu.grad.numpy(), np.array([[-52], [20], [12]]))
        np.testing.assert_allclose(self.linear_1.grad.numpy(), np.array([[-52], [20], [12]]))
        # np.testing.assert_allclose(self.W.grad.numpy(), np.array([[-52, -104], [20, 40], [12, 24]]))
        np.testing.assert_allclose(self.W.grad.numpy(), np.array([[-54, -90], [28, 56], [16, 30]]))
        np.testing.assert_allclose(self.b.grad.numpy(), np.array([[-52], [20], [12]]))
        np.testing.assert_allclose(self.M.grad.numpy(), np.array([[116, 186, 88], [-174, -278, -134]]))
        np.testing.assert_allclose(self.c.grad.numpy(), np.array([[8], [-12]]))


    def test_step(self):
        self.network.forward(torch.tensor([[1], [2]], dtype=torch.float64))
        self.network.backward()
        self.network.step(0.1)

        np.testing.assert_allclose(self.x.output.numpy(), np.array([[1], [2]]))
        np.testing.assert_allclose(self.W.output.numpy(), np.array([[4.4, 16], [1.2, 2.4], [0.4, 0]]))
        np.testing.assert_allclose(self.b.output.numpy(), np.array([[7.2], [1], [1.8]]))
        np.testing.assert_allclose(self.M.output.numpy(), np.array([[-13.6, -17.6, -8.8], [20.4, 26.8, 12.4]]))
        np.testing.assert_allclose(self.c.output.numpy(), np.array([[0.2], [1.2]]))
        np.testing.assert_allclose(self.y.output.numpy(), np.array([[2], [-1]]))


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)

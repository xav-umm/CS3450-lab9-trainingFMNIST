# ---------------
# Xavier Robbins
# CS3450-031 
# Lab 6 : Implementing Forward, Deriving All Unit Tests
# Apr 25 2023
# ---------------

from unittest import TestCase
import layers
import numpy as np
import torch
import unittest

class TestMSE(TestCase):
    """
    Tests the MSE Loss Layer class
    """
    def setUp(self):
        self.o = layers.Input((3, 1), train=True) 
        self.o.set(torch.tensor([[12], [3], [2]], dtype=torch.float64))

        self.y = layers.Input((3, 1), train=False)
        self.y.set(torch.tensor([[13], [5], [0]], dtype=torch.float64))

        self.mse = layers.MSELoss(self.o, self.y)

    def test_forward(self):
        self.mse.forward()
        np.testing.assert_allclose(self.mse.output.numpy(), np.array([3]))

    def test_backward(self):
        self.mse.forward()
        self.mse.accumulate_grad(torch.tensor([1]))
        self.mse.backward()

        np.testing.assert_allclose(self.o.grad.numpy(), np.array([[-1], [-2], [2]]))
        np.testing.assert_allclose(self.y.grad.numpy(), np.array([[-1], [-2], [2]]))

    def test_step(self):
        self.mse.forward()
        self.mse.accumulate_grad(torch.tensor([1]))
        self.mse.backward()
        
        self.o.step(0.1)
        self.y.step(0.1)

        np.testing.assert_allclose(self.o.output.numpy(), np.array([[12.1], [3.2], [1.8]]))
        np.testing.assert_allclose(self.y.output.numpy(), np.array([[13.1], [5.2], [-0.2]]))


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
# ---------------
# Xavier Robbins
# CS3450-031 
# Lab 9 : Training FMNIST
# Apr 25 2023
# ---------------

from unittest import TestCase
import layers
import numpy as np
import torch
import unittest

class TestLinear(TestCase):
    """
    Tests the Linear Layer class
    """
    def setUp(self):
        self.W = layers.Input((3, 2), train=True)
        self.W.set(torch.tensor([[1, 3], [3, 4], [6, 2]], dtype=torch.float64))

        self.x = layers.Input((2, 1), train=False)
        self.x.set(torch.tensor([[2], [4]], dtype=torch.float64))

        self.b = layers.Input((3, 1), train=True)
        self.b.set(torch.tensor([[1], [2], [1]], dtype=torch.float64))

        self.linear = layers.Linear(self.x, self.W, self.b)

    
    def test_forward(self):
        self.linear.forward()
        np.testing.assert_allclose(self.linear.output.numpy(), np.array([[15], [24], [21]]))

    
    def test_backward(self):
        self.linear.forward()
        self.linear.accumulate_grad(torch.ones((3, 1), dtype=torch.float64))
        self.linear.backward()

        np.testing.assert_allclose(self.W.grad.numpy(), np.array([[2, 4], [2, 4], [2, 4]]))
        np.testing.assert_allclose(self.x.grad.numpy(), np.array([[10], [9]]))
        np.testing.assert_allclose(self.b.grad.numpy(), np.ones((3, 1)))

    
    def test_step(self):
        self.linear.forward()
        self.linear.accumulate_grad(torch.ones((3, 1), dtype=torch.float64))
        self.linear.backward()
        self.W.step(0.1)
        self.b.step(0.1)
        self.x.step(0.1)

        np.testing.assert_allclose(self.W.output.numpy(), np.array([[0.8, 2.6], [2.8, 3.6], [5.8, 1.6]]))
        np.testing.assert_allclose(self.x.output.numpy(), np.array([[1], [3.1]]))
        np.testing.assert_allclose(self.b.output.numpy(), np.array([[0.9], [1.9], [0.9]]))


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)

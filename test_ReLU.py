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

class TestReLU(TestCase):
    """
    Tests the ReLU Layer class
    """
    def setUp(self):
        self.u = layers.Input((3, 1), train=True)
        self.u.set(torch.tensor([[1], [3], [-1]], dtype=torch.float64))
        
        self.relu = layers.ReLU(self.u)

    
    def test_forward(self):
        self.relu.forward()
        np.testing.assert_allclose(self.relu.output.numpy(), np.array([[1], [3], [0]]))

    
    def test_backward(self):
        self.relu.forward()
        self.relu.accumulate_grad(torch.tensor([[0.1], [0.1], [0.1]], dtype=torch.float64))
        self.relu.backward()

        np.testing.assert_allclose(self.u.grad.numpy(), np.array([[0.1], [0.1], [0]]))

    
    def test_step(self):
        self.relu.forward()
        self.relu.accumulate_grad(torch.tensor([[0.1], [0.1], [0.1]], dtype=torch.float64))
        self.relu.backward()

        self.u.step(1)

        np.testing.assert_allclose(self.u.output.numpy(), np.array([[0.9], [2.9], [-1]]))


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False) 

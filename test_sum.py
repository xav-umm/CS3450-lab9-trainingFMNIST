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

class TestSum(TestCase):
    """
    Tests the summation network layer.
    """
    def setUp(self):
        self.a = layers.Input((2,1), train=True)
        self.a.set(torch.tensor([[3],[5]], dtype=torch.float64))
        self.b = layers.Input((2,1), train=True)
        self.b.set(torch.tensor([[1],[2]], dtype=torch.float64))
        self.sum = layers.Sum(self.a, self.b)

    
    def test_forward(self):
        self.sum.forward()
        np.testing.assert_allclose(self.sum.output.numpy(), np.array([[4],[7]]))

    
    def test_backward(self):
        self.sum.forward()
        self.sum.accumulate_grad(torch.ones(2,1))
        self.sum.backward()

        np.testing.assert_allclose(self.a.grad.numpy(), np.ones((2,1)))
        np.testing.assert_allclose(self.b.grad.numpy(), np.ones((2,1)))

    
    def test_step(self):
        self.sum.forward()
        self.sum.accumulate_grad(np.ones((2,1)))
        self.sum.backward()
        self.a.step(step_size = 0.1)
        self.b.step(step_size = 0.1)

        np.testing.assert_allclose(self.a.output.numpy(), np.array([[2.9],[4.9]]))
        np.testing.assert_allclose(self.b.output.numpy(), np.array([[0.9],[1.9]]))


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)

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

class TestRegularize(TestCase):
    """
    Tests the Regularization Layer class
    """
    def setUp(self):
        self.W = layers.Input((3, 2), train=True)
        self.W.set(torch.tensor([[1, 3], [3, 4], [6, 2]], dtype=torch.float64))

        self.decay = 0.1

        self.regularize = layers.Regularization(self.W, self.decay)

    def test_forward(self):
        self.regularize.forward()
        np.testing.assert_allclose(self.regularize.output.numpy(), np.array([7.5]))

    def test_backward(self):
        self.regularize.forward()
        self.regularize.accumulate_grad(torch.tensor([5]))
        self.regularize.backward()

        np.testing.assert_allclose(self.W.grad.numpy(), np.array([[1, 3], [3, 4], [6, 2]]))

    def test_step(self): 
        self.regularize.forward()
        self.regularize.accumulate_grad(torch.tensor([5]))
        self.regularize.backward()

        self.W.step(0.1)

        np.testing.assert_allclose(self.W.output.numpy(), np.array([[0.9, 2.7], [2.7, 3.6], [5.4, 1.8]]))


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
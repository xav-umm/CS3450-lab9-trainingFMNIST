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

class TestSoftmax(TestCase):
    
    def setUp(self):
        self.v = layers.Input((3, 1), train=True)
        self.v.set(torch.tensor([[3], [35], [16]], dtype=torch.float64))

        self.y = layers.Input((3, 1), train=False)
        self.y.set(torch.tensor([[1], [0], [0]], dtype=torch.float64))

        self.epsilon = 1e-7

        self.softmax = layers.Softmax(self.v, self.y, self.epsilon)

    def test_forward(self):
        self.softmax.forward()
        np.testing.assert_allclose(self.softmax.classifications.numpy(), np.array([[1.266417e-14], [0.999999], [5.602796e-09]]), rtol=1e-5)
        np.testing.assert_allclose(self.softmax.output.numpy(), np.array([7.751]), rtol=1e-4)

    def test_backward(self):
        self.softmax.forward()
        self.softmax.accumulate_grad(torch.tensor([1]))
        self.softmax.backward()

        np.testing.assert_allclose(self.v.grad.numpy(), np.array([[-1], [0.999999], [5.602796e-09]]), rtol=1e-5)
        

    def test_step(self):
        self.softmax.forward()
        self.softmax.accumulate_grad(torch.tensor([1]))
        self.softmax.backward()
        self.v.step(1) 

        np.testing.assert_allclose(self.v.output.numpy(), np.array([[4], [34.0000001], [15.9999995]]))
 

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
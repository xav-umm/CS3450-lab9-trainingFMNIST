# ---------------
# Xavier Robbins
# CS3450-031 
# Lab 6 : Implementing Forward, Deriving All Unit Tests
# Apr 25 2023
# ---------------

import torch
from numpy import newaxis as np_newaxis

# TODO: Please be sure to read the comments in the main lab and think about your design before
# you begin to implement this part of the lab.

# Layers in this file are arranged in roughly the order they
# would appear in a network.


class Layer: 
    def __init__(self, output_shape):
        """
        Initializes a layer and sets output shape to given shape
        :output_shape: Output shape of layer 
        """
        # For Broadcasting Check if 2D
        self.output_shape = output_shape if type(output_shape) is tuple else (output_shape, 1)
        self.grad = torch.zeros(self.output_shape, dtype=torch.float64)

    def accumulate_grad(self, accum):
        """
        This method should accumulate its grad attribute with the value provided.
        """
        # accum = torch.sum(accum, dim=1)
        assert accum.shape == self.grad.shape, f'Shape to accumulate ({accum.shape}) does not match layer grad shape ({self.grad.shape})'
        self.grad += accum 

    def clear_grad(self):
        """
        Sets the grad to zeros in the shape of output shape.
        """
        self.grad = torch.zeros(self.output_shape, dtype=torch.float64)

    def step(self, step_size):
        """
        Most tensors do nothing during a step so we simply do nothing in the default case.
        """
        pass

class Input(Layer):
    def __init__(self, output_shape, train=True):
        """
        Initializes an input layer
        :output_shape: Shape of the input in the Input layer
        :train: Marks the values of the input as trainable parameters if true, use false if 
        input values are not to be trained
        """
        Layer.__init__(self, output_shape) 
        self.train = train
        self.output = torch.zeros(output_shape)

    def set(self, output):
        """
        Sets the values of the output of the layer to the given output
        :param output: The output to set, as a torch tensor. Raise an error if this output's size 
                        would change.
        """
        assert self.output_shape == output.shape, f'Provided output shape ({output.shape}) does not match expected shape ({self.output_shape})'
        self.output = output

    def randomize(self):
        """
        Sets the output of this input layer to random values sampled from the standard normal
        distribution.
        """
        self.output = torch.rand(self.output_shape, dtype=torch.float64) * 0.1

    def forward(self):
        """
        In forward input values do not perform any operations, simply pass
        """
        # Does an input layer do anything during the forward pass? No
        pass

    def backward(self):
        """
        This method does nothing as the Input layer should have already received its output
        gradient from the previous layer(s) before this method was called.
        """
        pass

    def step(self, step_size):
        """
        This method should have a precondition that the gradients have already been computed
        for a given batch.

        It should perform one step of stochastic gradient descent, updating the weights of
        this layer's output based on the gradients that were computed and a learning rate.
        """
        if(self.train == True):
            self.output -= (self.grad * step_size)
        

class Linear(Layer):
    def __init__(self, x: Layer, W: Layer, b: Layer):
        """
        Initializes a Linear layer
        :x: A layer containing the samples or input values to this layer
        :W: A layer containing the weights to the layer
        :b: A layer containing the bias for the layer
        """
        Layer.__init__(self, (W.output_shape[0], x.output_shape[1]))
        self.x = x
        self.W = W
        self.b = b
        self.output = self.output_shape

    def forward(self):
        """
        Performs the forward pass calculation for a linear layer
        """
        self.output = torch.matmul(self.W.output, self.x.output) + self.b.output

    def backward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """
        self.dx = torch.matmul(torch.transpose(self.W.output, 0, 1), self.grad)
        self.x.accumulate_grad(self.dx)

        self.dW = self.grad @ torch.transpose(self.x.output, 0, 1)
        self.W.accumulate_grad(self.dW)

        self.db = torch.sum(self.grad, dim=1, keepdim=True) * 1
        self.b.accumulate_grad(self.db)


class ReLU(Layer):
    def __init__(self, u: Layer):
        """
        Initializes a ReLU layer
        :u: A layer containing the input to use as this layer 
        """
        Layer.__init__(self, u.output_shape) 
        self.u = u

    def forward(self):
        """
        Performs the forward pass ReLU operation on the previously set layers output
        """
        self.output = torch.clamp(self.u.output, min=0)

    def backward(self):
        """
        This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """
        relu_prime = torch.ones(self.output_shape) * (self.output > 0)
        self.du = self.grad * relu_prime
        self.u.accumulate_grad(self.du)


class MSELoss(Layer):
    """
    This is a good loss function for regression problems.

    It implements the MSE norm of the inputs.
    """
    def __init__(self, y: Layer, o: Layer):
        """
        Initializes an MSE Loss layer.
        :y: Ground truth label for output as an input layer
        :o: Layer whose output is the input to this layer 
        """
        Layer.__init__(self, (1,)) # TODO: Pass along any arguments to the parent's initializer here.
        # print('Y OUTPUT SHAPE:', y.output_shape)
        # print('MSE OUTPUT SHAPE: ', self.output_shape)
        self.y = y
        self.o = o
        self.output = torch.zeros(self.output_shape)

    def forward(self):
        """
        Perform MSE forward pass operations on using the set initialized layer outputs 
        """
        self.output = torch.mean((self.o.output - self.y.output) ** 2)

    def backward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """
        self.do = self.grad * (self.o.output - self.y.output)
        self.o.accumulate_grad(self.do)

        self.dy = self.grad * (self.o.output - self.y.output)
        self.y.accumulate_grad(self.dy)


class Regularization(Layer):
    def __init__(self, W: Layer, decay):
        """
        Initializes a frobenius norm regularization layer 
        :W: Layer contatining weights to perform regularization over
        :decay: Floating point regularization constant 
        """
        Layer.__init__(self, (1,)) # TODO: Pass along any arguments to the parent's initializer here.
        self.W = W
        self.decay = decay
        self.output = torch.zeros(self.output_shape)

    def forward(self):
        """
        Performs the frobenius norm over the set layer of weights and multiplies that by the 
        regularization constant 
        """
        self.output = self.decay * torch.sum(self.W.output ** 2)

    def backward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """
        self.dW = self.W.output * (2 * self.decay * self.grad)
        self.W.accumulate_grad(self.dW)


class Softmax(Layer):
    """
    This layer is an unusual layer.  It combines the Softmax activation and the cross-
    entropy loss into a single layer.

    The reason we do this is because of how the backpropagation equations are derived.
    It is actually rather challenging to separate the derivatives of the softmax from
    the derivatives of the cross-entropy loss.

    So this layer simply computes the derivatives for both the softmax and the cross-entropy
    at the same time.

    But at the same time, it has two outputs: The loss, used for backpropagation, and
    the classifications, used at runtime when training the network.

    Create a self.classifications property that contains the classification output,
    and use self.output for the loss output.

    See https://www.d2l.ai/chapter_linear-networks/softmax-regression.html#loss-function
    in our textbook.

    Another unusual thing about this layer is that it does NOT compute the gradients in y.
    We don't need these gradients for this lab, and usually care about them in real applications,
    but it is an inconsistency from the rest of the lab.
    """
    def __init__(self, v: Layer, y: Layer, epsilon):
        """
        Initializes a Softmax/Cross Entropy combination Layer 
        :v: Layer containing output vector to use in softmax calculations
        :y: Ground truth labels to use in loss calculations
        :epsilon: Regularization constant to use in cross entropy calculation
        """
        Layer.__init__(self, (1,)) # TODO: Pass along any arguments to the parent's initializer here.
        self.v = v
        self.y = y
        self.epsilon = epsilon
        self.classifications = torch.zeros(v.output_shape)
        self.output = torch.zeros(self.output_shape)

    def forward(self):
        """
        Performs forward pass operations to calculate the softmax classifications and cross entropy loss
        given a layer containing a given output vector. 
        """
        m = torch.max(self.v.output, dim=0).values
        # print(f'm : {m}')
        e = torch.exp(self.v.output - m)
        # print(f'e : {e.shape}')
        # print(f'divisor : {torch.sum(e, dim=0).shape}')
        self.classifications = e / torch.sum(e, dim=0) # Sum along columns
        # print(self.classifications)
        # print(self.y.output)
        # print(f'o : {torch.sum(self.classifications, dim=0)}') # Should get row of 1s of shape batch size

        # print(f'o : {self.classifications}')
        log_o = torch.log2(self.classifications + self.epsilon)
        # print(f'log_o : {log_o}')
        intermediate = torch.sum((-1 * (self.y.output * log_o)), dim=1)
        # print(f'intermediate : {intermediate}')
        self.output = torch.mean(intermediate) # Cross-entropy loss

    def backward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: Set this layer's output based on the outputs of the layers that feed into it.
        TODO: This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """
        self.dv = (self.y.output - self.classifications) * -1 * self.grad
        self.v.accumulate_grad(self.dv)


class Sum(Layer):
    def __init__(self, s1: Layer, s2: Layer):
        """
        Initializes a sum layer that performs scalar addition 
        :s1: Layer containing output as argument one for addition
        :s2: Layer containing output as argument two for addition
        """
        assert s1.output_shape == s2.output_shape, f'Shape of s1 ({s1.output_shape}) does not match shape of s2 ({s2.output_shape})!'
        Layer.__init__(self, s1.output_shape) # TODO: Pass along any arguments to the parent's initializer here.
        self.s1 = s1
        self.s2 = s2
               

    def forward(self):
        """
        Performs addition using the two set layers in initialization
        """
        self.output = self.s1.output + self.s2.output


    def backward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """
        assert self.grad != torch.zeros(self.grad.shape), f'Must have grads accumulated!!!'
        self.ds1 = self.grad * 1
        self.s1.accumulate_grad(self.ds1)

        self.ds2 = self.grad * 1
        self.s2.accumulate_grad(self.ds2)

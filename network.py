# ---------------
# Xavier Robbins
# CS3450-031 
# Lab 6 : Implementing Forward, Deriving All Unit Tests
# Apr 25 2023
# ---------------

import torch

class Network:
    def __init__(self):
        """
        Intializes a network object
        """
        self.layers = []
        self.input = None
        self.output = None

    def add(self, layer): 
        """
        Adds a new layer to the network.

        Sublayers can *only* be added after their inputs have been added.
        (In other words, the DAG of the graph must be flattened and added in order from input to output)
        :param layer: The sublayer to be added
        """
        # TODO: Implement this method.
        assert len(self.layers) >= 1, f'Must set an input layer!'
        self.layers.append(layer)

    def set_input(self, input):
        """
        Sets the input layer of the network 
        :param input: The sublayer that represents the signal input (e.g., the image to be classified)
        """
        # TODO: Delete or implement this method. (Implementing this method is optional, but do not
        # leave it as a stub.)
        assert len(self.layers) == 0, f'Cannot add input to existing network!'
        self.input_layer = input
        self.layers.append(input)

    def set_output(self, output):
        """
        Sets the output layer of the network 
        :param output: SubLayer that produces the useful output (e.g., clasification decisions) as its output.
        """
        # TODO: Delete or implement this method. (Implementing this method is optional, but do not
        # leave it as a stub.)
        #
        # This becomes messier when your output is the variable o from the middle of the Softmax
        # layer -- I used try/catch on accessing the layer.classifications variable.
        # when trying to access read the output layer's variable -- and that ended up being in a
        # different method than this one.
        assert len(self.layers) >= 1, f'Must set an input layer!'
        self.output_layer = output
        self.layers.append(output)

    def forward(self, input_vals):
        """
        Compute the output of the network in the forward direction, working through the gradient
        tape forward

        :param input: A torch tensor that will serve as the input for this forward pass
        :return: A torch tensor with useful output (e.g., the softmax decisions)
        """
        self.input_layer.set(input_vals)
        # print(input)
        # print(self.input_layer.output)
        for layer in self.layers: 
            layer.forward()
            # print(layer.output)

        try: 
            self.output = self.output_layer.classification
        except AttributeError: 
            self.output = self.output_layer.output

        return self.output

    def backward(self):
        """
        Compute the gradient of the output of all layers through backpropagation backward through the 
        gradient tape.
        """
        self.dj = torch.ones(self.output.shape, dtype=torch.float64)
        self.dj = torch.unsqueeze(self.dj, 0)
        self.output_layer.accumulate_grad(self.dj)
        for layer in reversed(self.layers):
            layer.backward()

        # for layer in reversed(self.layers):
        #     print('--------------------')
        #     print(layer)
        #     print(layer.output)
        #     print(layer.grad)


    def step(self, step_size):
        """
        Perform one step of the stochastic gradient descent algorithm
        based on the gradients that were previously computed by backward, updating all learnable parameters 

        :param step_size: Learning rate to scale gradients for update
        """
        for layer in self.layers:
            layer.step(step_size)

        # for layer in reversed(self.layers):
        #     print('--------------------')
        #     print(layer)
        #     print(layer.output)
        #     print(layer.grad)

    def clear_grad(self):
        for layer in self.layers:
            layer.clear_grad()

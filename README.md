# CS3450-lab9-trainingFMNIST
Final lab for CS3450 - Deep Learning at MSOE (Spring Trimester 2023)

This final lab culminates the trimester of Deep Learning at MSOE, using object-oriented neural network components implemented by hand with PyTorch in previous labs from the trimester. 
This lab involves training and evaluating a custom, object-oriented implementation of a nueral network on the well known Fashion MNIST and CIFAR-10 datasets. 

**File Descriptions :** 
- client_cifar10.py : Python file utilizing `network.py` and `layers.py` to build and train a fully connected neural network on the well known CIFAR-10 dataset. This file was implemented on MSOE's supercomputer ROSIE, which held the CIFAR-10 dataset and allowed for training in a GPU environment. Included in this file are some methods for creating simple testing datasets (`create_linear_training_data()`, `create_folded_training_data()`, and `create_square()`) to verify the functionality of the network before training on the CIFAR-10 dataset. The CIFAR-10 dataset is loaded from ROSIE's file storage and flattened for training with the `load_dataset_flattened()` method. The network is then built and trained in the main method of the file.
  
- client_fmnist.py : Python file utilizing `network.py` and `layers.py` to build and train a fully connected neural network on the well known Fashion-MNIST dataset. This file was implemented on MSOE's supercomputer ROSIE, which held the Fashion-MNIST dataset and allowed for training in a GPU environment. This file follows a very similar structure to `client_cifar10.py` with the same three methods for creating simple testing datasets included as well as the `load_dataset_flattened()` method, which is again used to load the Fashion-MNIST dataset from ROSIE's file storage. The network is then built and trained in the main method of the file.
  
- network.py : Python class implementing a neural network. This class is expects layers from `layers.py` to be added using the methods `set_output()`, `set_input()`, and `add()`. Then, the `forward()`, `backward()`, and `step()` functions can be used to compute the forward pass, backward pass, and perform updates to the weights based on the gradients calculated through calling `backward()`.
  
- layers.py : Python file acting as a library of implemented neural network layers. Layer types implemented include : Input, Linear (Fully Connected), MSE Loss, Regularization, Softmax/Cross-Entropy Loss, and Summation
  
- test_<layer_type>.py : Simple tests for each layer class within layers.py to ensure the layer implementation correctly computes the expected results and behaves properly during backpropagation.

- lab9-conlusion.pdf : This file outlines the training results of the networks built and trained in `client_cifar10.py` and `client_fmnist.py`. Included in these results are 5 training runs of the network on each dataset, the best run of the 5 is highlighted and the training and testing plots are shown for this run to examine for overfitting or other unexpected behavior in the metrics during training. (**NOTE :** The datasets are not split into proper Train/Val/Test sets as is best practice when training deep learning algorithms. This was done because these networks were not ultimately used further in a production environment, and course requirements mandated a simple Test/Train split to place more emphasis on the implementation of the network itself and achieving a working training loop.) This document culminates in a short reflection on the course and the process of implementing these networks. 

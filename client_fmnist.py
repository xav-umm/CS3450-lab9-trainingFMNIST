import torch
import numpy as np # For loading cached datasets
import matplotlib.pyplot as plt
# import torchvision # For loading initial datasets
#                    # Commented out because Spring 2023 this is failing to load
#                    # in the conda-cs3450 environment
import time
import warnings
import os.path

import network
import layers

# warnings.filterwarnings('ignore')  # If you see warnings that you know you can ignore, it can be useful to enable this.

EPOCHS = 1
# For simple regression problem
TRAINING_POINTS = 1000

# For fashion-MNIST and similar problems
DATA_ROOT = '/data/cs3450/data/'
FASHION_MNIST_TRAINING = '/data/cs3450/data/fashion_mnist_flattened_training.npz'
FASHION_MNIST_TESTING = '/data/cs3450/data/fashion_mnist_flattened_testing.npz'
CIFAR10_TRAINING = '/data/cs3450/data/cifar10_flattened_training.npz'
CIFAR10_TESTING = '/data/cs3450/data/cifar10_flattened_testing.npz'
CIFAR100_TRAINING = '/data/cs3450/data/cifar100_flattened_training.npz'
CIFAR100_TESTING = '/data/cs3450/data/cifar100_flattened_testing.npz'

# With this block, we don't need to set device=DEVICE for every tensor.
torch.set_default_dtype(torch.float32)
if torch.cuda.is_available():
     torch.cuda.set_device(0)
     torch.set_default_tensor_type(torch.cuda.FloatTensor)
     print("Running on the GPU")
else:
     print("Running on the CPU")

def create_linear_training_data():
    """
    This method simply rotates points in a 2D space.
    Be sure to use L2 regression in the place of the final softmax layer before testing on this
    data!
    :return: (x,y) the dataset. x is a torch tensor where columns are training samples and
             y is a torch tensor where columns are one-hot labels for the training sample.
    """
    x = torch.randn((2, TRAINING_POINTS))
    x1 = x[0:1, :].clone()
    x2 = x[1:2, :]
    y = torch.cat((-x2, x1), axis=0)
    return x, y


def create_folded_training_data():
    """
    This method introduces a single non-linear fold into the sort of data created by create_linear_training_data. Be sure to REMOVE the final softmax layer before testing on this data!
    Be sure to use MSE in the place of the final softmax layer before testing on this
    data!
    :return: (x,y) the dataset. x is a torch tensor where columns are training samples and
             y is a torch tensor where columns are one-hot labels for the training sample.
    """
    x = torch.randn((2, TRAINING_POINTS))
    x1 = x[0:1, :].clone()
    x2 = x[1:2, :]
    x2 *= 2 * ((x2 > 0).float() - 0.5)
    y = torch.cat((-x2, x1), axis=0)
    return x, y


def create_square():
    """
    This is a square example in which the challenge is to determine
    if the points are inside or outside of a point in 2d space.
    insideness is true if the points are inside the square.
    :return: (points, insideness) the dataset. points is a 2xN array of points and insideness is true if the point is inside the square.
    """
    win_x = [2,2,3,3]
    win_y = [1,2,2,1]
    win = torch.tensor([win_x,win_y],dtype=torch.float32)
    win_rot = torch.cat((win[:,1:],win[:,0:1]),axis=1)
    t = win_rot - win # edges tangent along side of poly
    rotation = torch.tensor([[0, 1],[-1,0]],dtype=torch.float32)
    normal = rotation @ t # normal vectors to each side of poly
        # torch.matmul(rotation,t) # Same thing

    points = torch.rand((2,2000),dtype = torch.float32)
    points = 4*points

    vectors = points[:,np.newaxis,:] - win[:,:,np.newaxis] # reshape to fill origin
    insideness = (normal[:,:,np.newaxis] * vectors).sum(axis=0)
    insideness = insideness.T
    insideness = insideness > 0
    insideness = insideness.all(axis=1)
    return points, insideness


def load_dataset_flattened(train=True,dataset='Fashion-MNIST',download=False):
    """
    :param train: True for training, False for testing
    :param dataset: 'Fashion-MNIST', 'CIFAR-10', or 'CIFAR-100'
    :param download: True to download. Keep to false afterwords to avoid unneeded downloads.
    :return: (x,y) the dataset. x is a torch tensor where columns are training samples and
             y is a torch tensor where columns are one-hot labels for the training sample.
    """
    if dataset == 'Fashion-MNIST':
        if train:
            path = FASHION_MNIST_TRAINING
        else:
            path = FASHION_MNIST_TESTING
        num_labels = 10
    elif dataset == 'CIFAR-10':
        if train:
            path = CIFAR10_TRAINING
        else:
            path = CIFAR10_TESTING
        num_labels = 10
    elif dataset == 'CIFAR-100':
        if train:
            path = CIFAR100_TRAINING
        else:
            path = CIFAR100_TESTING
        num_labels = 100
    else:
        raise ValueError('Unknown dataset: '+str(dataset))

    if os.path.isfile(path):
        print('Loading cached flattened data for',dataset,'training' if train else 'testing')
        data = np.load(path)
        x = torch.tensor(data['x'],dtype=torch.float32)
        y = torch.tensor(data['y'],dtype=torch.float32)
        pass
    else:
        class ToTorch(object):
            """Like ToTensor, only redefined by us for 'historical reasons'"""

            def __call__(self, pic):
                return torchvision.transforms.functional.to_tensor(pic)

        if dataset == 'Fashion-MNIST':
            data = torchvision.datasets.FashionMNIST(
                root=DATA_ROOT, train=train, transform=ToTorch(), download=download)
        elif dataset == 'CIFAR-10':
            data = torchvision.datasets.CIFAR10(
                root=DATA_ROOT, train=train, transform=ToTorch(), download=download)
        elif dataset == 'CIFAR-100':
            data = torchvision.datasets.CIFAR100(
                root=DATA_ROOT, train=train, transform=ToTorch(), download=download)
        else:
            raise ValueError('This code should be unreachable because of a previous check.')
        x = torch.zeros((len(data[0][0].flatten()), len(data)),dtype=torch.float32)
        for index, image in enumerate(data):
            x[:, index] = data[index][0].flatten()
        labels = torch.tensor([sample[1] for sample in data])
        y = torch.zeros((num_labels, len(labels)), dtype=torch.float32)
        y[labels, torch.arange(len(labels))] = 1
        np.savez(path, x=x.numpy(), y=y.numpy())
    return x, y

class Timer(object):
    def __init__(self, name=None, filename=None):
        self.name = name
        self.filename = filename

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        message = 'Elapsed: %.2f seconds' % (time.time() - self.tstart)
        if self.name:
            message = '[%s] ' % self.name + message
        print(message)
        if self.filename:
            with open(self.filename,'a') as file:
                print(str(datetime.datetime.now())+": ",message,file=file)

# Training loop -- fashion-MNIST
# def main_linear():
if __name__ == '__main__':
    # Once you start this code, comment out the method name and uncomment the
    # "if __name__ == '__main__' line above to make this a main block.
    # The code in this section should NOT be in a helper method.
    #
    # In particular, your client code that uses your classes to stitch together a specific network 
    # _should_ be here, and not in a helper method.  This will give you access
    # to the layers of your network for debugging purposes.

    with Timer('Total time'):
        dataset = 'Fashion-MNIST'
        x_train, y_train = load_dataset_flattened(train=True, dataset=dataset, download=True)

        epochs = 5
        batch_size = 4
        learning_rate = 0.001
        regularize = 0.0001
        epsilon = 1e-7

        W = layers.Input((256, 784), train=True)
        W.randomize()
        b = layers.Input((256, 1), train=True)
        b.randomize()
        x = layers.Input((784, batch_size), train=False)
        # x input set when network.forward() called 

        linear_1 = layers.Linear(x, W, b) # u
        relu = layers.ReLU(linear_1) # h

        M = layers.Input((10, 256), train=True)
        M.randomize()
        c = layers.Input((10, 1), train=True)
        c.randomize()

        linear_2 = layers.Linear(relu, M, c)

        y = layers.Input((10, batch_size), train=False)
        softmax = layers.Softmax(linear_2, y, epsilon)

        regularize_1 = layers.Regularization(W, regularize)
        regularize_2 = layers.Regularization(M, regularize)
        sum_reg = layers.Sum(regularize_1, regularize_2)
        output = layers.Sum(softmax, sum_reg)

        network = network.Network()
        network.set_input(x)
        network.add(W)
        network.add(b)
        network.add(linear_1)
        network.add(relu)
        network.add(M)
        network.add(c)
        network.add(linear_2)
        network.add(y)
        network.add(softmax)
        network.add(regularize_1)
        network.add(regularize_2)
        network.add(sum_reg)
        network.set_output(output)

        accuracies = []
        test_accuracies = []
        losses = [] # These are loss for the last batch in the epoch not over the whole training set
        test_losses = []
        
        # TODO: Train your network.
        with Timer('Training time'):
            for e in range(1, epochs + 1):
                num_correct = 0
                print(f':::::::::::::::EPOCH {e}:::::::::::::::')
                for i in range(0, x_train.shape[1] - (batch_size - 1), batch_size):
                    y.set(y_train[:, i:i+batch_size])
                    network.forward(x_train[:, i:i+batch_size].double())

                    o_predict = torch.argmax(softmax.classifications, dim=0)
                    y_correct = torch.argmax(y.output, dim=0)
                    num_correct += torch.sum((y_correct == o_predict).long())  

                    network.backward()
                    network.step(learning_rate)
                    network.clear_grad()

                    if i % 5000 == 0:
                        print(f':::TRAIN PROGRESS:::\nJ:{network.output}\nCross Entropy Loss:{softmax.output}\nCurr. Acc:{num_correct/float(i)}\n')

                accuracy = num_correct / float(x_train.shape[1]) 
                # Track loss and acc for plotting
                accuracies.append(accuracy)
                losses.append(softmax.output)

                print(f':::::EPOCH COMPLETE:::::\nAccuracy:{accuracy}\nLoss:{softmax.output}\n')

                #### TEST SET ####
                # Compute the error on this test data:
                x_test, y_test = load_dataset_flattened(train=False, dataset=dataset, download=True)

                num_test_correct = 0
                for i in range(0, x_test.shape[1] - (batch_size - 1), batch_size):
                    y.set(y_test[:, i:i+batch_size])
                    network.forward(x_test[:, i:i+batch_size].double())
                    o_test_predict_loc = torch.argmax(softmax.classifications, dim=0)
                    y_test_correct_loc = torch.argmax(y.output, dim=0)
                    num_test_correct += torch.sum((y_test_correct_loc == o_test_predict_loc).long())  

                test_accuracy = num_test_correct / float(x_test.shape[1])
                test_accuracies.append(test_accuracy)
                test_losses.append(softmax.output)

                print(f':::TEST SET RESULTS:::\nAccuracy:{test_accuracy}\nLoss:{softmax.output}\n\n')
    
        # Report on GPU memory used for this script:
        peak_bytes_allocated = torch.cuda.memory_stats()['active_bytes.all.peak']
        print(f"Peak GPU memory allocated: {peak_bytes_allocated} Bytes")

    pass # You may wish to keep this line as a point to place a debugging breakpoint.

import matplotlib.pyplot as plt

plt.plot(range(len(accuracies)), accuracies)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Fashion-MNIST Training Accuracy')

plt.plot(range(len(losses)), losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Fashion-MNIST Training Loss')

plt.plot(range(len(test_accuracies)), test_accuracies)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Fashion-MNIST Testing Accuracy')

plt.plot(range(len(test_losses)), test_losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Fashion-MNIST Testing Loss')

'''
    Write a model for gesture classification.
'''
import torch.nn as nn

class CNN_Model(nn.Module):
    def __init__(self, input_size):
        super(CNN_Model, self).__init__()

        # Convolutional layers effectively enable the model to recognize more and more complex patterns.
            # For example, if the first convolutional layer enables the model to recognize lines and curves, the
            # second convolutional layer enables the model to recognize shapes made from those lines and curves such as
            # squares and circles. etc.

        # Kernels effectively determine how much of the input data you are determining. The stride determines the
            # speed at which the kernel is being examined and padding restores a certain amount of the information
            # that is inevitably lost from the kernel extraction
        self.conv1 = nn.Conv1d(input_size, int((input_size + 1) * 2), 31, padding=15, stride=1)
        self.conv2 = nn.Conv1d(int((input_size + 1) * 2), int((input_size + 1) * 4), 31, padding=15, stride=1)
        self.conv3 = nn.Conv1d(int((input_size + 1) * 4), int((input_size + 1) * 8), 31, padding=15, stride=1)
        self.conv4 = nn.Conv1d(int((input_size + 1) * 8), int((input_size + 1) * 16), 31, padding=15, stride=1)

        # Same as convolutional but not nearly as much complexity involved. It won't make the model as smart as
            # convolutional layers would but they're faster, they're also necessary to set the correct number of outputs
        self.lin1 = nn.Linear(112 * 92, int((112 * 92) * ((2 / 3) ** 6)))
        self.lin2 = nn.Linear(int((112 * 92) * ((2 / 3) ** 6)), int((112 * 92) * ((2 / 3) ** 12)))
        self.lin3 = nn.Linear(int((112 * 92) * ((2 / 3) ** 12)), 26)

        # Pooling is done to remove "uncertainties" from results of layers
        self.pool1 = nn.MaxPool1d(3, stride=1)
        self.pool2 = nn.MaxPool1d(3, stride=1)

        # Activation functions are extremely important. They "activate" the information from the layers to a usable form
            # Usually the final activation function is sigmoid if binary output or softmax if multiclass. For this
            # particular task, random relu happened to be the best.
            # Often leaky relu is a good default activation function to begin with
        self.random_relu = nn.RReLU()


    def forward(self, instances):
        pass

        # The forward passes the input data though the series of layers and activation functions and produces predictions

        # Convolutional:
        x = self.conv1(instances)
        x = self.pool1(x)
        x = self.random_relu(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.random_relu(x)
        x = self.conv3(x)
        x = self.pool2(x)
        x = self.random_relu(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.random_relu(x)

        # Linear:
        x = x.view(-1, 112 * 92)
        x = self.lin1(x)
        x = self.random_relu(x)
        x = self.lin2(x)
        x = self.random_relu(x)
        x = self.lin3(x)
        x = self.random_relu(x)
        return x

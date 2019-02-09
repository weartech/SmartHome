import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from dataset import DataServe
from torch.utils.data import DataLoader
from model import CNN_Model
import torch
import matplotlib.pyplot as plt
import scipy.signal as sp

# Getting the data
# Note that the data is sometimes normalized beforehand to help clean the data further
instances = np.load("./data/numpy/instances.npy")
labels = np.load("./data/numpy/labels.npy")

# Encoding:
# The point the encoding the data is that it is difficult do set labels that are not numerical.
# One hot encoding converts the strings to numerical values that are necessary for proper comparison with predictions
labels = labels[:,0]
label_encoder = LabelEncoder()
onehot_encoder = OneHotEncoder()
labels = label_encoder.fit_transform(labels)
labels = labels.reshape(-1,1)
labels = onehot_encoder.fit_transform(labels).toarray()
instances = instances.transpose(0, 2, 1)

# 2.5: Splitting training and validation sets
# The idea here is to take 80% of the data and use that to train the model, while using 20% to validate the model.
# Usually, there is also a test set used to validate after finalizing
instance_train_data, instance_valid_data, label_train_data, label_valid_data = \
    train_test_split(instances, labels, test_size=0.2, random_state=0)

# 3.1: PyTorch Dataset:
def produce_data(bs):
    # bs is batch size
    # Using the data split earlier to create datasets that can be easily processed by PyTorch.
    # The DataServe class comes from dataset.py

    train_dataset = DataServe(instance_train_data, label_train_data)
    valid_dataset = DataServe(instance_valid_data, label_valid_data)

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=True)

    return train_loader, val_loader

def produce_model(lr):
    # lr is learning rate
    # Simply a function to get the model, loss function and optimizer.
    # Honestly, the function isn't even necessary, this can just be done right before the training loop.
    model = CNN_Model(instance_train_data.shape[1])
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # Notice how the model parameters are passed to the optimizer, will explain later

    return model, loss_function, optimizer

def evaluate(model, validation_loader):
    # This function is used to determine the accuracy thus far. This is useful for a few reasons:


    total = 0

    for i, batch in enumerate(validation_loader):
        features, label = batch

        prediction = model(features.float())

        num_correct = torch.argmax(prediction, dim=1) == torch.argmax(label, dim=1)
        total += int(num_correct.sum())

    return float(total) / len(validation_loader.dataset)

def main():
    ### This is a majority of the hyperparameter tuning and overall messing around I did with this
      # The more you do it, a better of an understanding you'll get
      # Honestly I'm keeping around just to show how much more efficient it'll be to delegate this part

    ############
    # MSE + SGD + lr=10 + bs=32 +epoch=100     results with 40%
    # MSE + ADAM + Lr=0.001 + bs=32  + epoch=100  results with 83.3%
    # MSE + ADAM + Lr=0.001 + bs=128 + epoch=100   results with 83.3%
    # MSE + ADAM + Lr=0.001 + bs=128 + epoch=130   results with 85%       # This one is good for now
    # CROSS + ADAM + Lr=0.001 + bs=128 + epoch=130  results with 81%

    # MSE + ADAM + Lr=0.001 + bs=32 + epoch=130    results with 81.31%
    # MSE + ADAM + Lr=0.001 + bs=64 + epoch=130    results with 82.92%
    # MSE + ADAM + Lr=0.001 + bs=128 + epoch=130   results with 84.53%    # This one is good for now
    # MSE + ADAM + Lr=0.001 + bs=256 + epoch=130   results with 82.9%

    # Excssive linear layers results with 77%
    # Using Prelu instead of leaky relu results with 84.35%
    # Using normal relu instead of leaky relu results with 51%
    # Using tanh instead of leaky relu results with 81%
    # Using tanh+shrink instead of leaky relu results with sucky results
    # Using relu6 instead of leaky relu results with 83%
    # Using selu instead of leaky relu results with sucky results
    # Using randomized relu instead of leaky relu results with 86-87%      # This one is good for now
    # Keeping leaky relu and changing output activation to random relu results with 84%
    # Keeping random relu and changing output activation to leaky relu results with 84%
    # Changing final output activation function from random relu to softmax results with around 78-80%

    # Changing kernel sizes and padding for all convolutional layers after first from 7, 15 respectively to 31, 7 respectively results with more consistent 87% with some 86% with peak 87.6%, seems better     # This one is good for now
    # Changing kernel sizes and padding for all convolutional layers after first from 7, 15 respectively to 31, 7 respectively and changing first layer kernel and padding from 31, 7 respectively to 61, 30 respectively results with 84-85 %

    # Changing the convolutional layer scale from 3/2 to 2 results with 89% accuracy    # This one is good for now   NOTE: good results begin at epoch 63, between epoch 89-92, begin to have bad results, better stop at maybe around epoch 85, peak was 89.71% at step 3001, epoch 86
        # But it rises back up by step 4001, epoch 115

    # Changing size of pooling layers to 2 results results with peak of 88% then decrease at epoch 50

    # Adding batch normalization to convolutional layers and changing lr to 0.1 from 0.001, and adding model.evalate() and model.train() results with bad results

    # Removing activation functions after convolutional layers results with terrible values

    # Changing first convolutional layer to kernel size 41 and padding 20 from kernel size 31 and padding 15 results with worse results

    # Changing lr to 0.0015 from 0.001 results with 88% peak rather tha 89% peak so not as good
    # Changing lr to 0.0005 from 0.001 results with 88% peak rather than 89% peak so not as good

    # Changing bs to 100 from 128 results with basically the same results but slower
    # Changing bs to 256 from 128 results with 0.89445 but this only happens at epoch 128 which is basically the final recorded epoch
    # Changing bs to 256 from 128 and changing num epoch to 200 from 130 results with 3% after epoch 117 with peak of 88%
    # Changing bs to 192 from 128 and changing num epoch to 200 from 130 results with Max Validation Accuracy: 0.8908765652951699 | Occurs at: 1600   so it's worse, seems to be rising at the end though

    # Changing convolutional layer first multiplier from 2 to 3 and keeping the rest of the multipliers 2 results with 3% after epoch 60ish

    # Changing kernel and padding of convolutional layers after first to 27, 14 from 31, 15 for kernel and padding respectively, results with slightly, SLIGHTLY worse
    # Changing kernel and padding of all convolutional layers to 27, 14 from 31, 15 for kernel and padding respectively, results with slighly worse results
    # Changing kernel and padding of all convolutional layers to 41, 20 from 31, 15 results with same results
    # Changing padding of all convolutional layers from 20 to 10, and 92 to 12 results with 81%
    # Changing padding of all convolutional layers from 20 to 30, and 92 to 172 results with 88% then random
    # Changing kernel and padding of all convolutional layers to 47, 24 from 41, 20 results with 88% peak
    # Changing kernel and padding of all convolutional layers to 39, 20 from 41, 20 results with not as good

    # Changing 80:20 to 75:25 results with terrible values
    ############

    # One of the three most important parts begin here, the training loop:

    # Defining basic hyperparameters:
    num_epoch = 136 # This variable is used to determine how many loops the training loop should perform
                    # The term "epoch" refers to a single loop that goes through all the data

    lr = 0.001      # This variable is the learning rate. It determins how quickly the model learns
                    # While it seems like it would be a good idea to put the learning rate to a large value to train
                        # the model more quickly, doing so can quickly result with overfitting
                    # Overfitting refers to memorizing the dataset. The model tuned itself to the point where it can
                        # only classify examples from the training set. The accuracy using samples outside the training
                        # will drop significantly, ruining the model
                    # Overfitting can be avoided by ensuring the learning rate is sufficiently low. The issue with
                        # setting the learning rate too low is that it'll take too long to train your data

    bs = 128        # This variable is the batch size. Each iteration of the training loop has the model process
                        # batches of the test set. A lower batch size can result with validation accuracy closer to
                        # that of the training accuracy but will slow down the training
                    # Batches are normally randomized to reduce risk of overfitting

    t = 0           # Just a variable I created for printing, I will explain when I get to that part of the loop

    model, loss_func, optimizer = produce_model(lr) # Retrieving the model, loss function, and optimizer
    # The design of the model is key. I'll go into detail in model.py
    # The loss function is used to calculate how inaccurate your model is and informs the back propagation
        # Back propagation refers to how the model determines the gradients with which the weights in the model should
            # be altered.
    # The optimizer optimizes the model. They specifically perform the actual updates to the model based on the results of the loss function
        # As mentioned before, the parameters of the model are passed to the optimizer so the optimizer can access and update the weights

    train_accuracies = [] # This is used to produce the plots. Will discuss plots later in this file.
    valid_accuracies = [] # This is used to produce the plots. Will discuss plots later in this file.

    curr_max_valid_accuracy = 0 # This variable will hold the maximum validation accuracy in order to find the index
    curr_max_index = 0 # The aforementioned index. This is used to help optimize the num_epoch hyperparameter

    for epoch in range(num_epoch): # num_epoch informs the amount of training to be performed here
        accumulated_loss = 0 # Used to calculate accuracy
        total_correct = 0 # Used to calculate accuracy

        train_loader, val_loader = produce_data(bs) # Produce the data that will be processed

        for i, batch in enumerate(train_loader): # Go through each randomized batch
            instances, label = batch # Get instances (data) and labels (answers)
            optimizer.zero_grad() # Zero the gradients so optimization from previous iterations do not carry over

            predictions = model(instances.float()) # Use the models to produce predictions for the input data

            batch_loss = loss_func(input=predictions.squeeze(), target=label.float()) # Calculate loss using loss function
            accumulated_loss += batch_loss # Update accumulated loss
            batch_loss.backward() # Perform the backward propagation mentioned before

            optimizer.step() # Update the model using the optimizer

            num_correct = torch.argmax(predictions, dim=1) == torch.argmax(label, dim=1) # Calculate amount correct in the batch:
                # This is as simple as comparing the predictions and see if they match the labels

            total_correct += int(num_correct.sum()) # Update total correct


            if(t % 50 == 0):
                # Every 50 batch iterations, print the current information. Doing so is important for a few reasons:
                    # Making sure your training is running!
                    # Checking to see if your model has overfit/underfit, is it learning at all, is it getting dumber?
                    # Determine at what point approximately the best accuracy is achieved, to optimize num_epoch
                    # Do not perform this step every time because this will severely slow down training

                train_accuracy = evaluate(model, train_loader) # Determine accuracy of training
                valid_accuracy = evaluate(model, val_loader) # Determine accuracy of validation

                print("Epoch: {}, Step {} | Loss: {} | Number correct: {} | Train accuracy: {} | Validation accuracy: {}".format(epoch + 1, t + 1, accumulated_loss / 100, total_correct, train_accuracy, valid_accuracy))

                train_accuracies.append(train_accuracy) # Update for plots
                valid_accuracies.append(valid_accuracy) # Update for plots

                if(valid_accuracy > curr_max_valid_accuracy):
                    # This is to help optimize num_epoch

                    curr_max_valid_accuracy = valid_accuracy
                    curr_max_index = t

                accumulated_loss = 0 # Reset accumulated loss for accurate outputs

            t += 1

    # The next few parts is plotting the training and validation data over the steps
        # This is important because it enables us to more easily recognize trends in our training
        # These trends can help determine if our model is at a good point, etc.

    # Smoothing:
    train_accuracies = sp.savgol_filter(train_accuracies, 5, 2)
    valid_accuracies = sp.savgol_filter(valid_accuracies, 5, 2)

    # Plotting:
    plt.figure()
    plt.plot(train_accuracies, label="Training")
    plt.plot(valid_accuracies, label="Validation")

    plt.xlabel("Number of Gradient Step")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()

    print("Num Epoch: {} | LR: {} | BS: {}".format(num_epoch, lr, bs))
    print("Max Validation Accuracy: {} | Occurs at: {}".format(curr_max_valid_accuracy, curr_max_index))

    # Normally, a test evaluation would be performed at this point. This project happened to not have a test set to evaluate with
        # but the test evaluation would just be done by test_accuracy = evaluate(model, test_loader)
        # Note that testing can be done in live situations too

    torch.save(model, './data/model_jayjaewonyoo_1002939671.pt') # Save the model so it can actually be used outside of training

if __name__ == "__main__":
    main()

'''
    Save the data in the .csv file, save as a .npy file in ./data
'''
import numpy as np
import os

def createNParrays(currPath, instancesArray, labelsArray):
    currInstances = instancesArray
    currLabels = labelsArray
    elements = os.listdir(path = currPath) # Get all files and directories within current folder

    for element in elements:
        path = os.path.join(currPath, element) # Creating new directory path
        if(os.path.isdir(path)): # If element is a directory
            currInstances, currLabels = createNParrays(path, currInstances, currLabels)
        elif(element.split(".")[1] == "csv"): # If file is csv
            temp = np.loadtxt(open(path, "rb"), delimiter=",")

            if(len(currInstances) == 0):
                currInstances = [temp]
                currLabels = element.replace(".csv", "")
            else:
                currInstances = np.vstack((currInstances, [temp]))
                currLabels = np.vstack((currLabels, element.replace(".csv", "")))

    return currInstances, currLabels

if __name__ == "__main__":
    instances = np.array([])
    labels = np.array([])

    rootPath = os.path.abspath('.') # Get path of this project folder
    instances, labels = createNParrays(rootPath, instances, labels) # Recursively create array

    # Save numpy arrays as files stored in data folder
    np.save(rootPath + "/data/numpy/instances", instances)
    np.save(rootPath + "/data/numpy/labels", labels)

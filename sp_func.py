#import pandas as pd
import numpy as np
import cv2

def CsvToArr (filename, has_label = 0):
    #gay = pd.read_csv(filename, encoding = 'unicode_escape')
    gay = np.loadtxt (filename, delimiter = ",", skiprows = 1).astype (np.uint8)
    label = np.zeros ((gay.shape[0]))
    if (has_label == 1):
        label = gay[:,0].astype (np.uint8)
        gay = gay[:,1:]
    gay = np.reshape (gay/255, (gay.shape[0], 1, 28, 28))
    return gay, label

def CsvToTrainTest (filename, has_label = 0, numOfTestCount = 100):
    #gay = pd.read_csv(filename, encoding = 'unicode_escape')
    gay = np.loadtxt (filename, delimiter = ",", skiprows = 1).astype (np.uint8)
    label = np.zeros ((gay.shape[0]))
    if (has_label == 1):
        test_arr = gay [-1*numOfTestCount:]

        gay = gay[:-1*numOfTestCount]
        label = gay[:,0].astype (np.uint8)
        gay = gay[:,1:]

        test_label = test_arr[:,0].astype (np.uint8)
        test_input = test_arr[:,1:]

        gay = np.reshape (gay/255, (gay.shape[0], 1, 28, 28))
        test_input = np.reshape (test_input/255, (test_input.shape[0], 1, 28, 28))
        
        return gay, label, test_label, test_input
    gay = np.reshape (gay/255, (gay.shape[0], 1, 28, 28))
    return gay, label

def WriteSubmission (filename, arr):
    np.savetxt (filename, arr, delimiter = ',')
def RandomAugmentation (arr):
    #array of image, kich thuoc: [N, 28, 28] voi N la so luong anh
    return 0
    
if __name__ == "__main__":
    arr, label = CsvToArr ("./MNIST/train.csv", has_label = 1)
    

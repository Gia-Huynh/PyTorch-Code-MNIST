#import pandas as pd
import numpy as np

def CsvToArr (filename, has_label = 0):
    #gay = pd.read_csv(filename, encoding = 'unicode_escape')
    gay = np.loadtxt (filename, delimiter = ",", skiprows = 1).astype (np.uint8)
    label = np.zeros ((gay.shape[0]))
    if (has_label == 1):
        label = gay[:,0].astype (np.uint8)
        gay = gay[:,1:]
    gay = np.reshape (gay/255, (gay.shape[0], 1, 28, 28))
    return gay, label

def CsvToTrainVal (filename, has_label = 0, batch_size = 1, numOfTestCount = 100):
    #gay = pd.read_csv(filename, encoding = 'unicode_escape')
    gay = np.loadtxt (filename, delimiter = ",", skiprows = 1).astype (np.uint8)
    label = np.zeros ((gay.shape[0]))
    if (has_label == 1):
        numOfTestCount = gay.shape[0]-(int((gay.shape[0]-numOfTestCount)/batch_size)*batch_size)

        test_arr = gay [-1*numOfTestCount:]
        gay = gay[:-1*numOfTestCount]
        
        label = gay[:,0].astype (np.uint8)
        gay = gay[:,1:]

        test_label = test_arr[:,0].astype (np.uint8)
        test_input = test_arr[:,1:].astype (np.uint8)

        label = np.reshape (label, (int(label.shape[0]/batch_size), batch_size))
        gay = np.reshape (gay/255, (int(gay.shape[0]/batch_size), batch_size,1, 28, 28))
        test_input = np.reshape (test_input/255, (test_input.shape[0], 1, 1, 28, 28))
        
        return gay, label, test_input, test_label
    gay = np.reshape (gay/255, (gay.shape[0], 1, 28, 28))
    return gay, label

def WriteSubmission (filename, arr):
    #https://numpy.org/devdocs/reference/generated/numpy.savetxt.html
    np.savetxt (filename, arr, delimiter = ',', fmt = '%1d', header = "ImageId,Label", comments='')
def RandomAugmentation (arr):
    #array of image, kich thuoc: [N, 28, 28] voi N la so luong anh
    return 0
    
if __name__ == "__main__":
    arr, label = CsvToArr ("./MNIST/train.csv", has_label = 1)
    

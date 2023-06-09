#https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
import sp_func
import numpy as np

epochs = 30
batch_size = 4
use_gpu = 0
print ("Use gpu: ", use_gpu)

train_arr, train_label, val_arr, val_label = sp_func.CsvToTrainVal ("./MNIST/train.csv", has_label = 1, batch_size = batch_size)
train_arr.astype(np.float32)
val_arr.astype(np.float32)

test_arr, _ = sp_func.CsvToArr ("./MNIST/test.csv", has_label = 0)
test_arr.astype(np.float32)

import torch
#import torchvision
#import torchvision.transforms as transforms
import time
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:"+str(use_gpu) if (torch.cuda.is_available() and (use_gpu!=-1)) else "cpu")

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 40, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(40, 80, 3)
        self.conv3 = nn.Conv2d(80, 160, 3)
        self.fc1 = nn.Linear(160, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        #print(x.size())
        x = self.pool(F.relu(self.conv2(x)))
        #print(x.size())
        x = self.pool(F.relu(self.conv3(x)))
        #print(x.size())
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        #print(x.size())
        x = F.relu(self.fc1(x))
        #print(x.size())
        x = F.relu(self.fc2(x))
        #print(x.size())
        x = self.fc3(x)
        #print(x.size())
        return x

net = torch.load('test_model')
#print (net)
net.to(device)
net.eval()

if (torch.cuda.is_available() and (use_gpu!=-1)):
    print ("Using gpu: ", torch.cuda.get_device_name(use_gpu))
else:
    print ("Using cpu")

def create_testset_Output(net, test_arr, device):
    output_testset = net(torch.from_numpy(test_arr).type(torch.FloatTensor).to(device))
    gay = output_testset.detach().cpu().numpy()
    gay = np.argmax (gay, axis = 1).astype (np.uint16)

    gay = np.append (gay[:, np.newaxis], np.arange (1, gay.shape[0]+1)[:, np.newaxis],1)
    
    sp_func.WriteSubmission ('result.csv', gay[:,::-1])
    
create_testset_Output (net, test_arr, device)

#Sanity check
def sanity_check (val_label, val_arr, idx, net):
    from cv2 import imshow, imwrite
    imshow (str(val_label[idx]), (val_arr[idx][0][0]*255).astype(np.uint8))
    inputs = torch.from_numpy(val_arr[idx]).type(torch.FloatTensor).to(device)
    output = net(inputs)
    _, predicted = torch.max(output.data, 1)
    print ("Label: ",val_label[idx]," Predicted:",predicted)
    imwrite ('Bao_cao_image/' + str (idx)+'_prediction_'+str(predicted.item())+'.png', (val_arr[idx][0][0]*255).astype(np.uint8))

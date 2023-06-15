#https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
import sp_func
import numpy as np

epochs = 30
batch_size = 64
use_gpu = -1
print ("Use gpu: ", use_gpu)

train_arr, train_label, val_arr, val_label = sp_func.CsvToTrainVal ("./MNIST/train.csv", has_label = 1, batch_size = batch_size)
train_arr.astype(np.float32)
val_arr.astype(np.float32)

#test_arr, _ = sp_func.CsvToArr ("./MNIST/test.csv", has_label = 0)
#test_arr.astype(np.float32)

import torch
#import torchvision
#import torchvision.transforms as transforms
import time
import torch.nn as nn
import torch.nn.functional as F

if (torch.cuda.is_available() and (use_gpu!=-1)):
    print ("Using gpu: ", torch.cuda.get_device_name(use_gpu))
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
  
net = Net()
#print (net)
net.to(device)
import torch.optim as optim

criterion = nn.CrossEntropyLoss(reduction = 'mean')
optimizer = optim.SGD(net.parameters(), lr=0.001*(batch_size/16+1), momentum=0.95)



start = time.time()
for epoch in range(epochs):  # loop over the dataset multiple times
    epoch_start = time.time()
    running_loss = 0.0
    count = 0
    for i, data in enumerate(train_arr, 0):

        # get the inputs; data is a list of [inputs, labels]
        #inputs, labels = data
        labels = train_label[i]
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        if (use_gpu):
            input_tensor = torch.from_numpy(data).type(torch.cuda.FloatTensor).to(device)
            label_tensor = torch.from_numpy(labels).type(torch.cuda.LongTensor).to(device)
        else:
            input_tensor = torch.from_numpy(data).type(torch.FloatTensor).to(device)
            label_tensor = torch.from_numpy(labels).type(torch.LongTensor).to(device)

        
        outputs = net(input_tensor)
        
        #if (i<1):
        #    print ("ay___________")
        #    print (outputs)
        #    print (label_tensor)    
        loss = criterion(outputs, label_tensor)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        count+=1
        if i % (20480/batch_size) == (20480/batch_size-1): 
            #print(f'   [{epoch + 1}, {i*batch_size}] loss: {running_loss / count:.3f}')
            running_loss = 0.0
            count = 0
    #print (f'Epoch: {epoch}, Time: {time.time() - epoch_start}')
    #Accuracy Test
    #val_arr, val_label
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range (0, val_arr.shape[0]):
            inputs = torch.from_numpy(val_arr[i]).type(torch.FloatTensor).to(device)
            labels = torch.as_tensor(val_label[i]).type(torch.LongTensor).to(device)
            output = net(inputs)
            _, predicted = torch.max(output.data, 1)
            total += 1
            correct += (predicted == labels).sum().item()
            #if (i < 3):
            #    print (predicted, labels,(predicted == labels))

    print('   Validation set accuracy: %d %%' % (100 * correct / total))
net.eval()

if (torch.cuda.is_available() and (use_gpu!=-1)):
    print ("Using gpu: ", torch.cuda.get_device_name(use_gpu))
else:
    print ("Using cpu")
print('Finished Training, Total time taken: ', time.time() - start)

torch.save (net,'test_model')

def create_testset_Output(net, test_arr, device):
    output_testset = net(torch.from_numpy(test_arr).type(torch.FloatTensor).to(device))
    gay = output_testset.detach().cpu().numpy()
    gay = np.argmax (gay, axis = 1).astype (np.uint16)

    gay = np.append (gay[:, np.newaxis], np.arange (1, gay.shape[0]+1)[:, np.newaxis],1)
    
    sp_func.WriteSubmission ('result.csv', gay[:,::-1])
    
#create_testset_Output (net, test_arr, device)

#Sanity check
def sanity_check (val_label, val_arr, idx, net):
    from cv2 import imshow
    imshow (str(val_label[idx]), (val_arr[idx][0][0]*255).astype(np.uint8))
    inputs = torch.from_numpy(val_arr[idx]).type(torch.FloatTensor).to(device)
    output = net(inputs)
    _, predicted = torch.max(output.data, 1)
    print ("Label: ",val_label[idx]," Predicted:",predicted)

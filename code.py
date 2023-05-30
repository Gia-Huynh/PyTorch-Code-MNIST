#https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
import sp_func
import numpy as np
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
import time

torch.cuda.is_available()
use_gpu = 1 
device = torch.device("cuda:1" if (torch.cuda.is_available() and use_gpu) else "cpu")

train_arr, train_label, test_arr, test_label = sp_func.CsvToTrainTest ("./MNIST/train.csv", has_label = 1)
train_arr.astype(np.float32)
test_arr.astype(np.float32)

test_arr, _ = sp_func.CsvToArr ("./MNIST/test.csv", has_label = 0)
test_arr.astype(np.float32)

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.conv3 = nn.Conv2d(20, 40, 3)
        self.fc1 = nn.Linear(40, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 0) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
  
net = Net()
net.to(device)
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)



for epoch in range(1):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_arr, 0):
        start = time.time()

        # get the inputs; data is a list of [inputs, labels]
        #inputs, labels = data
        labels = train_label[i]
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        input_tensor = torch.from_numpy(data).type(torch.FloatTensor).to(device)
        label_tensor = torch.as_tensor(train_label[i]).type(torch.LongTensor).to(device)
        
        outputs = net(input_tensor)
        loss = criterion(outputs, label_tensor)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}, Time taken: {(time.time() - start) * 1000}')
            running_loss = 0.0

    #Accuracy Test
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test set: %d %%' % (100 * correct / total))
net.eval()
print('Finished Training')

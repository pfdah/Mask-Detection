import torch
from divide_images import test_train
from dataloader import create_dataloader
from model import Network
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1234)

src_folder = "./data/"
test_path = "./test_data/"
train_path = "./train_data/"
sanity_path = './sanity_data/'

test_train(src_folder, test_path, train_path)

train_loader = create_dataloader(train_path, batch_size = 4)
test_loader = create_dataloader(test_path, batch_size = 4)
sanity_loader = create_dataloader(sanity_path, batch_size = 1)

net = Network()
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

print('=========Training started ===========')
net.train()
for epoch in range(4):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        outputs =  F.sigmoid(outputs)
        outputs = torch.reshape(outputs,(-1,))
        labels = labels.type(torch.float)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 200 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
            running_loss = 0.0

print('=========Training stopped ===========')

torch.save(net,'./model.pt')


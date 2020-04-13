import torch
import numpy as np

train_data_path = './training/cnn_data'
train_data_path2 = './training/train_data'
test_data_path = './testing/cnn_data'
# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')


from torchvision import datasets
import torchvision.transforms as transforms

preprocess = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
dataset1 = datasets.ImageFolder(root = train_data_path,
                                        transform=preprocess)
dataset2 = datasets.ImageFolder(root = train_data_path2,
                                        transform=preprocess)
loader = torch.utils.data.DataLoader(dataset=dataset1, 
                                        batch_size = 20, 
                                        shuffle = True)
loader2 = torch.utils.data.DataLoader(dataset=dataset2, 
                                        batch_size = 20, 
                                        shuffle = True)
dataset = datasets.ImageFolder(root = test_data_path,
                                        transform=preprocess)
test_loader = torch.utils.data.DataLoader(dataset=dataset, 
                                        batch_size = 20, 
                                        shuffle = True)

import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        # pool / 2
        self.conv2 = nn.Conv2d(64, 128, 3, padding = 1)
        # pool / 2
        self.conv3 = nn.Conv2d(128, 256, 3, padding = 1)
        self.conv4 = nn.Conv2d(256, 256, 3, padding = 1)
        # pool / 2
        self.conv5 = nn.Conv2d(256, 512, 3, padding = 1)
        self.conv6 = nn.Conv2d(512, 512, 3, padding = 1)
        # pool / 2
        self.conv7 = nn.Conv2d(512, 512, 3, padding = 1)
        self.conv8 = nn.Conv2d(512, 512, 3, padding = 1)
        # pool / 2
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        self.softmax = torch.nn.Softmax(dim = 1)
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 20)
        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = self.pool(F.relu(self.conv6(F.relu(self.conv5(x)))))
        x = self.pool(F.relu(self.conv8(F.relu(self.conv7(x)))))
        # flatten image input
        x = x.view(-1, 512 * 4 * 4)
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        #x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# define the CNN architecture
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         # convolutional layer (sees 32x32x3 image tensor)
#         self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
#         # convolutional layer (sees 16x16x16 tensor)
#         self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
#         # convolutional layer (sees 8x8x32 tensor)
#         self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
#         # max pooling layer
#         self.pool = nn.MaxPool2d(2, 2)
#         # linear layer (64 * 4 * 4 -> 500)
#         self.fc1 = nn.Linear(64 * 8 * 8, 500)
#         # linear layer (500 -> 10)
#         self.fc2 = nn.Linear(500, 20)
#         # dropout layer (p=0.25)
#         self.dropout = nn.Dropout(0.25)

#     def forward(self, x):
#         # add sequence of convolutional and max pooling layers
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
#         # flatten image input
#         x = x.view(-1, 64 * 8 * 8)
#         # add dropout layer
#         x = self.dropout(x)
#         # add 1st hidden layer, with relu activation function
#         x = F.relu(self.fc1(x))
#         # add dropout layer
#         x = self.dropout(x)
#         # add 2nd hidden layer, with relu activation function
#         x = self.fc2(x)
#         return x

# create a complete CNN
model = Net()
if train_on_gpu:
    model.cuda()


import torch.optim as optim
# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()
# specify optimizer
optimizer = optim.SGD(model.parameters(), lr = 0.002)

# number of epochs to train the model
n_epochs = 30
loss_min = np.Inf # track change in validation loss

for epoch in range(1, n_epochs + 1):
    # keep track of training and validation loss
    train_loss = 0.0
    test_loss = 0.0
    ###################
    # train the model #
    ###################
    model.train()
    for data, target in loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()*data.size(0)
    for data, target in loader2:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()*data.size(0)
    model.eval()
    for data, target in test_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update average validation loss 
        test_loss += loss.item()*data.size(0)

    train_loss = train_loss/(len(loader.sampler) + len(loader2.sampler))
    test_loss = test_loss/len(test_loader.sampler)
    # print training/validation statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tTesting Loss: {:.6f}\n'.format(epoch, train_loss, test_loss))
    if test_loss <= loss_min:
        print("Saving model...")
        torch.save(model.state_dict(), 'model_voc.pt')
        loss_min = test_loss
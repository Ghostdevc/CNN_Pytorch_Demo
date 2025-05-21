'''
problem statement: CIFAR10 dataset, classification of ten classes
CNN
'''

# import libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np


# load dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Using device: {device}')

def get_data_loaders(batch_size = 64):

  # preprocess
  transform = transforms.Compose([
      transforms.ToTensor(), # transform image into tensor
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # normalizing rgb
  ])

  # download CIFAR10 and create train and test sets
  train_set = torchvision.datasets.CIFAR10(root = 'pytorch_demo/data', train = True, download = True, transform = transform)
  test_set = torchvision.datasets.CIFAR10(root = 'pytorch_demo/data', train = False, download = True, transform = transform)

  # data loader
  train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 2)
  test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle = False, num_workers = 2)

  return train_loader, test_loader



# visualize dataset

def imshow(img):
  img = img / 2 + 0.5
  npimg = img.numpy()
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()

def get_sample_images(loader):
  dataiter = iter(loader)
  images, labels = next(dataiter)
  return images, labels

def visualize(n):
  train_loader, test_loader = get_data_loaders()
  # visualizing n number of data
  images, labels = get_sample_images(train_loader)
  plt.figure()
  for i in range(n):
    plt.subplot(1, n, i+1)
    imshow(images[i])
    plt.title(f'Label: {labels[i].item()}')
    plt.axis('off')
  plt.show()

visualize(5)



# build CNN

class CNN(nn.Module):

  def __init__(self):

    super(CNN, self).__init__()

    # conv, activation, pooling, dropout, linear,

    self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1) # in_channel = 3 (rgb), out_channels = 32 filter number

    self.relu = nn.ReLU()

    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

    self.dropout = nn.Dropout(p=0.2)

    self.fc1 = nn.Linear(64 * 8 * 8, 128)

    self.fc2 = nn.Linear(128, 10)

    # image 3x32x32 -> conv (32) -> pool (16) -> conv (16) -> relu (16) -> pool (8) -> image = 8x8

  def forward(self, x):
    x = self.pool(self.relu(self.conv1(x)))
    x = self.pool(self.relu(self.conv2(x)))
    x = x.view(-1, 64 * 8 * 8) # flatten
    x = self.dropout(self.relu(self.fc1(x)))
    x = self.fc2(x)
    return x

define_loss_and_optimizer = lambda model: (
    nn.CrossEntropyLoss(),
    optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)
)


#model = CNN()
#define_loss_and_optimizer(model)




# training

def train_model(model, train_loader, test_loader, criterion, optimizer, epochs = 5):

  model.train() # switch model to train mode
  train_losses = [] # empty list to save loss
  test_losses = []


  for epoch in range(epochs):
    total_loss_train = 0.0
    total_loss_test = 0.0

    for images, labels in train_loader:

      images, labels = images.to(device), labels.to(device)

      optimizer.zero_grad() # zeroing gradients
      outputs = model(images) # forward propagation
      loss = criterion(outputs, labels) # loss value
      loss.backward() # back propagation
      optimizer.step() # weigth update

      total_loss_train += loss.item()

    for images, labels in test_loader:

      images, labels = images.to(device), labels.to(device)

      outputs = model(images) # forward propagation
      loss = criterion(outputs, labels) # loss value

      total_loss_test += loss.item()

    avg_loss_train = total_loss_train / len(train_loader)
    train_losses.append(avg_loss_train)

    avg_loss_test = total_loss_test / len(test_loader)
    test_losses.append(avg_loss_test)

    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss_train:.4f}, Test Loss: {avg_loss_test:.4f}')

  # loss graph
  plt.figure()
  plt.plot(range(1, epochs+1), train_losses, marker = 'o', linestyle = '-', color = 'red', label = 'Train Loss')
  plt.plot(range(1, epochs+1), test_losses, marker = 'o', linestyle = '-', color = 'blue', label = 'Test Loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.title('Training Loss')
  plt.legend()
  plt.show()




# model test and evaluation

def test_model(model, test_loader, dataset_type):
  model.eval() # switch model to eval mode

  correct = 0 # correct count
  total = 0

  with torch.no_grad():
    for images, labels in test_loader:
      images, labels = images.to(device), labels.to(device)

      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)

      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  accuracy = 100 * correct / total
  print(f'Accuracy of the model on the {dataset_type} images: {accuracy:.2f}%')



if __name__ == '__main__':
  # dataset loading
  train_loader, test_loader = get_data_loaders()

  # visualize

  # training
  model = CNN().to(device)
  criterion, optimizer = define_loss_and_optimizer(model)
  train_model(model, train_loader, test_loader, criterion, optimizer, epochs = 10)

  # test
  test_model(model, test_loader, 'test')
  test_model(model, train_loader, 'train')
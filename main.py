import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
import torchvision
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Download and load the training data
data_dir = './Data/flower_data/train'
test_dir = './Data/flower_data/valid'
train_data = datasets.ImageFolder(data_dir, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=1,
                                           shuffle=True, num_workers=1)
test_data = datasets.ImageFolder(test_dir, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1,
                                          shuffle=True, num_workers=1)

# Define a model


class Net(nn.Module):
    def __init__(self, input_shape=(3, 500, 500)):
        self.ModelPath = "convModel.txt"
        super(Net, self).__init__()
        # color chanel size |image size| kernel size
        self.conv1 = nn.Conv2d(3, 250, 5, 5)
        # image size = (250 - (10 - 1 - 1))/10 +1 = 25
        # image size after pool 25 / 2 = 246
        self.conv2 = nn.Conv2d(250, 200, 5, 5)
        # image size = 246 - 7 + 1 = 240
        # image size after pool 240 / 2 = 120
        self.conv3 = nn.Conv2d(200, 150, 1, 1)
        # image size = 120 - 5 + 1 = 116
        # image size after pool 116 / 2 = 58
        # kernel size | stride size
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(150 * 2 * 2, 250)
        self.fc2 = nn.Linear(250, 150)
        self.fc3 = nn.Linear(150, 102)

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        return None

    def forward(self, x):
        activationFunction = F.relu
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def train(self, dataLoader: torch.utils.data.DataLoader) -> None:
        self.model = Net()
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(self.device)
        # learning rate
        learning_rate = 0.001
        epochs = 1
        total_steps = len(dataLoader)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        with torch.no_grad():
            for epoch in range(epochs):
                for i, (images, labels) in enumerate(dataLoader):

                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    # Forward pass
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)

                    # Backward and optimize
                    # optimizer.zero_grad()
                    # loss.backward()
                    optimizer.step()

    def save(self):
        torch.save(self.model.state_dict(), self.ModelPath)

    def GetStats(self, dataLoader: torch.utils.data.DataLoader):
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(102)]
        n_class_samples = [0 for i in range(102)]
        batch_size = dataLoader.batch_size
        for images, labels in dataLoader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(images)
            # max returns (value ,index)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            for i in range(batch_size):
                label = labels[i]
                pred = predicted[i]
                if (label == pred):
                    n_class_correct[label] += 1
                n_class_samples[label] += 1

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network: {acc} %')

    # for i in range(10):
    #     acc = 100.0 * n_class_correct[i] / n_class_samples[i]
    #     print(f'Accuracy of {self.classes[i]}: {acc} %')


if __name__ == '__main__':
    model = Net()
    model.train(train_loader)
    model.save()
    model.GetStats(test_loader)

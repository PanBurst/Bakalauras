import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

import torch
import torchvision
from torchvision import datasets
from torchvision import transforms as T  # for simplifying the transforms
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split
from torchvision import models
import timm
from timm.loss import LabelSmoothingCrossEntropy
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
import time
import copy


def get_data_loaders(data_dir, batch_size, train=False):
    if train:
        # train
        transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomApply(torch.nn.ModuleList([T.ColorJitter()]), p=0.25),
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225)),  # imagenet means
            T.RandomErasing(p=0.2, value='random')
        ])
        train_data = datasets.ImageFolder(os.path.join(
            data_dir, "train/"), transform=transform)
        train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=4)
        return train_loader, len(train_data)
    else:
        # val/test
        transform = T.Compose([  # We dont need augmentation for test transforms
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225)),  # imagenet means
        ])
        val_data = datasets.ImageFolder(os.path.join(
            data_dir, "valid"), transform=transform)
        test_data = datasets.ImageFolder(os.path.join(
            data_dir, "valid"), transform=transform)
        val_loader = DataLoader(
            val_data, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(
            test_data, batch_size=batch_size, shuffle=True, num_workers=4)
        return val_loader, test_loader, len(val_data), len(test_data)


dataset_path = "./Data/flower_data"

(train_loader, train_data_len) = get_data_loaders(dataset_path, 128, train=True)
(val_loader, test_loader, valid_data_len, test_data_len) = get_data_loaders(
    dataset_path, 32, train=False)
classes = [x for x in range(1, 103)]
print(classes, len(classes))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.hub.load('facebookresearch/deit:main',
                       'deit_tiny_patch16_224', pretrained=True)

for param in model.parameters():  # freeze model
    param.requires_grad = False

n_inputs = model.head.in_features
model.head = nn.Sequential(
    nn.Linear(n_inputs, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, len(classes))
)
model = model.to(device)
criterion = LabelSmoothingCrossEntropy()
criterion = criterion.to(device)
optimizer = optim.Adam(model.head.parameters(), lr=0.001)
exp_lr_scheduler = optim.lr_scheduler.StepLR(
    optimizer, step_size=3, gamma=0.97)

dataloaders = {
    "train": train_loader,
    "val": val_loader
}
dataset_sizes = {
    "train": train_data_len,
    "val": valid_data_len
}


def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print("-"*10)

        for phase in ['train', 'val']:  # We do training and validation phase per epoch
            if phase == 'train':
                model.train()  # model to training mode
            else:
                model.eval()  # model to evaluate

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # no autograd makes validation go faster
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)  # used for accuracy
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()  # step at end of epoch

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print("{} Loss: {:.4f} Acc: {:.4f}".format(
                phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                # keep the best validation accuracy model
                best_model_wts = copy.deepcopy(model.state_dict())
        print()
    time_elapsed = time.time() - since  # slight error
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print("Best Val Acc: {:.4f}".format(best_acc))

    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':

    model_ft = train_model(model, criterion, optimizer, exp_lr_scheduler)

    test_loss = 0.0
    class_correct = list(0 for i in range(len(classes)))
    class_total = list(0 for i in range(len(classes)))
    model.eval()

    for data, target in tqdm(test_loader):
        data, target = data.to(device), target.to(device)
        with torch.no_grad():  # turn off autograd for faster testing
            output = model(data)
            loss = criterion(output, target)
        test_loss = loss.item() * data.size(0)
        _, pred = torch.max(output, 1)
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.cpu().numpy())
        if len(target) == 32:
            for i in range(32):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

    test_loss = test_loss / test_data_len
    print('Test Loss: {:.4f}'.format(test_loss))
    for i in range(len(classes)):
        if class_total[i] > 0:
            print("Test Accuracy of %5s: %2d%% (%2d/%2d)" % (
                classes[i], 100*class_correct[i] /
                class_total[i], np.sum(
                    class_correct[i]), np.sum(class_total[i])
            ))
        else:
            print("Test accuracy of %5s: NA" % (classes[i]))
    print("Test Accuracy of %2d%% (%2d/%2d)" % (
        100*np.sum(class_correct) /
        np.sum(class_total), np.sum(class_correct), np.sum(class_total)
    ))

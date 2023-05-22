import torch as th
import torch.nn as nn
import Model as _model
import torch.nn.functional as F
from torchvision import transforms as T  # for simplifying the transforms
from torch import nn, optim
from tqdm import tqdm
from torchvision import datasets
from timm.loss import LabelSmoothingCrossEntropy
from torch.utils.data import DataLoader, sampler, random_split
from tqdm import tqdm
import os
import wandb
import timm.loss
import random


import confusionMatrix as confMatrix

import matplotlib.pyplot as plt

import torchvision.transforms.functional as TF

from torchmetrics.classification import MulticlassAccuracy


def fileDFS(path) -> list:
    newFilePaths = []
    newPaths = os.listdir(path)
    for newPath in newPaths:
        if '.csv' in newPath:
            newFilePaths.append(path+"\\"+newPath)
        elif not os.path.isfile(path+"\\"+newPath):
            newFilePaths.extend(fileDFS(path+"\\"+newPath))
    return newFilePaths


class GLCM_LBPDataset(th.utils.data.Dataset):
    def __init__(self, path):
        filePaths = fileDFS(path)
        a = filePaths
        self.data = []
        for filePath in filePaths:
            className = int(filePath.split("\\")[2])-1
            with open(filePath, 'r') as csvfile:
                values = csvfile.readline().split(',')
                values = th.tensor(list(map(lambda x: float(x), values)))
                self.data.append((values, className))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def loadData(input: str) -> bool:
    a = input
    return a


def fileIsValid(x):
    return ".csv" in x


# pathGLCM_LBP = r'./DataGLCM_LBP/flower_data/train'
# dataset = datasets.DatasetFolder(
#     root=pathGLCM_LBP, loader=loadData, is_valid_file=fileIsValid)
# train_loader = DataLoader(dataset, 1, False)

# ogDataset = datasets.ImageFolder(r'./Data/flower_data/train')


class ConcatDataset(th.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        verticalFlip = random.random() < 0.5
        horizontalFlip = random.random() < 0.5

        retuple = []
        for d in self.datasets:
            a = d[i]
            t, n = d[i]
            if horizontalFlip:
                t = TF.hflip(t)
            if verticalFlip:
                t = TF.vflip(t)
            retuple.append((t, n))

        return tuple(retuple)

    def __len__(self):
        return min(len(d) for d in self.datasets)


class LinearNet(nn.Module):
    def __init__(self, input_shape=(3, 500, 500)):
        self.ModelPath = "GLCM_LBP_Model.txt"
        super(LinearNet, self).__init__()
        self.fc1 = nn.Linear(152, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 768)
        self.fc4 = nn.Linear(768, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 102)
        self.device = th.device(
            'cuda' if th.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x


class LinearModel(_model.Model):
    def __init__(self, modelPath):
        self.Model = LinearNet()
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.modelPath = modelPath

    def train(self, loader, criterion, optimizer, config, testLoader=None):
        self.config = dict(config)
        self.config["architecture"] = "LinearModel"
        example_ct = 0  # number of examples seen
        batch_ct = 0
        loss = 0
        for epoch in tqdm(range(self.config["epochs"])):
            for _, (images, labels) in enumerate(loader):
                images = images.to(self.device, dtype=th.float)
                labels = labels.to(self.device)

                # Forward pass ➡
                outputs = self.Model(images)
                loss = criterion(outputs, labels)

                # Backward pass ⬅
                optimizer.zero_grad()
                loss.backward()

                # Step with optimizer
                optimizer.step()

                example_ct += len(images)
                batch_ct += 1

            if testLoader is not None and (epoch+1) % 20 == 0 and epoch > 0:
                (top1valid, top5valid), (top1train,
                                         top5train) = self.test(testLoader, loader)
                example_ct += 1
                wandb.log({"epoch": epoch, "loss": loss,
                           "accuracy": top1valid, "top 1 validation": top1valid, "top 5 validation": top5valid, "top 1 train": top1train, "top 5 train": top5train}, step=epoch)
                if epoch % 5 == 0:
                    self.LogConfusionMatrix(epoch, testLoader)

        modelName = self.modelPath.split(".")[0]
        # th.onnx.export(self.Model, images, f"model_{modelName}.onnx")
        # wandb.save("model.onnx")
        self.SaveModel(f"model_{modelName}.pt")

    def test(self, validLoader, trainLoader=None):
        self.Model.eval()
        validacc = top1, top5 = self.testLoad(validLoader)
        if trainLoader is not None:
            trainacc = top1train, top5train = self.testLoad(trainLoader)
            return validacc, trainacc
        else:
            return validacc

    def SaveModel(self, path: str = "model.pt"):
        self.Model.eval()
        th.save(self.Model.state_dict(), path)

    def LogConfusionMatrix(self, epoch, testLoader):
        self.Model.eval()
        predictions = []
        trueLabels = []
        for _, (images, labels) in enumerate(testLoader):
            images, labels = images.to(self.device), labels.to(self.device)
            _, prediction = th.max(self.Model(images).data, 1)
            predictions += prediction.tolist()
            trueLabels += labels.tolist()
        allLabels = [x for x in range(1, 103)]
        confusionMatrix = confMatrix.ConfusionMatrix(predictions, trueLabels)
        confusionMatrixImage = wandb.Image(
            confusionMatrix, caption="Klaidų matrica")
        wandb.log(
            {"confusion matrix_basicTransformer": confusionMatrixImage, "epoch": epoch})
        # wandb.log({"my_conf_mat_id" : wandb.plot.confusion_matrix(
        #             preds=predictions, y_true=trueLabels,
        #             class_names=allLabels)})

    def testLoad(self, loader):
        self.Model.eval()
        top1acc = MulticlassAccuracy(num_classes=102, top_k=1).to(self.device)
        top5acc = MulticlassAccuracy(num_classes=102, top_k=5).to(self.device)
        with th.no_grad():
            for _, (images, labels) in enumerate(loader):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.Model(images)
                _ = top1acc.update(outputs, labels)
                _ = top5acc.update(outputs, labels)
            return top1acc.compute(), top5acc.compute()


def Make(config, saveModelPath):

    transform = T.Compose(
        [
            T.ToTensor()])
    model = LinearModel(saveModelPath)
    train_data = GLCM_LBPDataset(os.path.join(
        config["datasetPath"], "train/"))
    train_loader = DataLoader(
        train_data, batch_size=config["batch_size"], shuffle=True, num_workers=4)

    test_data = GLCM_LBPDataset(os.path.join(
        config["datasetPath"], "valid"))
    test_loader = DataLoader(
        test_data, batch_size=config["batch_size"], shuffle=True, num_workers=4)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.Model.parameters(),
                           lr=config["learning_rate"])
    n_config = dict(config)
    n_config["architecture"] = "LinearModel"
    return model, train_loader, test_loader, criterion, optimizer, n_config

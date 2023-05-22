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
import random 
import confusionMatrix as confMatrix
import wandbConvolution as rgbModel
import wandbConvolutionTrue as grayModel

import torchvision.transforms.functional as TF

from torchmetrics.classification import MulticlassAccuracy

class LinearModel(nn.Module):
    def __init__(self, grayModel, rgbModel):
        super(LinearModel, self).__init__()

        self.grayModel = grayModel
        self.rgbModel = rgbModel
        self.Layer1 = nn.Linear(102*2, 1024)
        self.Layer2 = nn.Linear(1024, 256)
        self.Layer3 = nn.Linear(256, 512)
        self.Layer4 = nn.Linear(512, 128)
        self.Layer5 = nn.Linear(128, 256)
        self.Layer6 = nn.Linear(256, 102)
        
        self.dropout = nn.Dropout(0.2)

    def forward(self, _x):

        _grayImage, _rgbImage = _x
        x1 = self.grayModel(_grayImage)
        x2 = self.rgbModel(_rgbImage)
        x = th.cat((x1, x2), dim=1)
        x = F.relu(self.Layer1(x))
        
        x = self.dropout(x)
        x = F.relu(self.Layer2(x))
        x = F.relu(self.Layer3(x))
        
        x = self.dropout(x)
        x = F.relu(self.Layer4(x))
        x = F.relu(self.Layer5(x))

        x = self.Layer6(x)
        return x


class ConvolutionModel2(_model.Model):
    def __init__(self, cnnModel1, cnnModel2, modelPath: str, model:LinearModel=None):
        super(_model.Model, self).__init__()
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        cnnModel2.eval()
        cnnModel1.eval()
        cnnModel1.to(self.device)
        cnnModel2.to(self.device)
        self.Model = LinearModel(cnnModel1, cnnModel2)
        if model is not None:
            self.Model = model
        self.modelPath = modelPath
        self.Model.to(self.device)

    def train(self, loader, criterion, optimizer, config, testLoader=None):
        self.config = dict(config)
        self.config["architecture"] = "Convolution2"
        example_ct = 0  # number of examples seen
        batch_ct = 0
        lastAccuracy = 0
        for epoch in tqdm(range(self.config["epochs"])):
            self.Model.train()
            for _, ((grayImages, _), (rgbImages, labels)) in enumerate(loader):
                grayImages = grayImages.to(self.device)
                rgbImages = rgbImages.to(self.device)
                images = (grayImages, rgbImages)
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
                # Report metrics every 25th batch
                if ((batch_ct + 1) % 25) == 0:
                    # logging to wandb
                    print(
                        f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")
            
            if testLoader is not None and (epoch+1)%2 == 0:

                (top1valid, top5valid), (top1train, top5train) = self.test(testLoader, loader)
                example_ct += 1
                wandb.log({"epoch": epoch, "loss": loss,
                          "accuracy": top1valid, "top 1 validation": top1valid, "top 5 validation": top5valid, "top 1 train": top1train, "top 5 train" : top5train}, step=epoch)
            
                self.LogConfusionMatrix(epoch, testLoader)

        modelName = self.modelPath.split(".")[0]
        # dummy_input = th.randn(1, 4, 500, 500).cuda()
        # th.onnx.export(self.Model, dummy_input, f"model_{modelName}.onnx")
        # wandb.save("model.onnx")
        self.SaveModel(f"model_{modelName}.pt")

    def test(self, validLoader, trainLoader = None):
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
        for _, ((grayImages, _), (rgbImages, labels)) in enumerate(testLoader):
            grayImages, rgbImages, labels = grayImages.to(self.device), rgbImages.to(self.device), labels.to(self.device)
            images = (grayImages, rgbImages)
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
            for _, ((grayImages, _), (rgbImages, labels)) in enumerate(loader):
                grayImages, rgbImages, labels = grayImages.to(self.device), rgbImages.to(self.device), labels.to(self.device)
                images = (grayImages, rgbImages)
                outputs = self.Model(images)
                _ = top1acc.update(outputs, labels)
                _ = top5acc.update(outputs, labels)
            return top1acc.compute(), top5acc.compute()

def Make(config, modelPath, grayModelPath: str, rgbModelPath: str):

    model1 = grayModel.ConvolutionNet()
    model1.load_state_dict(th.load(grayModelPath))

    model2 = rgbModel.ConvolutionNet()
    model2.load_state_dict(th.load(rgbModelPath))

    model = ConvolutionModel2(model1, model2, modelPath)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.Model.parameters(),
                           lr=config["learning_rate"])

    train_loader, test_loader = createDataLoader(config["grayDataset"], config["rgbDataset"], config)
    n_config = dict(config)
    n_config["architecture"] = "Convolution"
    return model, train_loader, test_loader, criterion, optimizer, n_config

def MakeLoad(config, modelPath, grayModelPath: str, rgbModelPath: str):

    model1 = grayModel.ConvolutionNet()
    model1.load_state_dict(th.load(grayModelPath))

    model2 = rgbModel.ConvolutionNet()
    model2.load_state_dict(th.load(rgbModelPath))

    tempModel = LinearModel(model1, model2)
    tempModel.load_state_dict(th.load(modelPath))
    model = ConvolutionModel2(model1, model2, modelPath, tempModel)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.Model.parameters(),
                           lr=config["learning_rate"])

    train_loader, test_loader = createDataLoader(config["grayDataset"], config["rgbDataset"], config)
    n_config = dict(config)
    n_config["architecture"] = "Convolution"
    return model, train_loader, test_loader, criterion, optimizer, n_config

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
                t=TF.vflip(t)
            retuple.append((t,n))
        
        return tuple(retuple)

    def __len__(self):
        return min(len(d) for d in self.datasets)


def createDataLoader(grayPath, rgbPath, config):

    transform1 =  T.Compose([T.Grayscale(num_output_channels=1),
                                     T.ToTensor()])
    transform2 = T.Compose(
        [   
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225))
        ]
     )
    train_loader = th.utils.data.DataLoader(
             ConcatDataset(
                 datasets.ImageFolder(os.path.join(grayPath, "train"), transform=transform1),
                 datasets.ImageFolder(os.path.join(rgbPath, "train"), transform=transform2)
             ),
             batch_size=config["batch_size"], shuffle=True,
             num_workers=4, pin_memory=True)
    test_loader = th.utils.data.DataLoader(
             ConcatDataset(
                 datasets.ImageFolder(os.path.join(grayPath, "valid"), transform=transform1),
                 datasets.ImageFolder(os.path.join(rgbPath, "valid"), transform=transform2)
             ),
             batch_size=config["batch_size"], shuffle=True,
             num_workers=4, pin_memory=True)
    return train_loader, test_loader
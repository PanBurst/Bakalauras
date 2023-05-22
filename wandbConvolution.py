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

import confusionMatrix as confMatrix
from torchmetrics.classification import MulticlassAccuracy

class ConvolutionNet(nn.Module):
    def __init__(self, input_shape=(3, 500, 500)):
        self.ModelPath = "convModel.txt"
        super(ConvolutionNet, self).__init__()
        # color chanel size |image size| kernel size
        self.conv1 = nn.Conv2d(3, 350, 5, 2)
        # image size = (250 - (10 - 1 - 1))/10 +1 = 25
        # image size after pool 25 / 2 = 246
        self.conv2 = nn.Conv2d(350, 250, 3, 2)
        # image size = 246 - 7 + 1 = 240
        # image size after pool 240 / 2 = 120
        self.conv3 = nn.Conv2d(250, 256, 5, 2)

        # image size = 120 - 5 + 1 = 116
        # image size after pool 116 / 2 = 58
        # kernel size | stride size
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(9216, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 102)
        

        self.device = th.device(
            'cuda' if th.cuda.is_available() else 'cpu')
        self.to(self.device)
        return None

    def forward(self, x):
        activationFunction = F.relu
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class ConvolutionModel(_model.Model):
    def __init__(self, modelPath:str, model:ConvolutionNet = None):
        self.Model = ConvolutionNet()
        if model is not None:
            self.Model = model
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.Model.to(self.device)
        self.modelPath = modelPath

    def train(self, loader, criterion, optimizer, config, testLoader=None):
        self.config = dict(config)
        self.config["architecture"] = "Convolution"
        example_ct = 0  # number of examples seen
        batch_ct = 0
        lastAccuracy = 0
        for epoch in tqdm(range(self.config["epochs"])):
            self.Model.train()
            for _, (images, labels) in enumerate(loader):
                images, labels = images.to(self.device), labels.to(self.device)

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
        th.onnx.export(self.Model, images, f"model_{modelName}.onnx")
        wandb.save(f"model_{modelName}.onnx")
        self.SaveModel(f"model_{modelName}.pt")

    def test(self, validLoader, trainLoader = None):
        self.Model.eval()
        validacc = top1, top5 = self.testLoad(validLoader)
        if trainLoader is not None:
            trainacc = top1train, top5train = self.testLoad(trainLoader)
            return validacc, trainacc
        else:
            return validacc

    

    def SaveModel(self, path:str ="model.pt"):
        self.Model.eval()
        th.save(self.Model.state_dict(), path)

    def LogConfusionMatrix(self, epoch, testLoader):
        self.Model.eval()
        predictions = []
        trueLabels = []
        for _, (images, labels) in enumerate(testLoader):
            images, labels = images.to(self.device), labels.to(self.device)
            _, prediction = th.max(self.Model(images).data,1)
            predictions += prediction.tolist()
            trueLabels += labels.tolist()
        allLabels = [x for x in range(1,103)]
        confusionMatrix = confMatrix.ConfusionMatrix(predictions, trueLabels)
        confusionMatrixImage = wandb.Image(confusionMatrix, caption="Klaidų matrica")
        wandb.log({"confusion matrix_basicTransformer": confusionMatrixImage, "epoch": epoch})
        # wandb.log({"my_conf_mat_id" : wandb.plot.confusion_matrix( 
        #             preds=predictions, y_true=trueLabels,
        #             class_names=allLabels)})
    def testLoad(self, loader):
        self.Model.eval()
        top1acc = MulticlassAccuracy(num_classes=102, top_k=1).to(self.device)
        top5acc = MulticlassAccuracy(num_classes=102, top_k=5).to(self.device)
        with th.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.Model(images)
                _ = top1acc.update(outputs, labels)
                _ = top5acc.update(outputs, labels)
            return top1acc.compute(), top5acc.compute()

def Make(config, modelPath:str):
    transform = T.Compose(
        [   
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomApply(th.nn.ModuleList([T.ColorJitter()]), p=0.25),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225))
        ]
     )
    model = ConvolutionModel(modelPath)
    train_data = datasets.ImageFolder(os.path.join(
        config["datasetPath"], "train/"), transform=transform)
    train_loader = DataLoader(
        train_data, batch_size=config["batch_size"], shuffle=True, num_workers=4)

    transform = T.Compose(
        [   T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225))
        ]
     )
    test_data = datasets.ImageFolder(os.path.join(
        config["datasetPath"], "valid"), transform=transform)
    test_loader = DataLoader(
        test_data, batch_size=config["batch_size"], shuffle=True, num_workers=4)

    criterion = LabelSmoothingCrossEntropy()
    optimizer = optim.Adam(model.Model.parameters(),
                           lr=config["learning_rate"])
    n_config = dict(config)
    n_config["architecture"] = "Convolution"
    return model, train_loader, test_loader, criterion, optimizer, n_config

def MakeLoad(config, modelPath:str):

    tempModel = ConvolutionNet()
    tempModel.load_state_dict(th.load(modelPath))
    model = ConvolutionModel("", tempModel)
    transform = T.Compose(
        [   
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomApply(th.nn.ModuleList([T.ColorJitter()]), p=0.25),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225))
        ]
     )
    train_data = datasets.ImageFolder(os.path.join(
        config["datasetPath"], "train/"), transform=transform)
    train_loader = DataLoader(
        train_data, batch_size=config["batch_size"], shuffle=True, num_workers=4)

    transform = T.Compose(
        [   T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225))
        ]
     )
    test_data = datasets.ImageFolder(os.path.join(
        config["datasetPath"], "valid"), transform=transform)
    test_loader = DataLoader(
        test_data, batch_size=config["batch_size"], shuffle=True, num_workers=4)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.Model.parameters(),
                           lr=config["learning_rate"])
    n_config = dict(config)
    n_config["architecture"] = "Convolution"
    return model, train_loader, test_loader, criterion, optimizer, n_config
    
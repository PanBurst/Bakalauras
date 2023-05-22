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

import confusionMatrix as confMatrix
import wandbConvolution as convolutionModel


from torchmetrics.classification import MulticlassAccuracy
class LinearModel(nn.Module):
    def __init__(self, model1, model2, model3, model4):
        super(LinearModel, self).__init__()

        self.Model1 = model1
        self.Model2 = model2
        self.Model3 = model3
        self.Model4 = model4
        self.Layer1 = nn.Linear(102*4, 1024)
        self.Layer2 = nn.Linear(1024, 600)
        self.Layer3 = nn.Linear(600, 512)
        self.Layer4 = nn.Linear(512, 128)
        self.Layer5 = nn.Linear(128, 256)
        self.Layer6 = nn.Linear(256, 512)
        self.Layer7 = nn.Linear(512, 128)
        self.Layer8 = nn.Linear(128, 256)
        self.Layer9 = nn.Linear(256, 512)
        self.Layer10 = nn.Linear(512, 128)
        self.Layer11 = nn.Linear(128, 256)
        self.Layer12 = nn.Linear(256, 102)

    def forward(self, _x):

        x1 = self.Model1(_x)
        x2 = self.Model2(_x)
        x3 = self.Model3(_x)
        x4 = self.Model4(_x)
        x = th.cat((x1, x2, x3, x4), dim=1)
        x = F.relu(self.Layer1(x))
        x = F.relu(self.Layer2(x))
        x = F.relu(self.Layer3(x))
        x = F.relu(self.Layer4(x))
        x = F.relu(self.Layer5(x))
        x = F.relu(self.Layer6(x))
        x = F.relu(self.Layer7(x))
        x = F.relu(self.Layer8(x))
        x = F.relu(self.Layer9(x))
        x = F.relu(self.Layer10(x))
        x = F.relu(self.Layer11(x))

        x = self.Layer12(x)
        return x

class ConvolutionModel2(_model.Model):
    def __init__(self, cnnModel1, cnnModel2, cnnModel3, cnnModel4, modelPath: str):
        super(_model.Model, self).__init__()
        self.Model = LinearModel(cnnModel1, cnnModel2, cnnModel3, cnnModel4)
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
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
            for _, (images, labels) in enumerate(loader):
                images, labels = images.to(self.device), labels.to(self.device)

                # Forward pass ➡
                outputs = self.Model(images)
                outputs.shape
                labels.shape
                _, lossoutputs = th.max(outputs, dim=1)
                lossoutputs.shape
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
        # th.onnx.export(self.Model, images, f"model_{modelName}.onnx")
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
            for _, ((grayImages, _), (rgbImages, labels)) in enumerate(loader):
                grayImages, rgbImages, labels = grayImages.to(self.device), rgbImages.to(self.device), labels.to(self.device)
                images = (grayImages, rgbImages)
                outputs = self.Model(images)
                _ = top1acc.update(outputs, labels)
                _ = top5acc.update(outputs, labels)
            return top1acc.compute(), top5acc.compute()

def Make(config, modelPath, model1Path: str, model2Path: str, model3Path: str, model4Path: str):
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

    model1 = convolutionModel.ConvolutionNet()
    model1.load_state_dict(th.load(model1Path))

    model2 = convolutionModel.ConvolutionNet()
    model2.load_state_dict(th.load(model2Path))

    model3 = convolutionModel.ConvolutionNet()
    model3.load_state_dict(th.load(model3Path))

    model4 = convolutionModel.ConvolutionNet()
    model4.load_state_dict(th.load(model4Path))

    model = ConvolutionModel2(model1, model2, model3, model4, modelPath)
    train_data = datasets.ImageFolder(os.path.join(
        config["datasetPath"], "train/"), transform=transform)
    train_loader = DataLoader(
        train_data, batch_size=config["batch_size"], shuffle=True, num_workers=4)

    transform = T.Compose(
        [T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225))
         ]
    )
    test_data = datasets.ImageFolder(os.path.join(
        config["datasetPath"], "valid"), transform=transform)
    test_loader = DataLoader(
        test_data, batch_size=config["batch_size"], shuffle=True, num_workers=4)

    criterion = timm.loss.SoftTargetCrossEntropy()
    optimizer = optim.Adam(model.Model.parameters(),
                           lr=config["learning_rate"])
    n_config = dict(config)
    n_config["architecture"] = "Convolution2"
    return model, train_loader, test_loader, criterion, optimizer, n_config
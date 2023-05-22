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
import wandbConvolution as convolutionModel


class LinearModel(nn.Module):
    def __init__(self, model1, model2):
        super(LinearModel, self).__init__()

        self.Model1 = model1
        self.Model2 = model2
        self.Layer1 = nn.Linear(102*2, 1024)
        self.Layer2 = nn.Linear(1024, 256)
        self.Layer3 = nn.Linear(256, 512)
        self.Layer4 = nn.Linear(512, 128)
        self.Layer5 = nn.Linear(128, 256)
        self.Layer6 = nn.Linear(256, 102)

    def forward(self, _x):

        x1 = self.Model1(_x)
        x2 = self.Model2(_x)
        x = th.cat((x1, x2), dim=1)
        x = F.relu(self.Layer1(x))
        x = F.relu(self.Layer2(x))
        x = F.relu(self.Layer3(x))
        x = F.relu(self.Layer4(x))
        x = F.relu(self.Layer5(x))

        x = self.Layer6(x)
        return x


class ConvolutionModel2(_model.Model):
    def __init__(self, cnnModel1, cnnModel2, modelPath: str):
        super(_model.Model, self).__init__()
        self.Model = LinearModel(cnnModel1, cnnModel2)
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
                    wandb.log({"epoch": epoch, "loss": loss,
                              "accuracy": lastAccuracy}, step=example_ct)
                    print(
                        f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")

            if testLoader is not None:
                lastAccuracy = self.test(testLoader)
                example_ct += 1
                wandb.log({"epoch": epoch, "loss": loss,
                          "accuracy": lastAccuracy}, step=example_ct)

                self.LogConfusionMatrix(epoch, testLoader)

        modelName = self.modelPath.split(".")[0]
        th.onnx.export(self.Model, images, f"model_{modelName}.onnx")
        wandb.save("model.onnx")
        self.SaveModel(f"model_{modelName}.pt")

    def test(self, loader):
        self.Model.eval()
        with th.no_grad():
            correct, total = 0, 0
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.Model(images)
                _, predicted = th.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            self.SaveModel(self.modelPath)
            # print(f"Accuracy of the model on the {total} " +
            #       f"test images: {correct / total:%}")
            # Save the model in the exchangeable ONNX format
            #th.onnx.export(self.Model, images, "model.onnx ")
            
            return correct / total

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


def Make(config, modelPath, model1Path: str, model2Path: str):
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

    model = ConvolutionModel2(model1, model2, modelPath)
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

    criterion = LabelSmoothingCrossEntropy()
    optimizer = optim.Adam(model.Model.parameters(),
                           lr=config["learning_rate"])
    n_config = dict(config)
    n_config["architecture"] = "Convolution2"
    return model, train_loader, test_loader, criterion, optimizer, n_config

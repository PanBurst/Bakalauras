import torch as th
import torch.nn as nn
import Model as _model
import torch.nn.functional as F
from torchvision import transforms as T  # for simplifying the transforms
from torch import nn, optim
from tqdm import tqdm
from torchvision import datasets
from timm.loss import LabelSmoothingCrossEntropy
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, sampler, random_split
from tqdm import tqdm
import os
import wandb

import confusionMatrix as confMatrix


from torchmetrics.classification import MulticlassAccuracy

import torchvision.transforms.functional as TF



import ResNet as RN



class ResNet:
    def __init__(self, modelName) -> None:
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.modelPath = modelName
        self.Model = RN.ResNet101(102, 1)
    def train(self, loader, criterion, optimizer, config , testLoader=None):
        self.config = dict(config)
        self.config["architecture"] = "Convolution"
        example_ct = 0  # number of examples seen
        batch_ct = 0
        lastAccuracy = 0
        self.Model.to(self.device)        
        self.Model.train()
        for epoch in tqdm(range(self.config["epochs"])):
            losses = []
            if epoch%2 ==0:
                modelName = self.modelPath.split(".")[0]
                self.SaveModel(f"model_{modelName}{epoch}.pt")
            for _, (images, labels) in enumerate(loader):
                self.Model.train()
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                # Forward pass ➡
                outputs = self.Model(images)
                loss = criterion(outputs, labels)
                # Backward pass ⬅
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
            
        
        # modelName = self.modelPath.split(".")[0]
        # # th.onnx.export(self.Model, images, f"model_{modelName}.onnx")
        # # wandb.save(f"model_{modelName}.onnx")
        # self.SaveModel(f"model_{modelName}.pt")

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
    transform = T.Compose([T.Grayscale(num_output_channels=1),
    T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
                                     T.ToTensor()])
    model = ResNet(modelPath)
    train_data = datasets.ImageFolder(os.path.join(
        config["datasetPath"], "train/"), transform=transform)
    train_loader = DataLoader(
        train_data, batch_size=config["batch_size"], shuffle=True, num_workers=4)

    transform = T.Compose([T.Grayscale(num_output_channels=1),
                                     T.ToTensor()])
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
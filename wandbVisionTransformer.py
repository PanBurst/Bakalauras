import torch as th
import torch.nn as nn
import Model as _model
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from torchvision import datasets
from torch.utils.data import DataLoader, sampler, random_split
from torchvision import transforms as T  # for simplifying the transforms
import os
from timm.loss import LabelSmoothingCrossEntropy
from torch import nn, optim
import json
import confusionMatrix as confMatrix
from torchmetrics.classification import MulticlassAccuracy

class VisionTransformerModel(_model.Model):
    def __init__(self, modelPath:str = None):
        model = th.hub.load('facebookresearch/deit:main',
                            'deit_tiny_patch16_224', pretrained=True)
        for param in model.parameters():  # freeze model
            param.requires_grad = False
        n_inputs = model.head.in_features
        model.head = nn.Sequential(
            nn.Linear(n_inputs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512,1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 102)
        )
        self.Model = model
        self.modelPath = modelPath
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.Model.to(self.device)

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
        # th.onnx.export(self.Model, images, f"model_{modelName}.onnx")
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
    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomApply(th.nn.ModuleList([T.ColorJitter()]), p=0.25),
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225)),  # imagenet means
        T.RandomErasing(p=0.2, value='random')
    ])
    model = VisionTransformerModel(modelPath)
    train_data = datasets.ImageFolder(os.path.join(
        config["datasetPath"], "train/"), transform=transform)
    train_loader = DataLoader(
        train_data, batch_size=config["batch_size"], shuffle=True, num_workers=4)

    transform = T.Compose([  # We dont need augmentation for test transforms
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225)),  # imagenet means
    ])
    test_data = datasets.ImageFolder(os.path.join(
        config["datasetPath"], "valid"), transform=transform)
    test_loader = DataLoader(
        test_data, batch_size=config["batch_size"], shuffle=True, num_workers=4)

    criterion = LabelSmoothingCrossEntropy()
    optimizer = optim.Adam(model.Model.head.parameters(),
                           lr=config["learning_rate"])
    n_config = dict(config)
    n_config["architecture"] = "Transformer"
    return model, train_loader, test_loader, criterion, optimizer, n_config


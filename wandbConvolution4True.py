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
import wandbConvolution as convolutionModel
import wandbConvolutionTrue as grayModel

import torchvision.transforms.functional as TF

from torchmetrics.classification import MulticlassAccuracy

class LinearModel(nn.Module):
    def __init__(self, rgbModel, edgeModel, skeletonModel, LBPmodel):
        super(LinearModel, self).__init__()

        self.rgbModel = rgbModel
        self.edgeModel = edgeModel
        self.skeletonModel = skeletonModel
        self.LBPmodel = LBPmodel
        self.Layer1 = nn.Linear(102*4, 1024)
        self.Layer2 = nn.Linear(1024, 512)
        self.Layer3 = nn.Linear(512, 102)

    def forward(self, _x):

        _rgb, _edge, _skeleton, _lbp = _x
        x1 = self.rgbModel(_rgb)
        x2 = self.edgeModel(_edge)
        x3 = self.skeletonModel(_skeleton)
        x4 = self.LBPmodel(_lbp)
        x = th.cat((x1, x2, x3, x4), dim=1)
        x = F.relu(self.Layer1(x))
        x = F.relu(self.Layer2(x))

        x = self.Layer3(x)
        return x

class ConvolutionModel2(_model.Model):
    def __init__(self, rgbModel, edgeModel, skeletonModel, LBPmodel, modelPath: str, model:LinearModel = None):
        super(_model.Model, self).__init__()
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        rgbModel.to(self.device)
        edgeModel.to(self.device)
        skeletonModel.to(self.device)
        LBPmodel.to(self.device)
        self.Model = LinearModel(rgbModel, edgeModel, skeletonModel, LBPmodel)
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
            for _, ((rgbImages, _), (edgeImages, __), (skeltonImages, ___), (lbpImages, labels)) in enumerate(loader):
                rgbImages, edgeImages, skeltonImages, lbpImages, labels = rgbImages.to(self.device), edgeImages.to(self.device), skeltonImages.to(self.device), lbpImages.to(self.device), labels.to(self.device)
                images = (rgbImages, edgeImages, skeltonImages, lbpImages)
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
        dummy_input = th.randn(1, 6, 500, 500).cuda()
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
        for _, ((rgbImages, _), (edgeImages, __), (skeltonImages, ___), (lbpImages, labels)) in enumerate(testLoader):
            rgbImages, edgeImages, skeltonImages, lbpImages, labels = rgbImages.to(self.device), edgeImages.to(self.device), skeltonImages.to(self.device), lbpImages.to(self.device), labels.to(self.device)
            images = (rgbImages, edgeImages, skeltonImages, lbpImages)
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
            for _, ((rgbImages, _), (edgeImages, __), (skeltonImages, ___), (lbpImages, labels)) in enumerate(loader):
                rgbImages, edgeImages, skeltonImages, lbpImages, labels = rgbImages.to(self.device), edgeImages.to(self.device), skeltonImages.to(self.device), lbpImages.to(self.device), labels.to(self.device)
                images = (rgbImages, edgeImages, skeltonImages, lbpImages)
                outputs = self.Model(images)
                _ = top1acc.update(outputs, labels)
                _ = top5acc.update(outputs, labels)
            return top1acc.compute(), top5acc.compute()
    

    def testClasses(self, loader: DataLoader):
        self.Model.eval()
        predictions = []
        trueLabels = []
        with th.no_grad():
            for _, ((rgbImages, _), (edgeImages, __), (skeltonImages, ___), (lbpImages, labels)) in enumerate(loader):
                rgbImages, edgeImages, skeltonImages, lbpImages, labels = rgbImages.to(self.device), edgeImages.to(self.device), skeltonImages.to(self.device), lbpImages.to(self.device), labels.to(self.device)
                images = (rgbImages, edgeImages, skeltonImages, lbpImages)
                _, prediction = th.max(self.Model(images).data, 1)
                predictions += prediction.tolist()
                trueLabels += labels.tolist()
        totalLabels = [0]*102
        totalPredictedLabels = [0]*102
        totalPredictedLabelProb = [0]*102
        for prediction, label in zip(predictions, trueLabels):
            totalLabels[label] += 1
            if prediction == label:
                totalPredictedLabels[prediction] += 1
        for idx, (prediction, label) in enumerate(zip(totalPredictedLabels, totalPredictedLabelProb)):
            totalPredictedLabelProb[idx] = totalPredictedLabels[idx] / \
                totalLabels[idx]
        totalPredictedLabels = [(idx, value)
                                for idx, value in enumerate(totalPredictedLabelProb)]
        totalPredictedLabels.sort(key=lambda x: x[1])
        totalPredictedLabels = totalPredictedLabels[:5]
        return totalPredictedLabels

    def showTop5(self, dataloader: DataLoader):
        self.Model.eval()
        worstClasses = self.testClasses(dataloader)
        _images = []
        classification = {
            "label": None,
            "images": [],
            "top5": []
        }
        classificationResults = []
        classificationResults.append(dict(classification))
        classificationResults.append(dict(classification))
        classificationResults.append(dict(classification))
        classificationResults.append(dict(classification))
        classificationResults.append(dict(classification))
        classificationResults[0]["top5"] = []
        classificationResults[1]["top5"] = []
        classificationResults[2]["top5"] = []
        classificationResults[3]["top5"] = []
        classificationResults[4]["top5"] = []
        for idx, classificationResult in enumerate(classificationResults):
            classificationResult["label"] = worstClasses[idx][0]

        predictions = []
        with th.no_grad():
            for _, ((rgbImages, _), (edgeImages, __), (skeltonImages, ___), (lbpImages, labels)) in enumerate(dataloader):
                rgbImages, edgeImages, skeltonImages, lbpImages, labels = rgbImages.to(self.device), edgeImages.to(
                    self.device), skeltonImages.to(self.device), lbpImages.to(self.device), labels.to(self.device)
                images = (rgbImages, edgeImages, skeltonImages, lbpImages)
                _images.append(images)
                _, topPredictions = th.topk(self.Model(images).data, 5)
                realimages = rgbImages.tolist()
                predictions += topPredictions.tolist()
                for idx, label in enumerate(labels):
                    trueLabel = label.item()
                    if 58 == trueLabel:
                        a = 'a'
                    for classificationResult in classificationResults:
                        if classificationResult["label"] == label.item():
                            classificationResult["images"].append(realimages[idx])
                            classificationResult["top5"].append(predictions[idx])
            
        return classificationResults
    
def Make(config, modelPath, rgbModelPath: str, edgeModelPath: str, skeletonModelPath: str, lbpModelPath: str, rgbDataPath:str, edgeDataPath:str, skeletonDataPath:str, lbpDataPath:str):
    transform = T.Compose(
        [
            # T.RandomHorizontalFlip(),
            # T.RandomVerticalFlip(),
            # T.RandomApply(th.nn.ModuleList([T.ColorJitter()]), p=0.25),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225))
        ]
    )

    rgbModel = convolutionModel.ConvolutionNet()
    rgbModel.load_state_dict(th.load(rgbModelPath))

    edgeModel = grayModel.ConvolutionNet()
    edgeModel.load_state_dict(th.load(edgeModelPath))

    skeletonModel = grayModel.ConvolutionNet()
    skeletonModel.load_state_dict(th.load(skeletonModelPath))

    lbpModel = grayModel.ConvolutionNet()
    lbpModel.load_state_dict(th.load(lbpModelPath))

    model = ConvolutionModel2(rgbModel, edgeModel, skeletonModel, lbpModel, modelPath)
    

    train_loader, test_loader = createDataLoader(rgbDataPath, edgeDataPath, skeletonDataPath, lbpDataPath, config)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.Model.parameters(),
                           lr=config["learning_rate"])
    n_config = dict(config)
    n_config["architecture"] = "Convolution2"
    return model, train_loader, test_loader, criterion, optimizer, n_config

def MakeLoad(config, modelPath, rgbModelPath: str, edgeModelPath: str, skeletonModelPath: str, lbpModelPath: str, rgbDataPath:str, edgeDataPath:str, skeletonDataPath:str, lbpDataPath:str):
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225))
        ]
    )

    rgbModel = convolutionModel.ConvolutionNet()
    rgbModel.load_state_dict(th.load(rgbModelPath))

    edgeModel = grayModel.ConvolutionNet()
    edgeModel.load_state_dict(th.load(edgeModelPath))

    skeletonModel = grayModel.ConvolutionNet()
    skeletonModel.load_state_dict(th.load(skeletonModelPath))

    lbpModel = grayModel.ConvolutionNet()
    lbpModel.load_state_dict(th.load(lbpModelPath))

    tempModel = LinearModel(rgbModel, edgeModel, skeletonModel, lbpModel)
    tempModel.load_state_dict(th.load(modelPath))

    model = ConvolutionModel2(rgbModel, edgeModel, skeletonModel, lbpModel, modelPath, tempModel)
    
    train_loader, test_loader = createDataLoader(rgbDataPath, edgeDataPath, skeletonDataPath, lbpDataPath, config)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.Model.parameters(),
                           lr=config["learning_rate"])
    n_config = dict(config)
    n_config["architecture"] = "Convolution2"
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


def createDataLoader(rgbPath, edgePath, skeletonPath, LBPpath, config):

    transform1 = T.Compose(
        [   
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225))
        ]
     )
    transform2 =  T.Compose([T.Grayscale(num_output_channels=1),
                                     T.ToTensor()])
    
    train_loader = th.utils.data.DataLoader(
             ConcatDataset(
                 datasets.ImageFolder(os.path.join(rgbPath, "train"), transform=transform1),
                 datasets.ImageFolder(os.path.join(edgePath, "train"), transform=transform2),
                 datasets.ImageFolder(os.path.join(skeletonPath, "train"), transform=transform2),
                 datasets.ImageFolder(os.path.join(LBPpath, "train"), transform=transform2)
             ),
             batch_size=config["batch_size"], shuffle=True,
             num_workers=4, pin_memory=True)
    test_loader = th.utils.data.DataLoader(
             ConcatDataset(
                 datasets.ImageFolder(os.path.join(rgbPath, "valid"), transform=transform1),
                 datasets.ImageFolder(os.path.join(edgePath, "valid"), transform=transform2),
                 datasets.ImageFolder(os.path.join(skeletonPath, "valid"), transform=transform2),
                 datasets.ImageFolder(os.path.join(LBPpath, "valid"), transform=transform2)
             ),
             batch_size=config["batch_size"], shuffle=True,
             num_workers=0, pin_memory=True)
    return train_loader, test_loader
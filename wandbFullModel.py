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

import PCA_model
import HOG_model

class LinearModel(nn.Module):
    def __init__(self, rgbModel, edgeModel, skeletonModel, LBPmodel, HOGmodel, PCAmodel):
        super(LinearModel, self).__init__()
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.rgbModel = rgbModel
        self.edgeModel = edgeModel
        self.skeletonModel = skeletonModel
        self.LBPmodel = LBPmodel
        self.HOGmodel = HOGmodel
        self.PCAmodel = PCAmodel
        self.Layer1 = nn.Linear(102*6, 1024)
        self.Layer2 = nn.Linear(1024, 512)
        self.Layer3 = nn.Linear(512, 102)

        self.dropout = nn.Dropout(0.25)
        self.softMax = nn.Softmax(dim=1)

    def forward(self, _x):

        _rgb, _edge, _skeleton, _lbp, _hog, _pca = _x
        x1 = self.softMax(self.rgbModel(_rgb))
        x2 = self.softMax(self.edgeModel(_edge))
        x3 = self.softMax(self.skeletonModel(_skeleton))
        x4 = self.softMax(self.LBPmodel(_lbp))
        x5 = self.HOGmodel.predict(_hog)
        x6 = self.PCAmodel.predict(_pca)
        tensorX5 = th.tensor(x5, device=self.device, dtype=th.float)
        tensorX6 = th.tensor(x6, device=self.device, dtype=th.float)
        x = th.cat((x1, x2, x3, x4, tensorX5, tensorX6), dim=1)
        x = F.relu(self.Layer1(x))
        x = self.dropout(x)
        x = F.relu(self.Layer2(x))
        x = self.dropout(x)

        x = self.Layer3(x)
        return x

class ConvolutionModel2(_model.Model):
    def __init__(self, rgbModel, edgeModel, skeletonModel, LBPmodel, HOGmodel, PCAmodel, modelPath: str, model:LinearModel = None):
        super(_model.Model, self).__init__()
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        rgbModel.to(self.device)
        edgeModel.to(self.device)
        skeletonModel.to(self.device)
        LBPmodel.to(self.device)
        self.Model = LinearModel(rgbModel, edgeModel, skeletonModel, LBPmodel, HOGmodel, PCAmodel)
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
            for _, data in enumerate(loader):
                ((rgbImages, _), (edgeImages, __), (skeltonImages, ___), (lbpImages, labels), (hogImageData, ____), (pcaImageData, _____)) = data
                rgbImages, edgeImages, skeltonImages, lbpImages, labels = rgbImages.to(self.device), edgeImages.to(self.device), skeltonImages.to(self.device), lbpImages.to(self.device), labels.to(self.device)
                imageData = (rgbImages, edgeImages, skeltonImages, lbpImages, hogImageData, pcaImageData)
                # Forward pass ➡
                outputs = self.Model(imageData)
                loss = criterion(outputs, labels)

                # Backward pass ⬅
                optimizer.zero_grad()
                loss.backward()

                # Step with optimizer
                optimizer.step()

                example_ct += len(imageData)
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
        for _, ((rgbImages, _), (edgeImages, __), (skeltonImages, ___), (lbpImages, labels), (hogImageData, ____), (pcaImageData, _____)) in enumerate(testLoader):
            rgbImages, edgeImages, skeltonImages, lbpImages, labels = rgbImages.to(self.device), edgeImages.to(self.device), skeltonImages.to(self.device), lbpImages.to(self.device), labels.to(self.device)
            imageData = (rgbImages, edgeImages, skeltonImages, lbpImages, hogImageData, pcaImageData)
            _, prediction = th.max(self.Model(imageData).data, 1)
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
            for _, ((rgbImages, _), (edgeImages, __), (skeltonImages, ___), (lbpImages, labels), (hogImageData, ____), (pcaImageData, _____)) in enumerate(loader):
                rgbImages, edgeImages, skeltonImages, lbpImages, labels = rgbImages.to(self.device), edgeImages.to(self.device), skeltonImages.to(self.device), lbpImages.to(self.device), labels.to(self.device)
                imageData = (rgbImages, edgeImages, skeltonImages, lbpImages, hogImageData, pcaImageData)
                outputs = self.Model(imageData)
                _ = top1acc.update(outputs, labels)
                _ = top5acc.update(outputs, labels)
            return top1acc.compute(), top5acc.compute()
    

    def testClasses(self, loader: DataLoader):
        self.Model.eval()
        predictions = []
        trueLabels = []
        with th.no_grad():
            for _, ((rgbImages, _), (edgeImages, __), (skeltonImages, ___), (lbpImages, labels), (hogImageData, ____), (pcaImageData, _____)) in enumerate(loader):
                rgbImages, edgeImages, skeltonImages, lbpImages, labels = rgbImages.to(self.device), edgeImages.to(self.device), skeltonImages.to(self.device), lbpImages.to(self.device), labels.to(self.device)
                imageData = (rgbImages, edgeImages, skeltonImages, lbpImages, hogImageData, pcaImageData)
                _, prediction = th.max(self.Model(imageData).data, 1)
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
            for _, ((rgbImages, _), (edgeImages, __), (skeltonImages, ___), (lbpImages, labels), (hogImageData, ____), (pcaImageData, _____)) in enumerate(dataloader):
                rgbImages, edgeImages, skeltonImages, lbpImages, labels = rgbImages.to(self.device), edgeImages.to(self.device), skeltonImages.to(self.device), lbpImages.to(self.device), labels.to(self.device)
                imageData = (rgbImages, edgeImages, skeltonImages, lbpImages, hogImageData, pcaImageData)
                _images.append(imageData)
                _, topPredictions = th.topk(self.Model(imageData).data, 5)
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

    pcaModel = PCA_model.PCA_model("pcaModed.joblib")
    PCA_datasets = pcaModel.Make(n_components=200)
    print("Loaded PCA model")

    hogModel = HOG_model.HOG_model("hogModel.joblib")
    HOG_datasets = hogModel.Make((32,32),(4,4),8)
    print("Loaded HOG model")


    model = ConvolutionModel2(rgbModel, edgeModel, skeletonModel, lbpModel, hogModel, pcaModel, modelPath)
    
    train_loader, test_loader = createDataLoader(rgbDataPath, edgeDataPath, skeletonDataPath, lbpDataPath, HOG_datasets, PCA_datasets, config)

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

    pcaModel = PCA_model.PCA_model("pcaModed.joblib")
    PCA_datasets = pcaModel.Make(n_components=1000)

    hogModel = HOG_model.HOG_model("hogModel.joblib")
    HOG_datasets = hogModel.Make((32,32),(4,4),8)



    model = ConvolutionModel2(rgbModel, edgeModel, skeletonModel, lbpModel, modelPath, tempModel)
    
    train_loader, test_loader = createDataLoader(rgbDataPath, edgeDataPath, skeletonDataPath, lbpDataPath, HOG_datasets, PCA_datasets, config)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.Model.parameters(),
                           lr=config["learning_rate"])
    n_config = dict(config)
    n_config["architecture"] = "AllConvolutionsHOG_PCA"
    return model, train_loader, test_loader, criterion, optimizer, n_config

class ConcatDataset(th.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        # verticalFlip = random.random() < 0.5
        # horizontalFlip = random.random() < 0.5
        
        retuple = []
        for dataset in self.datasets:
            image, label = dataset[i]
            # if horizontalFlip:
            #     t = TF.hflip(t)
            # if verticalFlip:
            #     t=TF.vflip(t)
            retuple.append((image,label))
        
        return tuple(retuple)

    def __len__(self):
        return min(len(d) for d in self.datasets)


def createDataLoader(rgbPath, edgePath, skeletonPath, LBPpath, HOG_datasets, PCA_datasets, config):

    transform1 = T.Compose(
        [   
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225))
        ]
     )
    transform2 =  T.Compose([T.Grayscale(num_output_channels=1),
                                     T.ToTensor()])
    
    hog_train_dataset, hog_test_dataset = HOG_datasets
    pca_train_dataset, pca_test_dataset = PCA_datasets

    train_loader = th.utils.data.DataLoader(
             ConcatDataset(
                 datasets.ImageFolder(os.path.join(rgbPath, "train"), transform=transform1),
                 datasets.ImageFolder(os.path.join(edgePath, "train"), transform=transform2),
                 datasets.ImageFolder(os.path.join(skeletonPath, "train"), transform=transform2),
                 datasets.ImageFolder(os.path.join(LBPpath, "train"), transform=transform2),

                 hog_train_dataset,
                 pca_train_dataset

             ),
             batch_size=config["batch_size"], shuffle=True,
             num_workers=4, pin_memory=True)
    test_loader = th.utils.data.DataLoader(
             ConcatDataset(
                 datasets.ImageFolder(os.path.join(rgbPath, "valid"), transform=transform1),
                 datasets.ImageFolder(os.path.join(edgePath, "valid"), transform=transform2),
                 datasets.ImageFolder(os.path.join(skeletonPath, "valid"), transform=transform2),
                 datasets.ImageFolder(os.path.join(LBPpath, "valid"), transform=transform2),

                 hog_test_dataset,
                 pca_test_dataset
             ),
             batch_size=config["batch_size"], shuffle=True,
             num_workers=4, pin_memory=True)
    return train_loader, test_loader
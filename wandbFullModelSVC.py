from torchvision import datasets
import torch as th

import os
import numpy as np
import cv2


from sklearn.svm import SVC


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
from sklearn.metrics import top_k_accuracy_score

import PCA_model
import HOG_model


class AllModelSVM:
    def __init__(self, rgbModel, edgeModel, skeletonModel, LBPmodel, HOGmodel, PCAmodel) -> None:
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.rgbModel = rgbModel
        self.edgeModel = edgeModel
        self.skeletonModel = skeletonModel
        self.LBPmodel = LBPmodel

        self.rgbModel.to(self.device)
        self.edgeModel.to(self.device)
        self.skeletonModel.to(self.device)
        self.LBPmodel.to(self.device)

        self.HOGmodel = HOGmodel
        self.PCAmodel = PCAmodel
        self.softMax = nn.Softmax(dim=1)

    def train(self, loader)->None:
        dataPredictions = []
        dataLabels = []
        for _, data in enumerate(loader):
            ((rgbImages, _), (edgeImages, __), (skeltonImages, ___), (lbpImages, labels), (hogImageData, ____), (pcaImageData, _____)) = data
            rgbImages, edgeImages, skeltonImages, lbpImages, labels = rgbImages.to(self.device), edgeImages.to(self.device), skeltonImages.to(self.device), lbpImages.to(self.device), labels.to(self.device)
            rgbPredictions = self.softMax(self.rgbModel(rgbImages).cpu()).detach().numpy()
            edgePredictions = self.softMax(self.edgeModel(edgeImages)).cpu().detach().numpy()
            skeletonPredictions = self.softMax(self.skeletonModel(skeltonImages)).cpu().detach().numpy()
            lbpPredictions = self.softMax(self.LBPmodel(lbpImages).cpu()).detach().numpy()
            hogPredictions = self.HOGmodel.predict(hogImageData)
            pcaPredictions = self.PCAmodel.predict(pcaImageData)
            
            predictionValues = rgbPredictions
            predictionValues = list(map(lambda x,y: np.append(x, y), predictionValues, edgePredictions))
            predictionValues = list(map(lambda x,y: np.append(x, y), predictionValues, skeletonPredictions))
            predictionValues = list(map(lambda x,y: np.append(x, y), predictionValues, lbpPredictions))
            predictionValues = list(map(lambda x,y: np.append(x, y), predictionValues, hogPredictions))
            predictionValues = list(map(lambda x,y: np.append(x, y), predictionValues, pcaPredictions))

            dataPredictions.extend(predictionValues)
            dataLabels.extend(labels.cpu().detach().numpy())

        SVC(class_weight='balanced')
        self.clf = SVC(kernel='rbf', class_weight='balanced', probability=True)
        self.clf.fit(dataPredictions, dataLabels)



    def testLoader(self, loader)->None:
        top1acc = MulticlassAccuracy(num_classes=102, top_k=1)
        top5acc = MulticlassAccuracy(num_classes=102, top_k=5)
        with th.no_grad():

            dataPredictions = []
            dataLabels = []
            for _, ((rgbImages, _), (edgeImages, __), (skeltonImages, ___), (lbpImages, labels), (hogImageData, ____), (pcaImageData, _____)) in enumerate(loader):
                rgbImages, edgeImages, skeltonImages, lbpImages, labels = rgbImages.to(self.device), edgeImages.to(self.device), skeltonImages.to(self.device), lbpImages.to(self.device), labels.to(self.device)
                imageData = (rgbImages, edgeImages, skeltonImages, lbpImages, hogImageData, pcaImageData)

                rgbPredictions = self.softMax(self.rgbModel(rgbImages).cpu()).detach().numpy()
                edgePredictions = self.softMax(self.edgeModel(edgeImages)).cpu().detach().numpy()
                skeletonPredictions = self.softMax(self.skeletonModel(skeltonImages)).cpu().detach().numpy()
                lbpPredictions = self.softMax(self.LBPmodel(lbpImages).cpu()).detach().numpy()
                hogPredictions = self.HOGmodel.predict(hogImageData)
                pcaPredictions = self.PCAmodel.predict(pcaImageData)
                
                predictionValues = rgbPredictions
                predictionValues = list(map(lambda x,y: np.append(x, y), predictionValues, edgePredictions))
                predictionValues = list(map(lambda x,y: np.append(x, y), predictionValues, skeletonPredictions))
                predictionValues = list(map(lambda x,y: np.append(x, y), predictionValues, lbpPredictions))
                predictionValues = list(map(lambda x,y: np.append(x, y), predictionValues, hogPredictions))
                predictionValues = list(map(lambda x,y: np.append(x, y), predictionValues, pcaPredictions))

                dataPredictions.extend(predictionValues)
                dataLabels.extend(labels.cpu().detach().numpy())

            outputs = self.clf.predict_proba(dataPredictions)
                

            
            return top_k_accuracy_score(dataLabels, outputs, k=1), top_k_accuracy_score(dataLabels, outputs, k=5)

    def test(self, trainLoader, testLoader):
        trainTop1, trainTop5 = self.testLoader(trainLoader)
        testTop1, testTop5 = self.testLoader(testLoader)

        wandb.log({"accuracy": testTop1, "top 1 validation": testTop1, "top 5 validation": testTop5, "top 1 train": trainTop1, "top 5 train" : trainTop5}, step=1)

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


    model = AllModelSVM(rgbModel, edgeModel, skeletonModel, lbpModel, hogModel, pcaModel)
    
    train_loader, test_loader = createDataLoader(rgbDataPath, edgeDataPath, skeletonDataPath, lbpDataPath, HOG_datasets, PCA_datasets, config)


    n_config = dict(config)
    n_config["architecture"] = "SVC"
    return model, train_loader, test_loader, n_config


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
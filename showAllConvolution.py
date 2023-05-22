
import wandb
import wandbVisionTransformer as transfomer
import wandbConvolution as convolution
import Model as _model
import wandbConvolution2 as convolution2
import wandbConvolution4 as convolution4

import wandbConvolutionTrue as convolutionGray
import wandbConvolution2True as convolutionRGBGray
import wandbConvolution4True as convolution4True
import torch

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
import os
import io
models = {
        "RGB" :  "model_ConvolutionNoFilter.pt",
        "grayEdges" : "model_ConvolutionEdgesTrue.pt",
        "graySkeleton" : "model_ConvolutionSkeletonTrue.pt",
        "grayLBP" : "model_ConvolutionLBPTrue.pt"
    }
dataPaths = {
    "RGB" :  "./Data/flower_data",
    "grayEdges" : "./DataEdgesCannyTrue/flower_data",
    "graySkeleton" : "./DataSkeletonTrue/flower_data",
    "grayLBP" : "./DataLBPTrue/flower_data"
}
testConfig = dict(_model.config)
testConfig["dataset"] = "Oxford 102 Skeleton"
testConfig["epochs"] = 51
testConfig["learning_rate"] = 0.00001
testConfig["datasetPath"] = ""

runName = "model_ConvolutionAllTrue"
model, train_loader, test_loader, criterion, optimizer, config = convolution4True.MakeLoad(
    testConfig, f"{runName}.pt",
    rgbModelPath=models["RGB"],
    edgeModelPath=models["grayEdges"],
    skeletonModelPath=models["graySkeleton"],
    lbpModelPath=models["grayLBP"],
    rgbDataPath=dataPaths["RGB"],
    edgeDataPath=dataPaths["grayEdges"],
    skeletonDataPath=dataPaths["graySkeleton"],
    lbpDataPath=dataPaths["grayLBP"],
)
model.Model.load_state_dict(torch.load(f"{runName}.pt"))
classes = model.showTop5(test_loader)

def getClassName(id:int)->str:
    return labels[str(id)]

imageFolderPath = r"C:\Users\stosk\Desktop\KursinisProjektas\Kodas\Data\flower_data\valid"
file = open("cat_to_name.json")
labels = json.load(file)
# for _class in classes:
#     classLabel = _class["label"]
#     top5Labels = _class["top5"]
#     images = _class["images"]
#     top5s = [0]*102
#     for top5Label in top5Labels:
#         for top5 in top5Label:
#             top5s[int(top5)-1] += 1
#     top5sIdx = [(idx, count) for idx, count in enumerate(top5s)]
#     top5sIdx.sort(key=lambda x: x[1])
#     top5sIdx = top5sIdx[:-1:4]
#     for top5Classes in top5sIdx:
#         imageNames = os.listdir(imageFolderPath+"\\"+str(top5Classes[1]+1))
#         f, axarr = plt.subplots(1,6)
#         baseImg = images[0]
#         axarr[0,0].imshow(baseImg)
#         axarr[0,0].set_title(getClassName(classLabel+1))
#         for idx, imageName in enumerate(imageNames):
#             img = mpimg.imread(imageFolderPath+"\\"+str(top5Classes[1]+1)+"\\"+imageName)

#             axarr[0,idx+1].set_title(getClassName(top5Classes[1]+1))
#             axarr[0,idx+1].imshow(img)
#         plt.show()
#     plt.title("")

for _class in classes:
    classLabel = _class["label"]
    top5Labels = _class["top5"]
    images = _class["images"]

    f, axarr = plt.subplots(1,6)
    classImage = os.listdir(imageFolderPath+"\\"+str(classLabel))[0]
    baseImg = img = mpimg.imread(imageFolderPath+"\\"+str(classLabel)+"\\"+classImage)
    axarr[0].imshow(baseImg)
    axarr[0].set_title(getClassName(classLabel))
    for idx, topLabel in enumerate(top5Labels[0]):
        imageName = os.listdir(imageFolderPath+"\\"+str(topLabel))[0]
        img = mpimg.imread(imageFolderPath+"\\"+str(topLabel)+"\\"+imageName)
        axarr[idx+1].set_title(getClassName(topLabel))
        axarr[idx+1].imshow(img)
    plt.show()
    plt.title("")
    

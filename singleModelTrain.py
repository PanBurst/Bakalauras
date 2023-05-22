import wandb
import wandbVisionTransformer as transfomer
import wandbConvolution as convolution
import Model as _model
import wandbConvolution2 as convolution2
import wandbConvolution4 as convolution4

import wandbConvolutionTrue as convolutionGray
import wandbConvolution2True as convolutionRGBGray
import wandbConvolution4True as convolution4True

import wandbLinearModel_GLCM_LBP as linearModel

import wandbFullModel as FullModel

import wandbResNetPreLoad as RNFull

if __name__ == '__main__':
    wandb.login()
    # testConfig = dict(_model.config)
    # testConfig["dataset"] = "Oxford 102"
    # testConfig["epochs"] = 16
    # testConfig["learning_rate"] = 0.0001
    # testConfig["datasetPath"] = "./Data/flower_data"
    # model, train_loader, test_loader, criterion, optimizer, config = convolution.Make(
    #     testConfig, "ConvolutionNoFilter.pth")
    # with wandb.init(project="KursinisProjektas", name="ConvolutionNoFilter", config=config):
    #     config = wandb.config
    #     model, train_loader, test_loader, criterion, optimizer, config = convolution.Make(
    #         config, "ConvolutionNoFilter.pth")
    #     model.train(train_loader, criterion, optimizer, config, test_loader)
    #     model.test(test_loader)
    models = {
        "RGB" :  "model_ResNet48.pt",
        "grayEdges" : "model_ResNetEdges48.pt",
        "graySkeleton" : "model_ResNetSkeleton48.pt",
        "grayLBP" : "model_ResNetLBP48.pt"
    }
    dataPaths = {
        "RGB" :  "./Data/flower_data",
        "grayEdges" : "./DataEdgesCannyTrue/flower_data",
        "graySkeleton" : "./DataSkeletonTrue/flower_data",
        "grayLBP" : "./DataLBPTrue/flower_data"
    }
    testConfig = dict(_model.config)
    testConfig["dataset"] = "AllDatasets"
    testConfig["epochs"] = 9
    testConfig["learning_rate"] = 0.001
    testConfig["datasetPath"] = ""

    runName = "ResNet Ensemble"

    with wandb.init(project="KursinisProjektas", name=runName, config=testConfig):
        config = wandb.config
        model, train_loader, test_loader, criterion, optimizer, config = RNFull.Make(
            testConfig, f"{runName}.pt",
            rgbModelPath=models["RGB"],
            edgeModelPath=models["grayEdges"],
            skeletonModelPath=models["graySkeleton"],
            lbpModelPath=models["grayLBP"],
            rgbDataPath=dataPaths["RGB"],
            edgeDataPath=dataPaths["grayEdges"],
            skeletonDataPath=dataPaths["graySkeleton"],
            lbpDataPath=dataPaths["grayLBP"]
        )
        model.train(train_loader, criterion, optimizer, config, test_loader)
        model.test(test_loader)
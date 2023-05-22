
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

import wandbFullModelSVC as FullModelSVC
import wandbResNet as ResNet

import wandbResNetGray as RNgray

import wandbResNetFull as RNFull

def logVisionTransformers():

    testConfig = dict(_model.config)
    testConfig["dataset"] = "Oxford 102"
    testConfig["datasetPath"] = "./Data/flower_data"
    testConfig["epochs"] = 20
    model, train_loader, test_loader, criterion, optimizer, config = transfomer.Make(
        _model.config, "VisionTransformerNoFilter.pth")
    with wandb.init(project="KursinisProjektas", name="VisionTransformerNoFilter", config=config):
        config = wandb.config
        model, train_loader, test_loader, criterion, optimizer, config = transfomer.Make(
            _model.config, "VisionTransformerNoFilter.pth")
        model.train(train_loader, criterion, optimizer, config, test_loader)
        model.test(test_loader)

    # vision transformer edge training
    testConfig = dict(_model.config)
    testConfig["dataset"] = "Oxford 102 Edges"
    testConfig["datasetPath"] = "./DataEdgesCanny/flower_data"
    model, train_loader, test_loader, criterion, optimizer, config = transfomer.Make(
        testConfig, "VisionTransformerEdgespth")
    with wandb.init(project="KursinisProjektas", name="VisionTransformerEdges", config=config):
        config = wandb.config
        model, train_loader, test_loader, criterion, optimizer, config = transfomer.Make(
            config, "VisionTransformerEdgespth")
        model.train(train_loader, criterion, optimizer, config, test_loader)
        model.test(test_loader)

    # vision transformer edge skeleton
    testConfig = dict(_model.config)
    testConfig["dataset"] = "Oxford 102 Skeleton"
    testConfig["datasetPath"] = "./DataSkeleton/flower_data"
    model, train_loader, test_loader, criterion, optimizer, config = transfomer.Make(
        testConfig, "VisionTransformerSkeleton.pth")
    with wandb.init(project="KursinisProjektas", name="VisionTransformerSkeleton", config=config):
        config = wandb.config
        model, train_loader, test_loader, criterion, optimizer, config = transfomer.Make(
            config, "VisionTransformerSkeleton.pth")
        model.train(train_loader, criterion, optimizer, config, test_loader)
        model.test(test_loader)

    # vision transformer LBP training
    testConfig = dict(_model.config)
    testConfig["dataset"] = "Oxford 102 LBP"
    testConfig["datasetPath"] = "./DataLBP/flower_data"
    model, train_loader, test_loader, criterion, optimizer, config = transfomer.Make(
        testConfig, "VisionTransformerLBP.pth")
    with wandb.init(project="KursinisProjektas", name="VisionTransformerLBP", config=config):
        config = wandb.config
        model, train_loader, test_loader, criterion, optimizer, config = transfomer.Make(
            config, "VisionTransformerLBP.pth")
        model.train(train_loader, criterion, optimizer, config, test_loader)
        model.test(test_loader)


def logConvolution():
    testConfig = dict(_model.config)
    testConfig["dataset"] = "Oxford 102"
    testConfig["epochs"] = 20
    testConfig["learning_rate"] = 0.0001
    testConfig["datasetPath"] = "./Data/flower_data"
    model, train_loader, test_loader, criterion, optimizer, config = convolution.Make(
        testConfig, "ConvolutionNoFilter.pth")
    with wandb.init(project="KursinisProjektas", name="ConvolutionNoFilter", config=config):
        config = wandb.config
        model, train_loader, test_loader, criterion, optimizer, config = convolution.Make(
            config, "ConvolutionNoFilter.pth")
        model.train(train_loader, criterion, optimizer, config, test_loader)
        model.test(test_loader)

    testConfig = dict(_model.config)
    testConfig["dataset"] = "Oxford 102 Edges"
    testConfig["epochs"] = 16
    testConfig["learning_rate"] = 0.0001
    testConfig["datasetPath"] = "./DataEdgesCanny/flower_data"
    model, train_loader, test_loader, criterion, optimizer, config = convolution.Make(
        testConfig, "ConvolutionEdges.pth")
    with wandb.init(project="KursinisProjektas", name="ConvolutionEdges", config=config):
        config = wandb.config
        model, train_loader, test_loader, criterion, optimizer, config = convolution.Make(
            config, "ConvolutionEdges.pth")
        model.train(train_loader, criterion, optimizer, config, test_loader)
        model.test(test_loader)

    testConfig = dict(_model.config)
    testConfig["dataset"] = "Oxford 102 Skeleton"
    testConfig["epochs"] = 16
    testConfig["learning_rate"] = 0.0001
    testConfig["datasetPath"] = "./DataSkeleton/flower_data"
    model, train_loader, test_loader, criterion, optimizer, config = convolution.Make(
        testConfig, "ConvolutionSkeleton.pth")
    with wandb.init(project="KursinisProjektas", name="ConvolutionSkeleton", config=config):
        config = wandb.config
        model, train_loader, test_loader, criterion, optimizer, config = convolution.Make(
            config, "ConvolutionSkeleton.pth")
        model.train(train_loader, criterion, optimizer, config, test_loader)
        model.test(test_loader)

    testConfig = dict(_model.config)
    testConfig["dataset"] = "Oxford 102 LBP"
    testConfig["epochs"] = 16
    testConfig["learning_rate"] = 0.0001
    testConfig["datasetPath"] = "./DataLBP/flower_data"
    model, train_loader, test_loader, criterion, optimizer, config = convolution.Make(
        testConfig, "ConvolutionLBP.pth")
    with wandb.init(project="KursinisProjektas", name="ConvolutionLBP", config=config):
        config = wandb.config
        model, train_loader, test_loader, criterion, optimizer, config = convolution.Make(
            config, "ConvolutionLBP.pth")
        model.train(train_loader, criterion, optimizer, config, test_loader)
        model.test(test_loader)

def logGrayConvolution():

    testConfig = dict(_model.config)
    testConfig["dataset"] = "Oxford 102 Edges gray"
    testConfig["epochs"] = 30
    testConfig["learning_rate"] = 0.0001
    testConfig["datasetPath"] = "./DataEdgesCannyTrue/flower_data"
    runName = "ConvolutionEdgesTrue"
    model, train_loader, test_loader, criterion, optimizer, config = convolutionGray.Make(
        testConfig, f"{runName}.pth")
    with wandb.init(project="KursinisProjektas", name=runName, config=config):
        config = wandb.config
        model, train_loader, test_loader, criterion, optimizer, config = convolutionGray.Make(
            config, f"{runName}.pt")
        model.train(train_loader, criterion, optimizer, config, test_loader)
        model.test(test_loader)

    testConfig = dict(_model.config)
    testConfig["dataset"] = "Oxford 102 Skeleton"
    testConfig["epochs"] = 30
    testConfig["learning_rate"] = 0.0001
    testConfig["datasetPath"] = "./DataSkeletonTrue/flower_data"
    runName = "ConvolutionSkeletonTrue"
    model, train_loader, test_loader, criterion, optimizer, config = convolutionGray.Make(
        testConfig, f"{runName}.pt")
    with wandb.init(project="KursinisProjektas", name=runName, config=config):
        config = wandb.config
        model, train_loader, test_loader, criterion, optimizer, config = convolutionGray.Make(
            config, f"{runName}.pt")
        model.train(train_loader, criterion, optimizer, config, test_loader)
        model.test(test_loader)

    testConfig = dict(_model.config)
    testConfig["dataset"] = "Oxford 102 LBP"
    testConfig["epochs"] = 30
    testConfig["learning_rate"] = 0.0001
    testConfig["datasetPath"] = "./DataLBPTrue/flower_data"
    runName = "ConvolutionLBPTrue"
    model, train_loader, test_loader, criterion, optimizer, config = convolutionGray.Make(
        testConfig, f"{runName}.pt")
    with wandb.init(project="KursinisProjektas", name=runName, config=config):
        config = wandb.config
        model, train_loader, test_loader, criterion, optimizer, config = convolutionGray.Make(
            config, f"{runName}.pt")
        model.train(train_loader, criterion, optimizer, config, test_loader)
        model.test(test_loader)

def logConvolution2():
    paths = {
        "1": "model_ConvolutionNoFilter.pt",
        "2": "model_ConvolutionEdges.pt",
        "3": "model_ConvolutionSkeleton.pt",
        "4": "model_ConvolutionLBP.pt"
    }
    testConfig = dict(_model.config)
    testConfig["dataset"] = "Oxford 102"
    testConfig["epochs"] = 30
    testConfig["learning_rate"] = 0.0001
    testConfig["datasetPath"] = "./Data/flower_data"

    model1Path = paths["1"]
    model2Path = paths["2"]
    testConfig["SupportModel1"] = model1Path
    testConfig["SupportModel2"] = model2Path
    runName = "ConvolutionNoFilterAndEdges"
    saveModelPath = runName+".pt"
    model, train_loader, test_loader, criterion, optimizer, config = convolution2.Make(
        testConfig, saveModelPath, model1Path, model2Path)
    with wandb.init(project="KursinisProjektas", name=runName, config=config):
        config = wandb.config
        model, train_loader, test_loader, criterion, optimizer, config = convolution2.Make(
            config, saveModelPath, model1Path, model2Path)
        model.train(train_loader, criterion, optimizer, config, test_loader)
        model.test(test_loader)

#########################################################################################################################################################
    model1Path = paths["1"]
    model2Path = paths["3"]
    testConfig["SupportModel1"] = model1Path
    testConfig["SupportModel2"] = model2Path
    runName = "ConvolutionNoFilterAndSkeleton"
    saveModelPath = runName+".pt"

    model, train_loader, test_loader, criterion, optimizer, config = convolution2.Make(
        testConfig, saveModelPath, model1Path, model2Path)
    with wandb.init(project="KursinisProjektas", name=runName, config=config):
        config = wandb.config
        model, train_loader, test_loader, criterion, optimizer, config = convolution2.Make(
            config, saveModelPath, model1Path, model2Path)
        model.train(train_loader, criterion, optimizer, config, test_loader)
        model.test(test_loader)

#########################################################################################################################################################
    model1Path = paths["1"]
    model2Path = paths["4"]
    testConfig["SupportModel1"] = model1Path
    testConfig["SupportModel2"] = model2Path
    runName = "ConvolutionNoFilterAndLBP"
    saveModelPath = runName+".pt"

    model, train_loader, test_loader, criterion, optimizer, config = convolution2.Make(
        testConfig, saveModelPath, model1Path, model2Path)
    with wandb.init(project="KursinisProjektas", name=runName, config=config):
        config = wandb.config
        model, train_loader, test_loader, criterion, optimizer, config = convolution2.Make(
            config, saveModelPath, model1Path, model2Path)
        model.train(train_loader, criterion, optimizer, config, test_loader)
        model.test(test_loader)

#########################################################################################################################################################
    model1Path = paths["2"]
    model2Path = paths["3"]
    testConfig["SupportModel1"] = model1Path
    testConfig["SupportModel2"] = model2Path
    runName = "ConvolutionEdgesAndSkeleton"
    saveModelPath = runName+".pt"

    model, train_loader, test_loader, criterion, optimizer, config = convolution2.Make(
        testConfig, saveModelPath, model1Path, model2Path)
    with wandb.init(project="KursinisProjektas", name=runName, config=config):
        config = wandb.config
        model, train_loader, test_loader, criterion, optimizer, config = convolution2.Make(
            config, saveModelPath, model1Path, model2Path)
        model.train(train_loader, criterion, optimizer, config, test_loader)
        model.test(test_loader)

#########################################################################################################################################################
    model1Path = paths["2"]
    model2Path = paths["4"]
    testConfig["SupportModel1"] = model1Path
    testConfig["SupportModel2"] = model2Path
    runName = "ConvolutionEdgesAndLBP"
    saveModelPath = runName+".pt"

    model, train_loader, test_loader, criterion, optimizer, config = convolution2.Make(
        testConfig, saveModelPath, model1Path, model2Path)
    with wandb.init(project="KursinisProjektas", name=runName, config=config):
        config = wandb.config
        model, train_loader, test_loader, criterion, optimizer, config = convolution2.Make(
            config, saveModelPath, model1Path, model2Path)
        model.train(train_loader, criterion, optimizer, config, test_loader)
        model.test(test_loader)

#########################################################################################################################################################
    model1Path = paths["3"]
    model2Path = paths["4"]
    testConfig["SupportModel1"] = model1Path
    testConfig["SupportModel2"] = model2Path
    runName = "ConvolutionSkeletonAndLBP"
    saveModelPath = runName+".pt"

    model, train_loader, test_loader, criterion, optimizer, config = convolution2.Make(
        testConfig, saveModelPath, model1Path, model2Path)
    with wandb.init(project="KursinisProjektas", name=runName, config=config):
        config = wandb.config
        model, train_loader, test_loader, criterion, optimizer, config = convolution2.Make(
            config, saveModelPath, model1Path, model2Path)
        model.train(train_loader, criterion, optimizer, config, test_loader)
        model.test(test_loader)

def logConvolutionRGBGray():
    models = {
        "RGB" :  "model_ConvolutionNoFilter.pt",
        "grayEdges" : "model_ConvolutionEdgesTrue.pt",
        "graySkeleton" : "model_ConvolutionSkeletonTrue.pt",
        "grayLBP" : "model_ConvolutionLBPTrue.pt"
    }
    testConfig = dict(_model.config)
    testConfig["dataset"] = "Oxford 102 Skeleton"
    testConfig["epochs"] = 20
    testConfig["learning_rate"] = 0.0001
    testConfig["datasetPath"] = ""
    testConfig["grayDataset"] = "./DataEdgesCannyTrue/flower_data"
    testConfig["rgbDataset"] = "./Data/flower_data"

    runName = "ConvolutionNoFilterEdgesTrue"
    model, train_loader, test_loader, criterion, optimizer, config = convolutionRGBGray.Make(
        testConfig, f"{runName}.pt",
        grayModelPath=models["grayEdges"],
        rgbModelPath=models["RGB"]
    )
    with wandb.init(project="KursinisProjektas", name=runName, config=config):
        config = wandb.config
        model, train_loader, test_loader, criterion, optimizer, config = convolutionRGBGray.Make(
            testConfig, f"{runName}.pt",
            grayModelPath=models["grayEdges"],
            rgbModelPath=models["RGB"]
        )
        model.train(train_loader, criterion, optimizer, config, test_loader)
        model.test(test_loader)

    testConfig = dict(_model.config)
    testConfig["dataset"] = "Oxford 102 Data"
    testConfig["epochs"] = 20
    testConfig["learning_rate"] = 0.0001
    testConfig["datasetPath"] = ""
    testConfig["grayDataset"] = "./DataSkeletonTrue/flower_data"
    testConfig["rgbDataset"] = "./Data/flower_data"

    runName = "ConvolutionNoFilterSkeletonTrue"
    model, train_loader, test_loader, criterion, optimizer, config = convolutionRGBGray.Make(
        testConfig, f"{runName}.pt",
        grayModelPath=models["graySkeleton"],
        rgbModelPath=models["RGB"]
    )
    with wandb.init(project="KursinisProjektas", name=runName, config=config):
        config = wandb.config
        model, train_loader, test_loader, criterion, optimizer, config = convolutionRGBGray.Make(
            testConfig, f"{runName}.pt",
            grayModelPath=models["graySkeleton"],
            rgbModelPath=models["RGB"]
        )
        model.train(train_loader, criterion, optimizer, config, test_loader)
        model.test(test_loader)

    testConfig = dict(_model.config)
    testConfig["dataset"] = "Oxford 102 Data"
    testConfig["epochs"] = 20
    testConfig["learning_rate"] = 0.0001
    testConfig["datasetPath"] = ""
    testConfig["grayDataset"] = "./DataLBPTrue/flower_data"
    testConfig["rgbDataset"] = "./Data/flower_data"

    runName = "ConvolutionNoFilterLBPTrue"
    model, train_loader, test_loader, criterion, optimizer, config = convolutionRGBGray.Make(
        testConfig, f"{runName}.pt",
        grayModelPath=models["grayLBP"],
        rgbModelPath=models["RGB"]
    )
    with wandb.init(project="KursinisProjektas", name=runName, config=config):
        config = wandb.config
        model, train_loader, test_loader, criterion, optimizer, config = convolutionRGBGray.Make(
            testConfig, f"{runName}.pt",
            grayModelPath=models["grayLBP"],
            rgbModelPath=models["RGB"]
        )
        model.train(train_loader, criterion, optimizer, config, test_loader)
        model.test(test_loader)

def logConvolution4():
    paths = {
        "1": "model_ConvolutionNoFilter.pt",
        "2": "model_ConvolutionEdges.pt",
        "3": "model_ConvolutionSkeleton.pt",
        "4": "model_ConvolutionLBP.pt"
    }
    testConfig = dict(_model.config)
    testConfig["dataset"] = "Oxford 102"
    testConfig["epochs"] = 45
    testConfig["learning_rate"] = 0.00001
    testConfig["datasetPath"] = "./Data/flower_data"

    model1Path = paths["1"]
    model2Path = paths["2"]
    model3Path = paths["3"]
    model4Path = paths["4"]
    testConfig["SupportModel1"] = model1Path
    testConfig["SupportModel2"] = model2Path
    testConfig["SupportModel3"] = model3Path
    testConfig["SupportModel4"] = model4Path
    testConfig["batch_size"] = 30
    runName = "ConvolutionAll"
    saveModelPath = runName+".pt"
    model, train_loader, test_loader, criterion, optimizer, config = convolution4.Make(
        testConfig, saveModelPath, model1Path, model2Path, model3Path, model4Path)
    with wandb.init(project="KursinisProjektas", name=runName, config=config):
        config = wandb.config
        model, train_loader, test_loader, criterion, optimizer, config = convolution4.Make(
            testConfig, saveModelPath, model1Path, model2Path, model3Path, model4Path)
        model.train(train_loader, criterion, optimizer, config, test_loader)
        model.test(test_loader)

def logConvolution4Gray():
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

    runName = "ConvolutionAllTrue"
    model, train_loader, test_loader, criterion, optimizer, config = convolution4True.Make(
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
    with wandb.init(project="KursinisProjektas", name=runName, config=config):
        config = wandb.config
        model, train_loader, test_loader, criterion, optimizer, config = convolution4True.Make(
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

def logAllModels():
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
    testConfig["dataset"] = "AllDatasets"
    testConfig["epochs"] = 16
    testConfig["learning_rate"] = 0.0001
    testConfig["datasetPath"] = ""

    runName = "AllModels"

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

def logAllModelsSVC():
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
    testConfig["dataset"] = "AllDatasets"
    testConfig["epochs"] = 16
    testConfig["learning_rate"] = 0.0001
    testConfig["datasetPath"] = ""

    runName = "AllModelsSVC"

    with wandb.init(project="KursinisProjektas", name=runName, config=testConfig):
        config = wandb.config
        model, train_loader, test_loader, config = FullModelSVC.Make(
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
        model.train(train_loader)
        model.test(train_loader, test_loader)

def log_GLCM_LBP_Model():
    testConfig = dict(_model.config)
    testConfig["dataset"] = "Oxford 102 GLCM LBP"
    testConfig["epochs"] = 500
    testConfig["learning_rate"] = 0.0001
    testConfig["datasetPath"] = "./DataGLCM_LBP/flower_data"
    model, train_loader, test_loader, criterion, optimizer, config = linearModel.Make(
        testConfig, "LinearModel_GLCM_LBP.pth")
    with wandb.init(project="KursinisProjektas", name="LinearModel_GLCM_LBP", config=config):
        config = wandb.config
        model, train_loader, test_loader, criterion, optimizer, config = linearModel.Make(
            config, "LinearModel_GLCM_LBP.pth")
        model.train(train_loader, criterion, optimizer, config, test_loader)
        model.test(test_loader)

def logResNet():
    testConfig = dict(_model.config)
    testConfig["dataset"] = "Oxford 102"
    testConfig["epochs"] = 50
    testConfig["learning_rate"] = 0.0001
    testConfig["datasetPath"] = "./Data/flower_data"
    model, train_loader, test_loader, criterion, optimizer, config = ResNet.Make(
        testConfig, "ResNet.pth")
    with wandb.init(project="KursinisProjektas", name="ResNet", config=config):
        config = wandb.config
        model, train_loader, test_loader, criterion, optimizer, config = ResNet.Make(
            config, "ResNet.pth")
        model.train(train_loader, criterion, optimizer, config, test_loader)
        model.test(test_loader)


    testConfig = dict(_model.config)
    testConfig["dataset"] = "Oxford 102 Edges"
    testConfig["epochs"] = 50
    testConfig["learning_rate"] = 0.0001
    testConfig["datasetPath"] = "./DataEdgesCanny/flower_data"
    model, train_loader, test_loader, criterion, optimizer, config = RNgray.Make(
        testConfig, "ResNetEdges.pth")
    with wandb.init(project="KursinisProjektas", name="ResNetEdges", config=config):
        config = wandb.config
        model, train_loader, test_loader, criterion, optimizer, config = RNgray.Make(
            config, "ResNetEdges.pth")
        model.train(train_loader, criterion, optimizer, config, test_loader)
        model.test(test_loader)

    testConfig = dict(_model.config)
    testConfig["dataset"] = "Oxford 102 Skeleton"
    testConfig["epochs"] = 50
    testConfig["learning_rate"] = 0.0001
    testConfig["datasetPath"] = "./DataSkeleton/flower_data"
    model, train_loader, test_loader, criterion, optimizer, config = RNgray.Make(
        testConfig, "ResNetSkeleton.pth")
    with wandb.init(project="KursinisProjektas", name="ResNetSkeleton", config=config):
        config = wandb.config
        model, train_loader, test_loader, criterion, optimizer, config = RNgray.Make(
            config, "ResNetSkeleton.pth")
        model.train(train_loader, criterion, optimizer, config, test_loader)
        model.test(test_loader)

    testConfig = dict(_model.config)
    testConfig["dataset"] = "Oxford 102 LBP"
    testConfig["epochs"] = 50
    testConfig["learning_rate"] = 0.0001
    testConfig["datasetPath"] = "./DataLBP/flower_data"
    model, train_loader, test_loader, criterion, optimizer, config = RNgray.Make(
        testConfig, "ResNetLBP.pth")
    with wandb.init(project="KursinisProjektas", name="ResNetLBP", config=config):
        config = wandb.config
        model, train_loader, test_loader, criterion, optimizer, config = RNgray.Make(
            config, "ResNetLBP.pth")
        model.train(train_loader, criterion, optimizer, config, test_loader)
        model.test(test_loader)

def logResNetEnsemble():
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
if __name__ == "__main__":

    wandb.login()
    logVisionTransformers()
    logConvolution()
    logGrayConvolution()
    logConvolutionRGBGray()
    logConvolution4Gray()
    logConvolution2()
    logConvolution4()

    logVisionTransformers()

    log_GLCM_LBP_Model()

    logGrayConvolution()
    logAllModels()
    logAllModelsSVC()

    logResNet()
    logResNetEnsemble()
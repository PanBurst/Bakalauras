
import wandb
import wandbVisionTransformer as transfomer
import wandbConvolution as convolution
import Model as _model
import wandbConvolution2 as convolution2
import wandbConvolution4 as convolution4

import wandbConvolutionTrue as convolutionGray
import wandbConvolution2True as convolutionRGBGray
import wandbConvolution4True as convolution4True


def logConvolution():
    models = {
        "rgb": "model_ConvolutionNoFilter.pt",
        "edges": "model_ConvolutionEdges.pt",
        "skeleton": "model_ConvolutionSkeleton.pt",
        "lbp": "model_ConvolutionLBP.pt"
    }
    testConfig = dict(_model.config)
    testConfig["dataset"] = "Oxford 102"
    testConfig["datasetPath"] = "./Data/flower_data"
    model, train_loader, test_loader, criterion, optimizer, config = convolution.MakeLoad(
        testConfig, models["rgb"])
    with wandb.init(project="KursinisProjektas", name="rgbConvolution", config=config):
        config = wandb.config
        model, train_loader, test_loader, criterion, optimizer, config = convolution.MakeLoad(
        testConfig, models["rgb"])
        
        (trainTop1, trainTop5) = model.testLoad(train_loader)
        (validationTop1, validationTop5) = model.testLoad(test_loader)
        wandb.log({"train top 1 ": trainTop1, "train top 5": trainTop5, "validation top 1": validationTop1, "validation top 5": validationTop5})

    model, train_loader, test_loader, criterion, optimizer, config = convolution.MakeLoad(
        testConfig, models["edges"])
    with wandb.init(project="KursinisProjektas", name="edgeConvolution", config=config):
        model, train_loader, test_loader, criterion, optimizer, config = convolution.MakeLoad(
        testConfig, models["edges"])

        (trainTop1, trainTop5) = model.testLoad(train_loader)
        (validationTop1, validationTop5) = model.testLoad(test_loader)
        wandb.log({"train top 1 ": trainTop1, "train top 5": trainTop5, "validation top 1": validationTop1, "validation top 5": validationTop5})


    model, train_loader, test_loader, criterion, optimizer, config = convolution.MakeLoad(
        testConfig, models["skeleton"])
    with wandb.init(project="KursinisProjektas", name="skeletonConvolution", config=config):
        model, train_loader, test_loader, criterion, optimizer, config = convolution.MakeLoad(
        testConfig, models["skeleton"])

        (trainTop1, trainTop5) = model.testLoad(train_loader)
        (validationTop1, validationTop5) = model.testLoad(test_loader)
        wandb.log({"train top 1 ": trainTop1, "train top 5": trainTop5, "validation top 1": validationTop1, "validation top 5": validationTop5})


    model, train_loader, test_loader, criterion, optimizer, config = convolution.MakeLoad(
        testConfig, models["lbp"])
    with wandb.init(project="KursinisProjektas", name="lbpConvolution", config=config):
        model, train_loader, test_loader, criterion, optimizer, config = convolution.MakeLoad(
        testConfig, models["lbp"])

        (trainTop1, trainTop5) = model.testLoad(train_loader)
        (validationTop1, validationTop5) = model.testLoad(test_loader)
        wandb.log({"train top 1 ": trainTop1, "train top 5": trainTop5, "validation top 1": validationTop1, "validation top 5": validationTop5})

def logGrayConvolution():
    models = {
        "grayEdges" : "model_ConvolutionEdgesTrue.pt",
        "graySkeleton" : "model_ConvolutionSkeletonTrue.pt",
        "grayLBP" : "model_ConvolutionLBPTrue.pt"
    }
    dataPaths = {
        "grayEdges" : "./DataEdgesCannyTrue/flower_data",
        "graySkeleton" : "./DataSkeletonTrue/flower_data",
        "grayLBP" : "./DataLBPTrue/flower_data"
    }
    testConfig = dict(_model.config)
    testConfig["dataset"] = "Oxford 102"
    testConfig["datasetPath"] = dataPaths["grayEdges"]
    model, train_loader, test_loader, criterion, optimizer, config = convolutionGray.MakeLoad(
        testConfig, models["grayEdges"])
    with wandb.init(project="KursinisProjektas", name="grayEdgesConvolution", config=config):
        config = wandb.config
        model, train_loader, test_loader, criterion, optimizer, config = convolutionGray.MakeLoad(
        testConfig, models["grayEdges"])
        
        (trainTop1, trainTop5) = model.testLoad(train_loader)
        (validationTop1, validationTop5) = model.testLoad(test_loader)
        wandb.log({"train top 1 ": trainTop1, "train top 5": trainTop5, "validation top 1": validationTop1, "validation top 5": validationTop5})

    testConfig = dict(_model.config)
    testConfig["dataset"] = "Oxford 102"
    testConfig["datasetPath"] = dataPaths["graySkeleton"]
    model, train_loader, test_loader, criterion, optimizer, config = convolutionGray.MakeLoad(
        testConfig, models["graySkeleton"])
    with wandb.init(project="KursinisProjektas", name="graySkeletonConvolution", config=config):
        config = wandb.config
        model, train_loader, test_loader, criterion, optimizer, config = convolutionGray.MakeLoad(
        testConfig, models["graySkeleton"])
        
        (trainTop1, trainTop5) = model.testLoad(train_loader)
        (validationTop1, validationTop5) = model.testLoad(test_loader)
        wandb.log({"train top 1 ": trainTop1, "train top 5": trainTop5, "validation top 1": validationTop1, "validation top 5": validationTop5})

    testConfig = dict(_model.config)
    testConfig["dataset"] = "Oxford 102"
    testConfig["datasetPath"] = dataPaths["grayLBP"]
    model, train_loader, test_loader, criterion, optimizer, config = convolutionGray.MakeLoad(
        testConfig, models["grayLBP"])
    with wandb.init(project="KursinisProjektas", name="grayLBPConvolution", config=config):
        config = wandb.config
        model, train_loader, test_loader, criterion, optimizer, config = convolutionGray.MakeLoad(
        testConfig, models["grayLBP"])
        
        (trainTop1, trainTop5) = model.testLoad(train_loader)
        (validationTop1, validationTop5) = model.testLoad(test_loader)
        wandb.log({"train top 1 ": trainTop1, "train top 5": trainTop5, "validation top 1": validationTop1, "validation top 5": validationTop5})

def logConvolution2True():

    models = {
        "RGB" :  "model_ConvolutionNoFilter.pt",
        "grayEdges" : "model_ConvolutionEdgesTrue.pt",
        "graySkeleton" : "model_ConvolutionSkeletonTrue.pt",
        "grayLBP" : "model_ConvolutionLBPTrue.pt"
    }
    testConfig = dict(_model.config)
    testConfig["dataset"] = "Oxford 102 Skeleton"
    testConfig["epochs"] = 16
    testConfig["learning_rate"] = 0.00001
    testConfig["datasetPath"] = ""
    testConfig["grayDataset"] = "./DataEdgesCannyTrue/flower_data"
    testConfig["rgbDataset"] = "./Data/flower_data"

    runName = "model_ConvolutionNoFilterEdgesTrue"
    model, train_loader, test_loader, criterion, optimizer, config = convolutionRGBGray.MakeLoad(
        testConfig, f"{runName}.pt",
        grayModelPath=models["grayEdges"],
        rgbModelPath=models["RGB"]
    )
    with wandb.init(project="KursinisProjektas", name=runName, config=config):
        config = wandb.config
        model, train_loader, test_loader, criterion, optimizer, config = convolutionRGBGray.MakeLoad(
            testConfig, f"{runName}.pt",
            grayModelPath=models["grayEdges"],
            rgbModelPath=models["RGB"]
        )
        (trainTop1, trainTop5) = model.testLoad(train_loader)
        (validationTop1, validationTop5) = model.testLoad(test_loader)
        wandb.log({"train top 1 ": trainTop1, "train top 5": trainTop5, "validation top 1": validationTop1, "validation top 5": validationTop5})

    testConfig = dict(_model.config)
    testConfig["dataset"] = "Oxford 102 Data"
    testConfig["epochs"] = 16
    testConfig["learning_rate"] = 0.00001
    testConfig["datasetPath"] = ""
    testConfig["grayDataset"] = "./DataSkeletonTrue/flower_data"
    testConfig["rgbDataset"] = "./Data/flower_data"

    runName = "model_ConvolutionNoFilterSkeletonTrue"
    model, train_loader, test_loader, criterion, optimizer, config = convolutionRGBGray.MakeLoad(
        testConfig, f"{runName}.pt",
        grayModelPath=models["graySkeleton"],
        rgbModelPath=models["RGB"]
    )
    with wandb.init(project="KursinisProjektas", name=runName, config=config):
        config = wandb.config
        model, train_loader, test_loader, criterion, optimizer, config = convolutionRGBGray.MakeLoad(
            testConfig, f"{runName}.pt",
            grayModelPath=models["graySkeleton"],
            rgbModelPath=models["RGB"]
        )
        (trainTop1, trainTop5) = model.testLoad(train_loader)
        (validationTop1, validationTop5) = model.testLoad(test_loader)
        wandb.log({"train top 1 ": trainTop1, "train top 5": trainTop5, "validation top 1": validationTop1, "validation top 5": validationTop5})

    testConfig = dict(_model.config)
    testConfig["dataset"] = "Oxford 102 Data"
    testConfig["epochs"] = 16
    testConfig["learning_rate"] = 0.00001
    testConfig["datasetPath"] = ""
    testConfig["grayDataset"] = "./DataLBPTrue/flower_data"
    testConfig["rgbDataset"] = "./Data/flower_data"

    runName = "model_ConvolutionNoFilterLBPTrue"
    model, train_loader, test_loader, criterion, optimizer, config = convolutionRGBGray.MakeLoad(
        testConfig, f"{runName}.pt",
        grayModelPath=models["grayLBP"],
        rgbModelPath=models["RGB"]
    )
    with wandb.init(project="KursinisProjektas", name=runName, config=config):
        config = wandb.config
        model, train_loader, test_loader, criterion, optimizer, config = convolutionRGBGray.MakeLoad(
            testConfig, f"{runName}.pt",
            grayModelPath=models["grayLBP"],
            rgbModelPath=models["RGB"]
        )
        (trainTop1, trainTop5) = model.testLoad(train_loader)
        (validationTop1, validationTop5) = model.testLoad(test_loader)
        wandb.log({"train top 1 ": trainTop1, "train top 5": trainTop5, "validation top 1": validationTop1, "validation top 5": validationTop5})

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
    testConfig["epochs"] = 30
    testConfig["learning_rate"] = 0.00001
    testConfig["datasetPath"] = ""

    runName = "model_ConvolutionAllTrue"
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
        (trainTop1, trainTop5) = model.testLoad(train_loader)
        (validationTop1, validationTop5) = model.testLoad(test_loader)
        wandb.log({"train top 1 ": trainTop1, "train top 5": trainTop5, "validation top 1": validationTop1, "validation top 5": validationTop5})


if __name__ == "__main__":

    wandb.login()
    
    testConfig = dict(_model.config)

    # logConvolution()
    # logGrayConvolution()
    # logConvolution2True()
    logConvolution4Gray()
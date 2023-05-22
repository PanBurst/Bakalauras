from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io

def ConfusionMatrix(dataPredictions:list, dataLabels:list, filePath:str="confusionMatrix.png", newLabels=None):
    uniqueLabels = np.unique(dataLabels)
    key = [str(x)for x in uniqueLabels]
    value = [0]*len( uniqueLabels)
    bucketsObj = zip(key, value)
    rowDict = dict(bucketsObj)
    eDict = dict(rowDict)
    rows = []
    for uniqueLabel in uniqueLabels:
        rowDict = dict(eDict)
        for idx, label in enumerate(dataLabels):
            if label == uniqueLabel:
                key = str(dataPredictions[idx])
                rowDict[key] = rowDict[key] + 1
        notNormalizedRow = list(rowDict.values())
        rowSum = sum(notNormalizedRow)
        normalizedRow = list(map(lambda x: x/rowSum, notNormalizedRow))
        rows.append(normalizedRow)
    
    plt.ioff()
    plt.clf()
    npRows = np.array(rows)
    heatMap = plt.pcolor(npRows)
    plt.colorbar()
    plt.savefig(filePath)
    im = Image.open(filePath)
    return im


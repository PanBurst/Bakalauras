
import os
import cv2
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
baseDir = "./Data"
fileType = ".jpg"
newBaseDir = "./DataEdgesCannyTrue"


def fileDFS(rootPath: str):
    currentDir = os.listdir(rootPath)
    for fileName in currentDir:
        if (fileType in fileName):

            TransformAndSave(rootPath, fileName)
        else:
            if not os.path.isfile(rootPath + "\\" + fileName):
                fileDFS(rootPath + "\\" + fileName)


sampleFile = r"C:\Users\Tomas\Desktop\Universiteto\KursinisProjektas\Kodas\Data\flower_data\test\image_00277.jpg"


def CannyImage(img, radius=10):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(gray.shape[:2], dtype="uint8")
    w = mask.shape[0]
    h = mask.shape[1]

    _thresh = cv.adaptiveThreshold(
        gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 51, 25)
    _edges = cv.Canny(_thresh, 50, 200)
    return _edges


def TransformAndSave(path: str, fileName: str):
    newpath = path.replace(baseDir, newBaseDir)
    fullFilePath = path + "\\" + fileName
    fullNewFilePath = newpath + "\\" + fileName
    img = cv2.imread(fullFilePath)
    masked = CannyImage(img, 10)
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    cv2.imwrite(fullNewFilePath, masked)


if __name__ == "__main__":
    fileDFS(baseDir)

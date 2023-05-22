import os
import cv2
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import math
baseDir = "./Data"
fileType = ".jpg"
newBaseDir = "./DataSkeleton"


def fileDFS(rootPath: str):
    currentDir = os.listdir(rootPath)
    for fileName in currentDir:
        if (fileType in fileName):
            # fullFilePath = rootPath + "\\" + fileName
            # img = cv2.imread(fullFilePath)
            # resized = cv2.resize(img, (500, 500))
            # cv2.imwrite(fullFilePath, resized)
            TransformAndSave(rootPath, fileName)
        else:
            if not os.path.isfile(rootPath + "\\" + fileName):
                fileDFS(rootPath + "\\" + fileName)


def TransformAndSave(path: str, fileName: str):
    newpath = path.replace(baseDir, newBaseDir)
    fullFilePath = path + "\\" + fileName
    fullNewFilePath = newpath + "\\" + fileName
    img = cv2.imread(fullFilePath)
    masked = morphSkeleton(img, 7)
    if path.find("\\4") != -1:
        tarck = True
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    cv2.imwrite(fullNewFilePath, masked)


def morphSkeleton(img, radius=7):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    
    skel = np.zeros(gray.shape, np.uint8)

    ret, _img = cv2.threshold(gray, 75, 255, 3)
    size = np.size(_img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    count = 500
    while (not done):
        eroded = cv2.erode(_img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(_img, temp)
        skel = cv2.bitwise_or(skel, temp)
        _img = eroded.copy()

        zeros = size - cv2.countNonZero(_img)
        if zeros == size:
            done = True
        count -= 1
        if count == 0:
            break
    mask = np.zeros(gray.shape[:2], dtype="uint8")
    w = mask.shape[0]
    h = mask.shape[1]

    for x in range(w):
        for y in range(h):
            if skel[y][x] > 15:
                p1x, p1y, p2x, p2y = 0, 0, 0, 0
                if x < radius:
                    p1x = 0
                else:
                    p1x = x-radius
                if y < radius:
                    p1y = 0
                else:
                    p1y = y - radius

                if x+radius > w:
                    p2x = w-1
                else:
                    p2x = x+radius

                if y+radius > h:
                    p2y = h - 1
                else:
                    p2y = y+radius
                cv2.rectangle(mask, (p1x, p1y), (p2x, p2y), 255, -1)
    masked = cv2.bitwise_and(img, img, mask=mask)
    return masked


sampleFile = r"C:\Users\Tomas\Desktop\Universiteto\KursinisProjektas\Kodas\Data\flower_data\test\image_00277.jpg"


if __name__ == "__main__":
    fileDFS(baseDir)

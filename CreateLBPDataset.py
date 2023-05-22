import os
import cv2
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import math
baseDir = "./Data"
fileType = ".jpg"
newBaseDir = "./DataLBP"


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
    masked = lbp(img)
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    cv2.imwrite(fullNewFilePath, masked)


def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value


def lbp_calculated_pixel(img, x, y):
    '''
     64 | 128 |   1
    ----------------
     32 |   0 |   2
    ----------------
     16 |   8 |   4    
    '''
    center = img[x][y]
    val_ar = []
    val_ar.append(get_pixel(img, center, x-1, y+1))     # top_right
    val_ar.append(get_pixel(img, center, x, y+1))       # right
    val_ar.append(get_pixel(img, center, x+1, y+1))     # bottom_right
    val_ar.append(get_pixel(img, center, x+1, y))       # bottom
    val_ar.append(get_pixel(img, center, x+1, y-1))     # bottom_left
    val_ar.append(get_pixel(img, center, x, y-1))       # left
    val_ar.append(get_pixel(img, center, x-1, y-1))     # top_left
    val_ar.append(get_pixel(img, center, x-1, y))       # top

    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
    return val


def lbp(img):
    img_bgr = img
    height, width, channel = img_bgr.shape
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    img_lbp = np.zeros((height, width, 3), np.uint8)
    for i in range(0, height):
        for j in range(0, width):
            img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)
    #(data - np.min(data)) / (np.max(data) - np.min(data))
    img_lbp = img_lbp / 255
    img_lbp = img_lbp * img_bgr
    img_lbp = img_lbp.astype('uint8')
    return img_lbp


if __name__ == '__main__':
    fileDFS(baseDir)

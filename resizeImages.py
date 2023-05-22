import os
import cv2
baseDir = "./Data"
fileType = ".jpg"


def fileDFS(rootPath: str):
    currentDir = os.listdir(rootPath)
    for fileName in currentDir:
        if (fileType in fileName):
            fullFilePath = rootPath + "\\" + fileName
            img = cv2.imread(fullFilePath)
            resized = cv2.resize(img, (500, 500))
            cv2.imwrite(fullFilePath, resized)
        else:
            if not os.path.isfile(rootPath + "\\" + fileName):
                fileDFS(rootPath + "\\" + fileName)


if __name__ == "__main__":
    # resizes all the images in folder
    fileDFS(baseDir)

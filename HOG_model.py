from torchvision import datasets
import torch as th

import os
import numpy as np
import cv2


from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

from skimage.feature import hog

from multiprocessing import Pool

from functools import partial

from joblib import dump, load


def ConvertImageToHogValues(imagePath:str, pixels_per_cell, cells_per_block, orientations):
            image = cv2.imread(imagePath)
            hogValues = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, feature_vector=True, channel_axis=-1)
            newImagePath = imagePath.replace('Data','DataHog').replace('jpg','npy')
            np.save(newImagePath, hogValues)

def createHogData(path, pixels_per_cell, cells_per_block, orientations):
    dataFolder = datasets.ImageFolder(path)
    
    imgPaths = list(map(lambda x: x[0], dataFolder.imgs))
    
    for dataFolderFile in imgPaths:
         fileDir = dataFolderFile.replace('\\','/')
         newFileDir = fileDir.replace('Data','DataHog')
         newFileDir = newFileDir[: newFileDir.rfind('/')]
         if not os.path.exists(newFileDir):
              os.makedirs(newFileDir)

    p = Pool(processes=20)
    p.map(partial(ConvertImageToHogValues, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, orientations=orientations), imgPaths)
    p.close()

def __loadData(path):
    return np.load(path)
def __isValidFile(path):
    return '.npy' in path
def loadHogData(path):
    dataFolder = datasets.DatasetFolder(path, loader=__loadData, is_valid_file=__isValidFile)
    dataFilePaths = list(map(lambda x: x[0], dataFolder.samples))
    data = []
    for dataFilePath in dataFilePaths:
         fileData = np.load(dataFilePath)
         data.append(fileData)
    dataLabels = list(map(lambda x: x[1], dataFolder.samples))
    return data, dataLabels 

def Make(pixels_per_cell, cells_per_block, orientations):
    trainPath = r'./Data/flower_data/train'
    testPath = r'./Data/flower_data/valid'

    
    createHogData(trainPath, pixels_per_cell, cells_per_block, orientations)
    print("Created the train images")
    createHogData(testPath, pixels_per_cell, cells_per_block, orientations)
    print("Created the test images")

    trainPath = r'./DataHog/flower_data/train'
    testPath = r'./DataHog/flower_data/valid'

    hog_train_data, hog_train_labels = loadHogData(trainPath)
    print("Loaded train data")
    # clf = SVC(kernel='rbf', class_weight='balanced')
    # SVC(class_weight='balanced')

    clf = RandomForestClassifier(n_estimators = 100,n_jobs=-1)
    clf.fit(hog_train_data, hog_train_labels)
    print("Fitted train data")

    hog_test_data, hog_test_labels = loadHogData(testPath)
    print("Loaded test data")
    print(
        f'''hog accuracy: {clf.score(hog_test_data, hog_test_labels)}
                pixels_per_cell: {pixels_per_cell}
                cells_per_block: {cells_per_block}
                orientations: {orientations}
            ''')
    
def loadGrayImages(path):
    dataFolder = datasets.ImageFolder(path)

    def GetGrayImage(imgPath):
        image = cv2.imread(imgPath)

        # To Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        grayImage = np.asarray(cv2.normalize(
            image, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)).flatten()
        # grayImage = np.asarray(image.convert('L')).flatten()
        return grayImage

    imgData = list(map(lambda x: GetGrayImage(x[0]), dataFolder.imgs))
    return (imgData, list(map(lambda x: x[1], dataFolder.imgs)))


class HOG_Dataset(th.utils.data.Dataset):
    def __init__(self, data, labels):
        data = list(map(lambda x: x.astype(float), data))
        self.data = list(zip(data, labels))

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)


def CreateDatbasesPOC(n_components):
    trainPath = r'./Data/flower_data/train'
    testPath = r'./Data/flower_data/valid'
    train_data = loadGrayImages(trainPath)

    train_images, train_labels = train_data

    pca = PCA(n_components=n_components).fit(train_images)
    x_train_pca = pca.transform(train_images)

    test_data = loadGrayImages(trainPath)
    test_images, test_labels = test_data
    x_test_pca = pca.transform(test_images)

    trainDataset = HOG_Dataset(x_train_pca, train_labels)
    testDataset = HOG_Dataset(x_test_pca, test_labels)
    return trainDataset, testDataset


class HOG_model:
    def __init__(self, ModelPath=None):
        self.clf = None
        self.ModelPath = ModelPath
        if ModelPath is not None and os.path.exists(ModelPath):
            self.clf = load(ModelPath)
        else:
            self.clf = RandomForestClassifier(n_estimators = 100,n_jobs=-1)
        

    def train(self, train_data):
        train_images, train_labels = train_data
        self.clf.fit(train_images, train_labels)

    def predict(self, test_data):
        return self.clf.predict_proba(test_data)

    def SaveModel(self, ModelPath):
        dump(self.clf, ModelPath)
    
    def Make(self, pixels_per_cell, cells_per_block, orientations):
        trainPath = r'./Data/flower_data/train'
        testPath = r'./Data/flower_data/valid'

        
        createHogData(trainPath, pixels_per_cell, cells_per_block, orientations)
        createHogData(testPath, pixels_per_cell, cells_per_block, orientations)

        trainPath = r'./DataHog/flower_data/train'
        testPath = r'./DataHog/flower_data/valid'

        hog_train_data, hog_train_labels = hog_train_dataset = loadHogData(trainPath)
        if self.ModelPath is not None:
            if not os.path.exists(self.ModelPath):
                self.clf.fit(hog_train_data,hog_train_labels)
                self.SaveModel(self.ModelPath)
        else:
            self.clf.fit(hog_train_data,hog_train_labels)

        hog_test_data, hog_test_labels = hog_test_dataset = loadHogData(testPath)
        hog_train_dataset = HOG_Dataset(hog_train_data, hog_train_labels)
        hog_test_dataset = HOG_Dataset(hog_test_data, hog_test_labels)
        return hog_train_dataset, hog_test_dataset

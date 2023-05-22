from torchvision import datasets
from timm.loss import LabelSmoothingCrossEntropy
from torch.utils.data import DataLoader, sampler, random_split
import torch as th

import os
import matplotlib.pyplot as plt
import numpy as np
import cv2


from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

from skimage.feature import hog

from multiprocessing import Process
from multiprocessing import Pool

from functools import partial

from joblib import dump, load

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


def loadImages(path):

    dataFolder = datasets.ImageFolder(path)

    def GetImage(imagePath):
        image = cv2.imread(imagePath)
        return image
    imgData = list(map(lambda x: GetImage(x[0]), dataFolder.imgs))
    return imgData, list(map(lambda x: x[1], dataFolder.imgs))

# def GetHogValues(image, pixels_per_cell, cells_per_block, orientations):
#             return hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell,
#                                  cells_per_block=cells_per_block, feature_vector=True, channel_axis=-1)

# def test_hog(pixels_per_cell, cells_per_block, orientations, train_images, train_labels, test_images, test_labels):
        
#     hog_train_images = []
#     coreCount = 16

#     coreResults = [dict()]*coreCount
#     #convert train images to hog values
#     procs = []
#     trainImageCount = len(train_images)
#     trainImageCountForCore = int(trainImageCount/coreCount)
#     p = Pool(processes=20)
#     hog_train_images = p.map(partial(GetHogValues, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, orientations=orientations), train_images)
#     p.close()


#     #convert test images to hog values
#     procs = []
#     testImageCount = len(test_images)
#     trainImageCountForCore = int(trainImageCount/coreCount)
#     for core in range(coreCount):
#         proc_images = test_images[core*trainImageCountForCore: (core+1)*trainImageCountForCore]
#         proc_images_labels = test_labels[core*trainImageCountForCore: (core+1)*trainImageCountForCore]
#         proc = Process(target=GetHogValues, args=(proc_images,proc_images_labels, pixels_per_cell,cells_per_block,orientations, coreResults[core]))
#         procs.append(proc)
#         proc.start()
#     for proc in procs:
#         proc.join()
    
#     hog_test_images = list(map(lambda x: x[0], coreResults))
#     test_labels = list(map(lambda x: x[1], coreResults))

#     # for train_image in train_images:
#     #     hog_image_data = hog(train_image, orientations=orientations, pixels_per_cell=pixels_per_cell,
#     #                          cells_per_block=cells_per_block, feature_vector=True, channel_axis=-1)
#     #     hog_train_images.append(hog_image_data)

#     # hog_test_images = []
#     # for test_image in test_images:
#     #     hog_image_data = hog(test_image, orientations=orientations, pixels_per_cell=pixels_per_cell,
#     #                          cells_per_block=cells_per_block, feature_vector=True, channel_axis=-1)
#     #     hog_test_images.append(hog_image_data)

#     clf = SVC(kernel='rbf', class_weight='balanced')
#     SVC(class_weight='balanced')
#     clf.fit(hog_train_images, train_labels)
#     print(
#         f'''hog accuracy: {clf.score(hog_test_images, test_labels)}
#                 pixels_per_cell: {pixels_per_cell}
#                 cells_per_block: {cells_per_block}
#                 orientations: {orientations}
#             ''')
#     return None

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

def test_hog(pixels_per_cell, cells_per_block, orientations):
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

    
def hog_analysis():

    pixels_per_cell = (32, 32)
    cells_per_block = (4, 4)
    orientations = 10
    test_hog(pixels_per_cell, cells_per_block, orientations)
    
    

def pca_analysis():

    # train data
    trainPath = r'./Data/flower_data/train'
    testPath = r'./Data/flower_data/valid'
    train_data = loadGrayImages(trainPath)
    test_data = loadGrayImages(trainPath)

    train_images, train_labels = train_data
    test_images, test_labels = test_data

    def pca_score(n_components: int):

        pca = PCA(n_components=n_components).fit(train_images)
        x_train_pca = pca.transform(train_images)
        x_test_pca = pca.transform(test_images)

        clf = SVC(kernel='rbf', class_weight='balanced')
        SVC(class_weight='balanced')

        clf.fit(x_train_pca, train_labels)
        print(
            f'pca accuracy with {n_components} components: {clf.score(x_test_pca, test_labels)}')

    pca_score(100)
    pca_score(150)
    pca_score(200)
    pca_score(250)
    pca_score(300)
    pca_score(350)
    return None

def CreateDatbases(n_components):
    trainPath = r'./Data/flower_data/train'
    testPath = r'./Data/flower_data/valid'
    train_data = loadGrayImages(trainPath)

    train_images, train_labels = train_data
    

    pca = PCA(n_components=n_components).fit(train_images)
    x_train_pca = pca.transform(train_images)

    
    test_data = loadGrayImages(trainPath)
    test_images, test_labels = test_data
    x_test_pca = pca.transform(test_images)

    trainDataset = PCA_Dataset(x_train_pca, train_labels)
    testDataset = PCA_Dataset(x_test_pca, test_labels)
    return trainDataset, testDataset

class PCA_Dataset(th.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = list(zip(data, labels))

    def __getitem__(self, i):
        return self.data[i]
    def __len__(self):
        return len(self.data)

class PCA_model:
    def __init__(self, ModelPath=None):
        self.clf = None
        if ModelPath is not None:
            SVC(class_weight='balanced')
            self.clf = load(ModelPath)
        else:
            self.clf = SVC(kernel='rbf', class_weight='balanced')
            SVC(class_weight='balanced')
    def train(self, train_data):
        train_images, train_labels = train_data
        self.clf.fit(train_images, train_labels)
    def predict(self, test_data):
        return self.clf.predict_proba(test_data)

    def SaveModel(self, ModelPath):
        dump(self.clf, ModelPath)
    
    def Make(self, config, n_components):
        trainDataset, testDataset = CreateDatbases(n_components=n_components)
        train_loader = DataLoader(
        trainDataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
        train_data, train_labels =zip(*trainDataset.data)
        train_data, train_labels = list(train_data), list(train_labels)
        self.train(train_data=(train_data, train_labels))
        test_loader = DataLoader(
        testDataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
        return train_loader, test_loader
    
if __name__ == '__main__':
    hog_analysis()
from torchvision import datasets
import torch as th

import os
import numpy as np
import cv2


from sklearn.svm import SVC
from sklearn.decomposition import PCA

from skimage.feature import hog

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


class PCA_Dataset(th.utils.data.Dataset):
    def __init__(self, data, labels):
        data = data.astype(float)
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

    trainDataset = PCA_Dataset(x_train_pca, train_labels)
    testDataset = PCA_Dataset(x_test_pca, test_labels)
    return trainDataset, testDataset


class PCA_model:
    def __init__(self, ModelPath=None):
        self.clf = None
        self.ModelPath = ModelPath
        if ModelPath is not None and os.path.exists(ModelPath):
            SVC(class_weight='balanced')
            self.clf = load(ModelPath)
        else:
            self.clf = SVC(kernel='rbf', class_weight='balanced', probability=True)
            SVC(class_weight='balanced')

    def train(self, train_data):
        train_images, train_labels = train_data
        self.clf.fit(train_images, train_labels)

    def predict(self, test_data):
        return self.clf.predict_proba(test_data)

    def SaveModel(self, ModelPath):
        dump(self.clf, ModelPath)

    def Make(self, n_components):
        trainDataset, testDataset = CreateDatbasesPOC(
            n_components=n_components)
        # train_loader = DataLoader(
        #     trainDataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
        train_data, train_labels = zip(*trainDataset.data)
        train_data, train_labels = list(train_data), list(train_labels)
        if self.ModelPath is not None:
            if not os.path.exists(self.ModelPath):
                self.train(train_data=(train_data, train_labels))
                self.SaveModel(self.ModelPath)
        else:
            self.train(train_data=(train_data, train_labels))
        # test_loader = DataLoader(
        #     testDataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
        return trainDataset, testDataset

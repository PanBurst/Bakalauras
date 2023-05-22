from abc import ABCMeta, abstractmethod


class Model:

    def __init__(self):
        pass

    @abstractmethod
    def train(self, args, dataLoader):
        pass

    @abstractmethod
    def test(self, testLoader):
        pass

    @abstractmethod
    def ConfusionMatrix(self):
        pass

    @abstractmethod
    def SaveModel(self, path):
        pass


config = dict(
    epochs=10,
    learning_rate=0.0001,
    batch_size=7,
    dataset="Oxford 102",
    architecture=None,
    classes=102,
    sample_time=2000,
    datasetPath="./Data/flower_data"
)

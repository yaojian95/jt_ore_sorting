# classifiers/base_classifier.py
from abc import ABC, abstractmethod

class BaseClassifier(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def classify(self, pixels):
        """
        这个方法应由子类实现，用于分类矿石的像素数据。
        :param pixels: 矿石的像素数据
        :return: 分类结果（例如精矿或废矿）
        """
        pass

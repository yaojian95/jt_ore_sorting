# classifiers/base_classifier.py
from abc import ABC, abstractmethod

class BaseClassifier(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def classify(self, pixels, mode, ground_truth=None):
        """
        这个方法应由子类实现，用于分类矿石的像素数据。
        :param pixels: 矿石的像素数据
        :param mode: 工作模式，调优参数或者已有最佳参数直接进行分类。
        :return: 分类结果（例如精矿或废矿）
        """
        if mode == 'tune':
            assert ground_truth is not None, "ground_truth is required in tuning mode"
            # 在这里实现调优参数的逻辑
            pass
        elif mode == 'best':
            # 在这里实现已有最佳参数的逻辑

            if ground_truth is not None:
                #self.calculate_rates()
                pass
            pass
        else:
            raise ValueError("Invalid mode. Choose 'tune' or 'best'.")
    
    def calculate_rates(self, predictions, ground_truth):
        """
        计算分类器的准确率和召回率。
        :param predictions: 分类器的预测结果
        :param ground_truth: 真实的分类结果
        :return: 准确率和召回率
        """
        assert len(predictions) == len(ground_truth), "predictions and ground_truth must have the same length"
        # 计算准确率
        accuracy = sum(p == g for p, g in zip(predictions, ground_truth)) / len(predictions)
        # 计算召回率
        recall = sum(p == g and g == 1 for p, g in zip(predictions, ground_truth)) / sum(g == 1 for g in ground_truth)
        return accuracy, recall

# classifiers/base_classifier.py
from abc import ABC, abstractmethod
# from mpl_toolkits.axes_grid1 import inset_locator
import numpy as np


class BaseClassifier(ABC):
    def __init__(self, name, pixels, truth = None, include_Fe = False):
        self.name = name
        self.pixels = pixels

        if truth is not None:
            self.truth = truth
            self.weight = truth['weight']
            self.pb_grade = truth['Pb_grade'] / 100
            self.zn_grade = truth['Zn_grade'] / 100
            self.fe_grade = truth['Fe_grade'] / 100
            self.include_Fe = include_Fe
            self.pb_zn_fe = (truth['Pb_grade'] + truth['Zn_grade'] + truth['Fe_grade']) / 100

            self.avg_pb_grade_all = np.average(self.pb_grade, weights=self.weight)
            self.avg_zn_grade_all = np.average(self.zn_grade, weights=self.weight)
            self.avg_fe_grade_all = np.average(self.fe_grade, weights=self.weight)

            if not include_Fe:
                self.y = self.pb_grade + self.zn_grade  # 综合品位
            else:
                self.y = self.pb_grade + self.zn_grade + self.fe_grade
            
    @abstractmethod
    def classify_ores(self):
        """
        这个方法应由子类实现，用于分类矿石的像素数据。
        :param pixels: 矿石的像素数据
        :param mode: 工作模式，调优参数或者已有最佳参数直接进行分类。
        :return: 分类结果（例如精矿或废矿）
        """
        pass

    @abstractmethod
    def tuning(self):
        """
        这个方法应由子类实现，用于调优分类器的参数。
        :param pixels: 矿石的像素数据
        :param ground_truth: 真实的分类结果
        :return: 调优后的参数
        """

        assert self.truth is not None, "ground_truth is required in tuning mode"
        pass

    def safe_average(self, values, weights):
        if len(values) == 0 or len(weights) == 0 or weights.sum() == 0:
            return 0.0
        return np.average(values, weights=weights)

    def calculate_tuning_metrics(self, predictions: np.ndarray):
        """
        计算用于超参数调优的指标，与品位阈值无关。

        参数:
        - predictions (np.ndarray): 预测结果数组（1 表示高品位，0 表示低品位）。

        返回:
        - Dict[str, float]: 包含调优指标的字典。
        """
        high_grade_mask = predictions == 1
        low_grade_mask = predictions == 0

        # 计算抛废率和回收率
        scrap_rate = self.weight[low_grade_mask].sum() / self.weight.sum()
        recovery_rate = (self.weight[high_grade_mask] * self.y[high_grade_mask]).sum() / (self.weight * self.y).sum()

        # 计算富集率
        # avg_pb_grade_all = self.pb_grade.mean()
        # avg_zn_grade_all = self.zn_grade.mean()
        # avg_pb_grade_high = np.nan_to_num(self.pb_grade[high_grade_mask].mean())
        # avg_zn_grade_high = np.nan_to_num(self.zn_grade[high_grade_mask].mean())
        # avg_pb_grade_low = np.nan_to_num(self.pb_grade[low_grade_mask].mean())
        # avg_zn_grade_low = np.nan_to_num(self.zn_grade[low_grade_mask].mean())

        avg_pb_grade_high = self.safe_average(self.pb_grade[high_grade_mask], weights=self.weight[high_grade_mask])
        avg_zn_grade_high = self.safe_average(self.zn_grade[high_grade_mask], weights=self.weight[high_grade_mask])
        avg_pb_grade_low = self.safe_average(self.pb_grade[low_grade_mask], weights=self.weight[low_grade_mask])
        avg_zn_grade_low = self.safe_average(self.zn_grade[low_grade_mask], weights=self.weight[low_grade_mask])
        avg_fe_grade_low = self.safe_average(self.fe_grade[low_grade_mask], weights=self.weight[low_grade_mask])
        avg_fe_grade_high = self.safe_average(self.fe_grade[high_grade_mask], weights=self.weight[high_grade_mask])

        enrichment_Pb = avg_pb_grade_high / self.avg_pb_grade_all if self.avg_pb_grade_all != 0 else 0
        enrichment_Zn = avg_zn_grade_high / self.avg_zn_grade_all if self.avg_zn_grade_all != 0 else 0

        return {
            '抛废率': '{:.2f}%'.format(scrap_rate*100),
            '回收率': '{:.2f}%'.format(recovery_rate*100),
            '铅富集比': '{:.2f}'.format(enrichment_Pb),
            '锌富集比': '{:.2f}'.format(enrichment_Zn),
            '铅、锌、铁平均品位（所有）': '{:.2f}%, {:.2f}%, {:.2f}%'.format(self.avg_pb_grade_all*100, self.avg_zn_grade_all*100, self.avg_fe_grade_all*100),
            '铅、锌、铁平均品位（保留）': '{:.2f}%, {:.2f}%, {:.2f}%'.format(avg_pb_grade_high*100, avg_zn_grade_high*100, avg_fe_grade_high*100),
            '铅、锌、铁平均品位（抛废）': '{:.2f}%, {:.2f}%, {:.2f}%'.format(avg_pb_grade_low*100, avg_zn_grade_low*100, avg_fe_grade_low*100),
        }
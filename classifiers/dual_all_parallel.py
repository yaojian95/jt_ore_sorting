# classifiers/dual_mean.py
# wrap all algorithms (dual_thresh, dual_thresh_mean) into this class to make use of multiprocessing. 
# But dual_thresh seems doesn't need the parallel.  
# Before parallel: dual_thresh 1.3s, dual_thresh_mean 3m31s; 
# After: 2.8s, 53s

from .base_classifier import BaseClassifier
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import itertools
from multiprocessing import Pool, cpu_count

class ParallelClassifier(BaseClassifier):
    def __init__(self, pixels, pixel_kind = 'grayness', truth = None, include_Fe = False):
        '''
        初始化 DualThreshClassifier 类。
        参数:
        - pixels: 矿石的像素数据。
        - ground_truth: 真实的分类结果（可选）。
        - pixel_kind: 输入的是灰度图还是R值图, 默认为 'grayness', grayness or R, 不同的图对应的分选逻辑不一样。

        '''
        super().__init__(name = 'Dual_thresh', pixels = pixels, truth = truth, include_Fe=include_Fe)
        self.pixel_kind = pixel_kind
        
    def classify_ores(self, I_th, ratio_th, mean_th=None):
        """
        基于“灰度阈值”和“比例阈值”将矿石分类为高品位 (1) 或低品位 (0)。
        如果提供mean_th，将进一步根据平均值进行二次筛选。
        
        Parameters
        ----------
        :param I_th: 灰度或R值的阈值 (int)
        :param ratio_th: 比例阈值 (float)
        :param mean_th: 平均值阈值 (float)，可选参数。
        
        Returns
        ----------
        :return: 预测结果数组（1 表示高品位，0 表示低品位）。
        """
        try:
            # 计算每个样本的低像素比例
            if self.pixel_kind == 'grayness':
                low_pixel_ratios = self.pixels.apply(lambda x: (np.array(x) < I_th).mean())
            elif self.pixel_kind == 'R':
                low_pixel_ratios = self.pixels.apply(lambda x: (np.array(x) > I_th).mean())
            else:
                raise ValueError(f"Unknown pixel_kind: {self.pixel_kind}")

            pre_th = (low_pixel_ratios > ratio_th).astype(int).values

            if mean_th is None:
                # 只根据比例阈值分类
                return pre_th
            else:
                # 还需要进一步根据平均值进行筛选
                mean_value = self.pixels.apply(np.mean)

                if self.pixel_kind == 'grayness':
                    pre_mean = (mean_value < mean_th).astype(int).values
                elif self.pixel_kind == 'R':
                    pre_mean = (mean_value > mean_th).astype(int).values

                # 两者都满足才是高品位
                predictions = np.logical_and(pre_th, pre_mean)
                return predictions

        except Exception as e:
            print(f"Error during classification: {e}")
            raise

    def tuning(
            self,
            min_recovery_rate=None,
            min_scrap_rate=None,
            A_range=np.arange(0, 256, 5), 
            step_B=0.05, 
            mean_range=np.arange(0.5, 1.2, 0.01),
            grade_real_th=None
    ):
        """
        调优超参数以找到基于指定约束条件的最佳“灰度阈值”和“比例阈值”。

        参数:
        - min_recovery_rate (float, optional): 最低可接受的回收率。
        - min_scrap_rate (float, optional): 最低可接受的抛废率。
        - step_B (float): “比例阈值”的步长。
        - grade_real_th (float, optional): 真值对应的品位（以铅锌铁为对象）
        - A_range (np.ndarray): “灰度阈值”的取值范围。
        约束条件:
        - 回收率 >= min_recovery_rate
        - 抛废率 >= min_scrap_rate

        返回:
        - None，结果存储在对象属性中，包括:
            - self.tuning_results: 所有参数组合的调优结果（DataFrame）
            - self.best_params: 最佳参数组合 (threshold_A, threshold_B, threshold_C)
            - self.best_metrics: 与最佳参数对应的指标
            - self.pareto_front: 根据调优结果计算的 Pareto 前沿
        """
        
        # 确保已设置 ground truth 数据
        assert self.truth is not None, "ground_truth is required in tuning mode"
        self.best_under_constraints = None
        
        self.grade_real_th = grade_real_th
        best_metrics = None
        best_params = None
        best_score = -np.inf  # 初始化一个非常小的数，用于最大化目标

        threshold_A_steps = A_range
        threshold_B_steps = np.arange(0, 1.01, step_B)
        th_C_steps = mean_range

        # 生成所有参数组合列表
        if mean_range is not None:
            all_params = list(itertools.product(threshold_A_steps, threshold_B_steps, th_C_steps))
        else:
            all_params = list(itertools.product(threshold_A_steps, threshold_B_steps))

        # 如果在 Windows 系统上运行，请确保在 __name__=="__main__" 下执行下面的代码
        with Pool(processes=cpu_count()) as pool:
            results = pool.map(self.evaluate_combo, all_params)
        
        tuning_results = []
        # 遍历所有并行处理的结果，并检查是否满足用户指定的约束条件
        for res in results:
            tuning_results.append(res)
            meets_recovery = True
            meets_scrap = True

            if min_recovery_rate is not None and res['回收率'] < min_recovery_rate:
                meets_recovery = False
            if min_scrap_rate is not None and res['抛废率'] < min_scrap_rate:
                meets_scrap = False

            if (min_recovery_rate is None or meets_recovery) and (min_scrap_rate is None or meets_scrap):
                # 定义优化分数，可以根据用户的约束条件做加权或简单求和
                if min_recovery_rate is not None and min_scrap_rate is not None:
                    current_score = res['回收率'] + res['抛废率']
                elif min_recovery_rate is not None:
                    current_score = res['抛废率']
                elif min_scrap_rate is not None:
                    current_score = res['回收率']
                else:
                    current_score = res['回收率'] + res['抛废率']

                if current_score > best_score:
                    best_score = current_score
                    best_metrics = res['all_metrics']
                    best_params = (res['threshold_A'], res['threshold_B'], res['threshold_C'])
                    self.best_under_constraints = (res['threshold_A'], res['threshold_B'], res['threshold_C'], best_metrics)
        
        # 将调优结果列表转换为 DataFrame，便于进一步分析和展示
        self.tuning_results = pd.DataFrame(tuning_results)
        # 根据实际需要调用计算 Pareto 前沿的方法
        self.pareto_front = self._compute_pareto_front(self.tuning_results)
        self.best_params = best_params
        self.best_metrics = best_metrics

        # return best_params, best_metrics

    def evaluate_combo(self, params):
        """
        对一个参数组合执行评估：计算预测结果、各项调优指标、品位阈值、准确率，并返回一个字典结果
        """
        if len(params) == 2:
            I_th, ratio_th = params
            mean_th = None  # 不使用平均值
        elif len(params) == 3:
            I_th, ratio_th, mean_th = params

        predictions = self.classify_ores(I_th, ratio_th, mean_th)
        tuning_metrics = self.calculate_tuning_metrics(predictions)
        
        # 计算品位阈值：如果全部预测为“抛废”，则取所有样本的最大品位；否则取高低品位交界处的平均值。
        if predictions.sum() == 0:
            grade_threshold = self.y.max()
        else:
            # 注意这里假设 self.y 支持 pandas.Series 的 empty 属性
            low_grade = self.y[predictions == 0]
            high_grade = self.y[predictions == 1]
            low_grade_max = low_grade.max() if not low_grade.empty else 0
            high_grade_min = high_grade.min() if not high_grade.empty else 0
            grade_threshold = (low_grade_max + high_grade_min) / 2

        true_labels = (self.pb_zn_fe >= self.grade_real_th).astype(int)
        accuracy = accuracy_score(true_labels, predictions)
        tuning_metrics['准确率'] = accuracy

        result = {
            'threshold_A': I_th,
            'threshold_B': ratio_th,
            'threshold_C': mean_th,
            '抛废率': tuning_metrics['抛废率'],
            '回收率': tuning_metrics['回收率'],
            '品位阈值': grade_threshold,
            '铅富集比': tuning_metrics['铅富集比'],
            '锌富集比': tuning_metrics['锌富集比'],
            '准确率': accuracy,
            'all_metrics': tuning_metrics  # 将所有指标都传回，方便后续进一步处理
        }
        return result
    
    def _compute_pareto_front(self, df):
        """
        计算 Pareto 前沿。

        参数:
        - df (pd.DataFrame): 包含调优结果的 DataFrame。

        返回:
        - pd.DataFrame: 仅包含 Pareto 前沿点的 DataFrame。
        """
        # 按回收率降序排序，如果回收率相同，则按抛废率降序排序
        df_sorted = df.sort_values(['回收率', '抛废率'], ascending=[False, False]).reset_index(drop=True)
        pareto_front = []
        current_max_scrap = -np.inf

        for _, row in df_sorted.iterrows():
            if row['抛废率'] > current_max_scrap:
                pareto_front.append(row)
                current_max_scrap = row['抛废率']

        return pd.DataFrame(pareto_front)
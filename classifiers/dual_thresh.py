# classifiers/dual_thresh.py
from .base_classifier import BaseClassifier
import logging
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class DualThreshClassifier(BaseClassifier):
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

    def classify_ores(self, I_th, ratio_th):

        """
        基于“灰度阈值”和“比例阈值”将矿石分类为高品位 (1) 或低品位 (0)。
        对于灰度图, 平均像素值高意味着品位低；
        对于R值图, 平均像素值高意味着品位高。

        Parameters
        ----------
        I_th: 灰度阈值 (int): 像素低于该值的阈值。
        ratio_th: 比例阈值 (float): 像素低于“灰度阈值”的比例阈值。

        Returns
        ----------
        return: 预测结果数组（1 表示高品位，0 表示低品位）。
        """
        try:
            # 计算每个样本的低像素比例
            if self.pixel_kind == 'grayness':
                low_pixel_ratios = self.pixels.apply(lambda x: (np.array(x) < I_th).mean())
            elif self.pixel_kind == 'R':
                low_pixel_ratios = self.pixels.apply(lambda x: (np.array(x) > I_th).mean())
            predictions = (low_pixel_ratios > ratio_th).astype(int).values
            return predictions
        except Exception as e:
            logging.error(f"分类时出错: {e}")
            return np.zeros(len(self.pixels))  # 若出错，则默认全部为低品位

    def tuning(
            self,
            min_recovery_rate = None,
            min_scrap_rate = None,
            A_range = np.arange(0, 256, 5), 
            step_B: float = 0.05, 
            grade_real_th = None, 
            score_on = False,
            score_weight = False,
            check_metrics = True
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
        - Tuple 包含:
            - best_params (Optional[Tuple[int, float]]): 优化目标的 (“灰度阈值”, “比例阈值”)。
            - best_metrics (Optional[Dict[str, float]]): 与 best_params 对应的指标。
        """

        assert self.truth is not None, "ground_truth is required in tuning mode"
        self.tuning_results = []  # 存储调优结果的列表
        self.best_under_constraints = None

        best_metrics = None
        best_params = None
        best_score = -np.inf  # 初始化为负无穷，用于最大化

        threshold_A_steps = A_range  # “灰度阈值”从 0 到 255，步长为 step_A
        threshold_B_steps = np.arange(0, 1.01, step_B)  # “比例阈值”从 0.0 到 1.0，步长为 step_B

        # logging.info("开始进行超参数调优...")
        for I_th in threshold_A_steps:
            for ratio_th in threshold_B_steps:
                predictions = self.classify_ores(I_th, ratio_th)
                tuning_metrics = self.calculate_tuning_metrics(predictions)

                # 计算品位阈值
                if predictions.sum() == 0:
                    # 所有矿石为抛废，设置品位阈值为所有矿石的最大品位
                    grade_threshold = self.y.max()
                else:
                    # 存在高品位矿石，计算低品位中的最大值和高品位中的最小值的平均值
                    low_grade_max = self.y[predictions == 0].max() if not self.y[predictions == 0].empty else 0
                    high_grade_min = self.y[predictions == 1].min() if not self.y[predictions == 1].empty else 0
                    grade_threshold = (low_grade_max + high_grade_min) / 2

                # 计算准确率等
                # if np.isscalar(grade_real_th):
                true_labels = (self.y >= grade_real_th).astype(int)

                if score_on == True:
                    
                    accuracy = accuracy_score(true_labels, predictions)
                    precision = precision_score(true_labels, predictions, zero_division=0)
                    recall = recall_score(true_labels, predictions, zero_division=0)

                    if score_weight == True:
                        accuracy_weight = accuracy_score(true_labels, predictions, sample_weight=self.weight)
                        precision_weight = precision_score(true_labels, predictions, sample_weight=self.weight, zero_division=0)
                        recall_weight = recall_score(true_labels, predictions, sample_weight=self.weight, zero_division=0)

                        accuracy_grade = accuracy_score(true_labels, predictions, sample_weight=self.y)
                        precision_grade = precision_score(true_labels, predictions, sample_weight=self.y, zero_division=0)
                        recall_grade = recall_score(true_labels, predictions, sample_weight=self.y, zero_division=0)

                    else:

                        accuracy_weight = precision_weight = recall_weight = 0
                        accuracy_grade = precision_grade = recall_grade = 0

                else:
                    accuracy = accuracy_weight = accuracy_grade = 0
                    precision = precision_weight = precision_grade = 0
                    recall = recall_weight = recall_grade = 0
                    
                # 记录调优结果，包括品位阈值
                self.tuning_results.append({
                    'threshold_A': I_th,
                    'threshold_B': ratio_th,
                    '抛废率': tuning_metrics['抛废率'],
                    '回收率': tuning_metrics['回收率'],
                    '品位阈值': grade_threshold,
                    '铅富集比': tuning_metrics['铅富集比'],
                    '锌富集比': tuning_metrics['锌富集比'],            
                    '准确率': accuracy,
                    '精确率': precision,
                    '召回率': recall,
                    '准确率_重量': accuracy_weight,
                    '精确率_重量': precision_weight,
                    '召回率_重量': recall_weight,
                    '准确率_品位': accuracy_grade,
                    '精确率_品位': precision_grade,
                    '召回率_品位': recall_grade,
                    # 'F1 分数': f1
    
                })
                
                if check_metrics == False:
                    continue
                    
                else:

                    # 检查是否满足约束条件
                    meets_recovery = True
                    meets_scrap = True

                    if min_recovery_rate is not None and tuning_metrics['回收率'] < min_recovery_rate:
                        meets_recovery = False
                    if min_scrap_rate is not None and tuning_metrics['抛废率'] < min_scrap_rate:
                        meets_scrap = False

                    if (min_recovery_rate is None or meets_recovery) and (min_scrap_rate is None or meets_scrap):
                        # 定义优化分数
                        if min_recovery_rate is not None and min_scrap_rate is not None:
                            current_score = tuning_metrics['回收率'] + tuning_metrics['抛废率']
                        elif min_recovery_rate is not None:
                            current_score = tuning_metrics['抛废率']
                        elif min_scrap_rate is not None:
                            current_score = tuning_metrics['回收率']
                        else:
                            current_score = tuning_metrics['回收率'] + tuning_metrics['抛废率']

                        # 如果当前分数更高，则更新最佳参数
                        if current_score > best_score:
                            best_score = current_score
                            tuning_metrics.update({'准确率': accuracy})
                            best_metrics = tuning_metrics
                            best_params = (I_th, ratio_th)
                            self.best_under_constraints = (I_th, ratio_th, best_metrics)

        self.tuning_results = pd.DataFrame(self.tuning_results)
        
        if check_metrics == False:
            return
        
        else:
            # 计算 Pareto 前沿
            self.pareto_front = self._compute_pareto_front(self.tuning_results)
            self.best_params = best_params
            self.best_metrics = best_metrics

        # return best_params, best_metrics
    
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
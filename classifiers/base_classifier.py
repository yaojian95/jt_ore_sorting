# classifiers/base_classifier.py
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import io
import sys

class BaseClassifier(ABC):
    def __init__(self, name, pixels, truth = None):
        self.name = name
        self.pixels = pixels

        if truth is not None:
            self.truth = truth
            self.pb_grade = truth['Pb_grade'] / 100
            self.zn_grade = truth['Zn_grade'] / 100
            self.y = self.pb_grade + self.zn_grade  # 综合品位
            self.weight = truth['weight']
            

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
    

    def calculate_grade_based_metrics(self, predictions: np.ndarray, grade_threshold: float) -> Dict[str, float]:
        """
        计算依赖于品位阈值的分类指标。

        参数:
        - predictions (np.ndarray): 预测结果数组（1 表示高品位，0 表示低品位）。
        - grade_threshold (float): 基于综合品位确定高品位矿石的阈值。

        返回:
        - Dict[str, float]: 包含分类指标的字典。
        """
        # 基于品位阈值计算分类指标
        true_labels = (self.y >= grade_threshold).astype(int)
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, zero_division=0)
        recall = recall_score(true_labels, predictions, zero_division=0)
        f1 = f1_score(true_labels, predictions, zero_division=0)

        return {
            '准确率': accuracy,
            '精确率': precision,
            '召回率': recall,
            'F1 分数': f1
        }

    def _compute_rates(self, sorting_positions: np.ndarray) -> Tuple[list, list, list]:
        """
        根据（综合品位升序）排序位置计算抛废率、回收率和品位阈值。

        参数:
        - sorting_positions (np.ndarray): 排序后的矿石位置索引。

        返回:
        - Tuple[list, list, list]: 抛废率、回收率、品位阈值的列表。
        """
        sorted_weights = self.weight.iloc[sorting_positions].values
        sorted_grades = self.y.iloc[sorting_positions].values
        sorted_valuable_material = sorted_grades * sorted_weights

        total_weight = np.sum(sorted_weights)
        total_valuable_material = np.sum(sorted_valuable_material)

        scrap_rates = []
        recovery_rates = []
        grade_thresholds = []

        cumulative_weight_scrapped = 0
        cumulative_valuable_material_scrapped = 0

        n_rocks = len(self.y)
        for i in range(n_rocks + 1):
            # 计算抛废率
            scrap_rate = cumulative_weight_scrapped / total_weight
            scrap_rates.append(scrap_rate)

            # 计算回收率
            valuable_material_kept = total_valuable_material - cumulative_valuable_material_scrapped
            recovery_rate = valuable_material_kept / total_valuable_material
            recovery_rates.append(recovery_rate)

            # 记录品位阈值（当前矿石的品位）
            if i == n_rocks:
                # 所有矿石均被抛废，设置品位阈值为最大品位
                grade_threshold = self.y.max()
            else:
                # 保留矿石的最低品位
                grade_threshold = sorted_grades[i]
            grade_thresholds.append(grade_threshold)

            if i < n_rocks:
                # 更新累计被抛废的重量和有价物料
                cumulative_weight_scrapped += sorted_weights[i]
                cumulative_valuable_material_scrapped += sorted_valuable_material[i]

        return scrap_rates, recovery_rates, grade_thresholds
    
    def compute_ideal_curve(self) -> Tuple[list, list, list]:
        """
        基于按综合品位升序排序计算理想曲线。

        返回:
        - Tuple 包含:
            - scrap_rates_ideal (list): 理想抛废率。
            - recovery_rates_ideal (list): 理想回收率。
            - grade_thresholds (list): 与每个理想点对应的品位阈值。
        """
        # 按综合品位升序排序矿石
        sorted_indices = np.argsort(self.y.values)
        scrap_rates_ideal, recovery_rates_ideal, grade_thresholds = self._compute_rates(sorted_indices)
        return scrap_rates_ideal, recovery_rates_ideal, grade_thresholds
    
    def compute_auc(self, scrap_rates_ideal: list, recovery_rates_ideal: list) -> Dict[str, float]:
        """
        计算理想曲线和实际 Pareto 前沿曲线的曲线下面积（AUC）。

        参数:
        - scrap_rates_ideal (list): 理想抛废率。
        - recovery_rates_ideal (list): 理想回收率。

        返回:
        - Dict[str, float]: 包含理想曲线和实际曲线 AUC 的字典。
        """
        auc_results = {}

        # 使用梯形法则计算理想曲线的 AUC
        auc_ideal = np.trapz(recovery_rates_ideal, scrap_rates_ideal)
        auc_results['AUC_基于化验品位'] = auc_ideal

        # 计算实际曲线（Pareto 前沿）的 AUC
        if not self.pareto_front.empty:
            # 按抛废率排序 Pareto 前沿点
            pareto_sorted = self.pareto_front.sort_values('抛废率')
            scrap_rates_actual = pareto_sorted['抛废率'].values
            recovery_rates_actual = pareto_sorted['回收率'].values

            auc_actual = np.trapz(recovery_rates_actual, scrap_rates_actual)
            auc_results['AUC_双阈值算法'] = auc_actual
        else:
            auc_results['AUC_双阈值算法'] = np.nan
            logging.warning("没有可用的调优结果来计算实际 AUC。")

        # 使用梯形法则计算基线曲线的 AUC
        try:
            scrap_rates_baseline, recovery_rates_baseline, _ = self.compute_baseline_curve()
            auc_baseline = np.trapz(recovery_rates_baseline, scrap_rates_baseline)
            auc_results['AUC_平均值算法'] = auc_baseline
        except Exception as e:
            auc_results['AUC_平均值算法'] = np.nan
            logging.error(f"计算基线 AUC 时出错: {e}")

        return auc_results

    def find_closest_point_1(self, target_recovery: float) -> Optional[Tuple[float, float]]:
        """
        找到回收率最接近目标回收率的抛废率。

        参数:
        - target_recovery (float): 目标回收率（例如 0.95 表示 95%）。

        返回:
        - Tuple[float, float]: (对应的抛废率, 对应的回收率) 或 None 如果未找到。
        """
        scrap_rates = self.pareto_front['抛废率'].values
        recovery_rates = self.pareto_front['回收率'].values

        if len(recovery_rates) == 0:
            logging.warning("调优结果为空，无法找到最接近的点。")
            return None

        # 计算与目标回收率的绝对差
        differences = np.abs(recovery_rates - target_recovery)
        min_index = np.argmin(differences)
        closest_scrap = scrap_rates[min_index]
        closest_recovery = recovery_rates[min_index]
        return closest_scrap, closest_recovery
    
    def find_closest_point(self, target= [0.2, 0.95]):
        '''
        找到回收率最接近目标回收率的抛废率， 或者最接近目标抛废率的回收率。

        :param scrap_rates: 抛废率列表
        :param recovery_rates: 回收率列表
        :param target: 目标回收率

        :return: 最接近目标回收率的抛废率, and indexes

        '''
        scrap_rates = np.array(self.pareto_front['抛废率'].values)
        recovery_rates = np.array(self.pareto_front['回收率'].values)
        
        min_index_scrap = np.argmin(np.abs(scrap_rates - target[0]))
        min_index_recovery = np.argmin(np.abs(recovery_rates - target[1]))

        closest_scrap = [min_index_scrap, scrap_rates[min_index_scrap], recovery_rates[min_index_scrap]]
        closest_recovery = [min_index_recovery, scrap_rates[min_index_recovery], recovery_rates[min_index_recovery]]
        return [closest_scrap, closest_recovery]


    def report_closest_scrap_rate(self, target_recovery: float) -> Optional[Tuple[float, float]]:
        """
        找到并报告回收率最接近目标回收率的抛废率。

        参数:
        - target_recovery (float): 目标回收率（例如 0.95 表示 95%）。

        返回:
        - Optional[Tuple[float, float]]: 最接近的 (抛废率, 回收率) 或 None 如果未找到。
        """
        closest_point = self.find_closest_point(target_recovery)
        if closest_point:
            scrap, recovery = closest_point
            logging.info(f"当回收率约为 {recovery*100:.2f}% 时，抛废率为 {scrap*100:.2f}%。")
        else:
            logging.info("未找到符合条件的抛废率和回收率。")
        return closest_point

    def report_rates(self):
        '''
        print the detailed results

        Returns:
        --------
        res_dict: dict
            a dictionary containing the detailed results, including the ideal curve, baseline curve, and pareto front.
        '''
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        
        if self.best_params:
            best_threshold_A, best_threshold_B = self.best_params
            print("\n=== 最佳超参数 ===")
            print(f"灰度阈值: {best_threshold_A}")
            print(f"比例阈值: {best_threshold_B * 100:.2f}%")

            # 使用最佳超参数进行矿石分类
            predictions = self.classify_ores(best_threshold_A, best_threshold_B)

            grade_threshold = 0.025
            # 使用最佳超参数计算基于品位的指标
            grade_based_metrics = self.calculate_grade_based_metrics(
                predictions=predictions,
                grade_threshold=grade_threshold
            )

            print("\n=== 调优指标 ===")
            for metric, value in self.best_metrics.items():
                print(f"{metric}: {value:.2%}")

            print(f"\n=== 基于品位阈值{grade_threshold*100:.2f}%的二分类指标 ===")
            for metric, value in grade_based_metrics.items():
                print(f"{metric}: {value:.2%}")
        else:
            print("未找到符合指定约束条件的合适参数。")

        # 计算曲线数据

        res_dict = {}

        try:
            # scrap_rates_ideal, recovery_rates_ideal, grade_thresholds_ideal = self.compute_ideal_curve()
            # scrap_rates_baseline, recovery_rates_baseline, grade_thresholds_baseline = self.compute_baseline_curve()
            res_dict['ideal'] = self.compute_ideal_curve()
            res_dict['baseline'] = self.compute_baseline_curve() # 定义在dual_thresh.py 里面。 
            scrap_rates_ideal = res_dict['ideal'][0]
            recovery_rates_ideal = res_dict['ideal'][1]

            pareto_front = self.pareto_front
            pareto_sorted = self.pareto_front.sort_values('抛废率')
            res_dict['dual_thresh'] = [pareto_sorted['抛废率'].values, pareto_sorted['回收率'].values]

        except Exception as e:
            logging.error(f"计算曲线时出错: {e}")

        # 计算 AUC 结果
        auc_results = self.compute_auc(scrap_rates_ideal, recovery_rates_ideal)

        # 显示 AUC 结果
        print("\n=== AUC 结果 ===")
        for key, value in auc_results.items():
            if not np.isnan(value):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: 未计算")

        # 识别最佳点
        best_sum_point = None
        best_enrichment_point = None
        best_constraint_point = None

        if not pareto_front.empty:
            # 1. 最佳总和点（最大抛废率 + 回收率）； 不一定满足抛废率和回收率>95的要求
            pareto_front['score_sum'] = pareto_front['抛废率'] + pareto_front['回收率']
            best_sum_point = pareto_front.loc[pareto_front['score_sum'].idxmax()]

            # 2. 最佳富集点（最大铅和锌富集比之和）； 满足回收率和抛废率要求下的 双率之和最大的点
            pareto_front['enrichment_sum'] = pareto_front['铅富集比'] + pareto_front['锌富集比']
            best_enrichment_point = pareto_front.loc[pareto_front['enrichment_sum'].idxmax()]

            res_dict['best_sum_point'] = best_sum_point
            res_dict['best_enrichment_point'] = best_enrichment_point

            # 3. 最佳约束点（如果存在）
            if self.best_under_constraints:
                best_constraint_point = pd.Series({
                    'threshold_A': self.best_under_constraints[0],
                    'threshold_B': self.best_under_constraints[1],
                    '抛废率': self.best_under_constraints[2]['抛废率'],
                    '回收率': self.best_under_constraints[2]['回收率'],
                    '铅富集比': self.best_under_constraints[2]['铅富集比'],
                    '锌富集比': self.best_under_constraints[2]['锌富集比'],
                    '品位阈值': (self.best_under_constraints[2]['抛废率'] +
                                    self.best_under_constraints[2]['回收率']) / 2
                })
                res_dict['best_constraint_point'] = best_constraint_point
        else:
            logging.warning("没有可用的 Pareto 前沿来识别最佳点。")

        # 找到最接近目标回收率的点
        # target_recovery = 0.95
        closest_point = self.find_closest_point(target = [0.2, 0.95])

        # 获取 closest_point 对应的 灰度阈值 和 比例阈值
        if closest_point:
            print(f"\n=== 20%抛废率和95%回收率指标 ===")
            for point in closest_point:
                id, scrap, recovery = point
                matching_rows = pareto_front[
                    (pareto_front['抛废率'] == scrap) & (pareto_front['回收率'] == recovery)
                ]
                if not matching_rows.empty:
                    closest_threshold_A = matching_rows.iloc[0]['threshold_A']
                    closest_threshold_B = matching_rows.iloc[0]['threshold_B']
                    print(f"当回收率约为 {recovery * 100:.2f}% 时，抛废率为 {scrap * 100:.2f}%。此时的灰度阈值为{closest_threshold_A}，比例阈值为{closest_threshold_B * 100:.2f}%。")
                else:
                    print(f"当回收率约为 {recovery * 100:.2f}% 时，抛废率为 {scrap * 100:.2f}%。阈值信息不可用。")
        else:
            print("未找到符合条件的抛废率和回收率。")
        
        output = new_stdout.getvalue()
        sys.stdout = old_stdout

        return res_dict, output
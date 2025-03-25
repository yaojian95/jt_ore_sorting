# BaselineOreGrading.py

import numpy as np
import pandas as pd
import logging
import pickle
from typing import Tuple, Dict, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class BaselineOreGrading:
    """
    BaselineOreGrading 是一个用于执行基线矿石分级的类，通过基于抛废率和回收率的超参数调优进行优化。
    """

    def __init__(self, ann: pd.DataFrame, pixels: pd.Series, pixel_kind = 'grayness'):
        """
        初始化 BaselineOreGrading 类。

        参数:
        - ann (pd.DataFrame): 包含矿石信息的注释数据。必须包含 'weight'、'Pb_grade'、'Zn_grade' 列。
        - pixels (pd.Series): 每个矿石的像素数据，以列表或数组形式。
        - pixel_kind (str): 像素数据的类型，可以是 'grayness' 或 'R'。
        """
        self.data = ann
        self.pixels = pixels
        self.weight = self.data['weight']
        self.pb_grade = self.data['Pb_grade'] / 100
        self.zn_grade = self.data['Zn_grade'] / 100
        self.y = self.pb_grade + self.zn_grade  # 综合品位
        self.pixel_kind = pixel_kind

        # 初始化用于调优结果的存储列表
        self.tuning_results = []

        # 存储满足约束条件的最佳点
        self.best_under_constraints = None

        # 设置 Matplotlib 使用支持中文的字体
        self._set_chinese_font()

    def _set_chinese_font(self):
        """
        设置 Matplotlib 使用支持中文的字体。
        优先使用 SimHei；如果不可用，则使用其他中文字体。
        """
        chinese_fonts = ['SimHei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'Microsoft YaHei', 'PingFang SC']
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        for font in chinese_fonts:
            if font in available_fonts:
                plt.rcParams['font.sans-serif'] = [font]
                break
        else:
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 默认字体
            logging.warning("指定的中文字体未找到。使用默认字体。")
        plt.rcParams['axes.unicode_minus'] = False  # 修正负号显示

    def classify_ores(self, I_th: int, ratio_th: float) -> np.ndarray:
        """
        基于“灰度阈值”和“比例阈值”将矿石分类为高品位 (1) 或低品位 (0)。
        对于灰度图, 平均像素值高意味着品位低；
        对于R值图, 平均像素值高意味着品位高。

        参数:
        - I_th, 灰度阈值 (int): 像素低于该值的阈值。
        - ratio_th, 比例阈值 (float): 像素低于“灰度阈值”的比例阈值。
        - input (str): 输入的是灰度图还是R值图, grayness or R, 不同的图对应的分选逻辑不一样。

        返回:
        - np.ndarray: 预测结果数组（1 表示高品位，0 表示低品位）。
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

    def calculate_tuning_metrics(self, predictions: np.ndarray) -> Dict[str, float]:
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
        avg_pb_grade_all = self.pb_grade.mean()
        avg_zn_grade_all = self.zn_grade.mean()
        avg_pb_grade_high = np.nan_to_num(self.pb_grade[high_grade_mask].mean())
        avg_zn_grade_high = np.nan_to_num(self.zn_grade[high_grade_mask].mean())
        avg_pb_grade_low = np.nan_to_num(self.pb_grade[low_grade_mask].mean())
        avg_zn_grade_low = np.nan_to_num(self.zn_grade[low_grade_mask].mean())

        enrichment_Pb = avg_pb_grade_high / avg_pb_grade_all if avg_pb_grade_all != 0 else 0
        enrichment_Zn = avg_zn_grade_high / avg_zn_grade_all if avg_zn_grade_all != 0 else 0

        return {
            '抛废率': scrap_rate,
            '回收率': recovery_rate,
            '铅富集比': enrichment_Pb,
            '锌富集比': enrichment_Zn,
            '铅平均品位（保留）': avg_pb_grade_high,
            '锌平均品位（保留）': avg_zn_grade_high,
            '铅平均品位（抛废）': avg_pb_grade_low,
            '锌平均品位（抛废）': avg_zn_grade_low,
            '铅平均品位': avg_pb_grade_all,
            '锌平均品位': avg_zn_grade_all,
        }


    def hyperparameter_tuning(
            self,
            min_recovery_rate: Optional[float] = None,
            min_scrap_rate: Optional[float] = None,
            A_range = np.arange(0, 256, 5), 
            step_B: float = 0.05, 
    ) -> Tuple[Optional[Tuple[int, float]], Optional[Dict[str, float]]]:
        """
        调优超参数以找到基于指定约束条件的最佳“灰度阈值”和“比例阈值”。

        参数:
        - min_recovery_rate (float, optional): 最低可接受的回收率。
        - min_scrap_rate (float, optional): 最低可接受的抛废率。
        - step_A (int): “灰度阈值”的步长。
        - step_B (float): “比例阈值”的步长。

        返回:
        - Tuple 包含:
            - best_params (Optional[Tuple[int, float]]): 优化目标的 (“灰度阈值”, “比例阈值”)。
            - best_metrics (Optional[Dict[str, float]]): 与 best_params 对应的指标。
        """
        best_metrics = None
        best_params = None
        best_score = -np.inf  # 初始化为负无穷，用于最大化

        threshold_A_steps = A_range  # “灰度阈值”从 0 到 255，步长为 step_A
        threshold_B_steps = np.arange(0, 1.01, step_B)  # “比例阈值”从 0.0 到 1.0，步长为 step_B

        logging.info("开始进行超参数调优...")
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

                # 记录调优结果，包括品位阈值
                self.tuning_results.append({
                    'threshold_A': I_th,
                    'threshold_B': ratio_th,
                    '抛废率': tuning_metrics['抛废率'],
                    '回收率': tuning_metrics['回收率'],
                    '品位阈值': grade_threshold,
                    '铅富集比': tuning_metrics['铅富集比'],
                    '锌富集比': tuning_metrics['锌富集比']
                })

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
                        best_metrics = tuning_metrics
                        best_params = (I_th, ratio_th)
                        self.best_under_constraints = (I_th, ratio_th, tuning_metrics)

        # 将调优结果列表转换为 DataFrame
        self.tuning_results = pd.DataFrame(self.tuning_results)

        # 计算 Pareto 前沿
        self.pareto_front = self._compute_pareto_front(self.tuning_results)
        self.best_params = best_params
        self.best_metrics = best_metrics

        return best_params, best_metrics

    def _compute_pareto_front(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def compute_baseline_curve(self) -> Tuple[list, list, list]:
        """
        基于按平均像素值降序(即品位升序)排序计算基线曲线。对于灰度图，平均像素值高意味着品位低；
        对于R值图, 平均像素值高意味着品位高。

        返回:
        - Tuple 包含:
            - scrap_rates_baseline (list): 基线抛废率。
            - recovery_rates_baseline (list): 基线回收率。
            - grade_thresholds_baseline (list): 与每个基线点对应的品位阈值。
        """
        # 计算每个矿石的平均像素值
        mean_pixel_values = self.pixels.apply(np.mean)

        if self.pixel_kind == 'grayness':
            # 按平均像素值降序排序（平均像素值高意味着品位低）
            sorted_indices = mean_pixel_values.sort_values(ascending=False).index
        elif self.pixel_kind == 'R':
            # 按平均像素值升序排序（平均像素值高意味着品位高）
            sorted_indices = mean_pixel_values.sort_values(ascending=True).index
        else:
            raise ValueError("无效的 pixel_kind 值。请使用 'grayness' 或 'R'。")

        # 将索引标签转换为位置索引
        sorted_positions = self.weight.index.get_indexer(sorted_indices)
        if (sorted_positions == -1).any():
            raise ValueError("一个或多个排序后的索引在 self.weight 中未找到。")

        scrap_rates_baseline, recovery_rates_baseline, grade_thresholds_baseline = self._compute_rates(sorted_positions)

        return scrap_rates_baseline, recovery_rates_baseline, grade_thresholds_baseline

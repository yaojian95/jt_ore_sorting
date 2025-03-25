'''
utility functions.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

def append_generic(arr1, arr2):
    '''
    list1 = [1, 2] 
    list2 = [3, 4]
    print(append_generic(list1, list2))


    list1 = [[1, 2], [3,4]]
    list2 = [[5,6], [7,8]]
    print(append_generic(list1, list2))
    '''
    if isinstance(arr1, np.ndarray):
        return np.concatenate((arr1, arr2))
    elif isinstance(arr1, list):
        # 检查列表元素是否还是列表
        if all(isinstance(i, list) for i in arr1) and all(isinstance(i, list) for i in arr2):
            result = [sub_arr1 + sub_arr2 for sub_arr1, sub_arr2 in zip(arr1, arr2)]
            return result
        else:
            return arr1 + arr2
    else:
        raise TypeError("Unsupported type")
    
def find_closest_point(scrap_rates, recovery_rates, target= [20, 95]):
    '''
    找到回收率最接近目标回收率的抛废率， 或者最接近目标抛废率的回收率。

    :param scrap_rates: 抛废率列表
    :param recovery_rates: 回收率列表
    :param target: 目标回收率

    :return: 最接近目标回收率的抛废率, and indexes

    '''
    scrap_rates = np.array(scrap_rates)
    recovery_rates = np.array(recovery_rates)
    
    min_index_scrap = np.argmin(np.abs(scrap_rates - target[0]))
    min_index_recovery = np.argmin(np.abs(recovery_rates - target[1]))

    closest_scrap = [min_index_scrap, scrap_rates[min_index_scrap], recovery_rates[min_index_scrap]]
    closest_recovery = [min_index_recovery, scrap_rates[min_index_recovery], recovery_rates[min_index_recovery]]
    return [closest_scrap, closest_recovery]
    
def plot_curves(
    rates, 
    scrap_rates_ideal,
    grade_thresholds_ideal: list,
    best_sum_point = None,
    best_enrichment_point = None,
    best_constraint_point = None,
) -> None:
    """
    绘制理想曲线、基线曲线、Pareto 前沿，并突出显示最佳点。

    参数:
    - rates (list): 不同算法的抛废率和回收率列表。
    - grade_thresholds_ideal (list): 理想曲线的品位阈值。
    - pareto_front (pd.DataFrame): Pareto 前沿 DataFrame。
    - best_sum_point (pd.Series, optional): 最佳总和点。
    - best_enrichment_point (pd.Series, optional): 最佳富集点。
    - best_constraint_point (pd.Series, optional): 最佳约束点。
    """

    _, ax1 = plt.subplots(figsize=(10, 7))
    for i in range(len(rates)):
        ax1.plot(rates[i][0], rates[i][1], label=rates[i][2], color = rates[i][3], linewidth=2, marker='o')

    ax1.set_xlim(0, 1.05)
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(np.arange(0, 1.1, 0.05))
    ax1.set_xticklabels([f'{int(x * 100)}%' for x in np.arange(0, 1.1, 0.05)])
    ax1.set_yticks(np.arange(0, 1.1, 0.05))
    ax1.set_yticklabels([f'{int(y * 100)}%' for y in np.arange(0, 1.1, 0.05)])

    # 设置坐标轴标签和标题
    ax1.set_xlabel('抛废率 (%)', fontsize=12)
    ax1.set_ylabel('回收率 (%)', fontsize=12)
    ax1.set_title('化验、均值与双阈值算法“回收率-抛废率”曲线', fontsize=14)

    # 创建一个双轴来显示品位阈值
    ax2 = ax1.twiny()

    # 根据品位阈值设置双轴的刻度和标签
    tick_scrap_rates = np.arange(0, 1.1, 0.2)  # 0, 0.2, 0.4, 0.6, 0.8, 1.0
    tick_grade_thresholds = []
    for tsr in tick_scrap_rates:
        idx = np.argmin(np.abs(np.array(scrap_rates_ideal) - tsr))
        tick_grade_thresholds.append(grade_thresholds_ideal[idx])

    # 将品位阈值转换为百分比
    tick_grade_thresholds_percent = [f'{(gt * 100):.2f}%' for gt in tick_grade_thresholds]

    # 设置双轴的刻度和标签
    ax2.set_xticks(tick_scrap_rates)
    ax2.set_xticklabels(tick_grade_thresholds_percent)
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xlabel('品位阈值 (%)', fontsize=12)

    # 如果有 Pareto 前沿点，则绘制
    if best_sum_point is not None:
        # 按抛废率排序 Pareto 前沿
        # pareto_sorted = pareto_front.sort_values('抛废率')

        # ax1.plot(pareto_sorted['抛废率'], pareto_sorted['回收率'],
        #             label='双阈值算法', color='red', alpha=0.6, marker='o')

        # 绘制最佳总和点
        if best_sum_point is not None:
            ax1.scatter(best_sum_point['抛废率'], best_sum_point['回收率'],
                        label='最佳总和点', color='green', edgecolors='black', s=100)

        # 绘制最佳富集点
        if best_enrichment_point is not None:
            ax1.scatter(best_enrichment_point['抛废率'], best_enrichment_point['回收率'],
                        label='最佳富集点', color='purple', edgecolors='black', s=100)

        # 绘制最佳约束点（如果存在）
        if best_constraint_point is not None:
            ax1.scatter(best_constraint_point['抛废率'], best_constraint_point['回收率'],
                        label='最佳约束点', color='orange', edgecolors='black', s=100)

    ax1.legend()
    ax1.grid(True)
    plt.tight_layout()
    plt.show()
'''
utility functions.
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def select_ores_greedy(df, N, target_grade, max_iter=1000):
    """
    使用贪心算法选择矿石
    
    参数:
    - df: pandas.DataFrame, 包含列: 'weight', 'pb', 'zn', 'fe'
    - N: int, 需要选择的矿石数量
    - target_grade: float, 目标(Pb+Zn)加权平均品位
    - max_iter: int, 最大迭代次数
    
    返回:
    - selected_df: pandas.DataFrame, 选中的矿石数据
    """

    df = df.copy()
    df['delta'] = abs((df['Pb_grade'] + df['Zn_grade']) - target_grade)
    
    # 初始选择：delta最小的N块矿石
    selected = df.nsmallest(N, 'delta').copy()
    rest = df.drop(selected.index).copy()
    
    def calc_avg(s_df):
        total_weight = s_df['weight'].sum()
        return ((s_df['Pb_grade'] * s_df['weight']).sum() + 
                (s_df['Zn_grade'] * s_df['weight']).sum()) / total_weight
    
    best_diff = abs(calc_avg(selected) - target_grade)
    
    for _ in range(max_iter):
        improved = False
        # 遍历selected的副本，避免修改循环中的集合
        for i in selected.index.copy():
            if improved:
                break
            # 遍历rest的副本
            for j in rest.index.copy():
                try:
                    # 尝试交换i和j
                    new_selected = pd.concat([
                        selected.drop(i),
                        rest.loc[[j]]  # 确保用[[j]]保留DataFrame结构
                    ])
                    new_avg = calc_avg(new_selected)
                    new_diff = abs(new_avg - target_grade)
                    
                    if new_diff < best_diff:
                        # 更新rest：移除j，添加i
                        new_rest = pd.concat([
                            rest.drop(j),
                            selected.loc[[i]]
                        ])
                        # 确认索引同步
                        assert i in new_rest.index, f"矿石{i}未正确添加到rest！"
                        assert j not in new_rest.index, f"矿石{j}未从rest中移除！"
                        
                        # 更新selected和rest
                        selected = new_selected
                        rest = new_rest
                        best_diff = new_diff
                        improved = True
                        break
                except KeyError as e:
                    print(f"交换时出错：{e}，跳过此次交换")
                    continue
        if not improved:
            break
    
    # 验证结果
    # assert selected.index.is_unique, "选中的矿石索引重复！"
    # assert rest.index.is_unique, "剩余的矿石索引重复！"
    # assert len(selected) == N, f"选中的矿石数量不正确：{len(selected)} != {N}"
    
    print(f"目标品位: {target_grade:.4f}, 实际得到: {calc_avg(selected):.4f}")
    return selected.drop(columns=['delta'])

def merge_datasets(datasets):
    """
    合并多个数据集
    Parameters:
        datasets: 要合并的数据集列表，每个元素格式为 [pixels, truth]
                  pixels是包含低能和高能数据的列表
    Returns:
        合并后的数据集，格式为 [pixels, truth]
    """
    merged = [[], None]
    offset = 0
    
    # 合并pixels
    for i, dataset in enumerate(datasets):
        print(i)
        if not merged[0]:
            merged[0] = [d.copy(deep=True) for d in dataset[0]]
        else:
            for j in range(2):
                # 在复制的DataFrame上增加索引偏移量
                copied_data = dataset[0][j].copy(deep=True)
                copied_data.index += offset
                merged[0][j] = pd.concat([merged[0][j], copied_data], axis=0)

        if merged[1] is None:
            merged[1] = dataset[1].copy(deep=True)
            merged[1]['source'] = f'source_{i}'

        else:
            # print(i)
            copied_truth = dataset[1].copy(deep=True)
            copied_truth['source'] = f'source_{i}'
            copied_truth.index += offset
            merged[1] = pd.concat([merged[1], copied_truth], axis=0)

        offset += (dataset[1].index[-1] + 1)
    
    # 计算Zn_Pb_grade
    merged[1]['Zn_Pb_grade'] = merged[1]['Zn_grade'] + merged[1]['Pb_grade']
    
    return merged

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
    grade_thresholds_ideal,
    best_sum_point = None,
    best_enrichment_point = None,
    best_constraint_point = None, 
    title = '回收率-抛废率曲线'):
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
        ax1.plot(rates[i][0][0], rates[i][0][1], label=rates[i][1], color = rates[i][2], 
                 linewidth=1, linestyle = rates[i][3], marker = rates[i][4], markersize = 3)

    ax1.set_xlim(0, 1.05)
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(np.arange(0, 1.1, 0.05))
    ax1.set_xticklabels([f'{int(x * 100)}%' for x in np.arange(0, 1.1, 0.05)])
    ax1.set_yticks(np.arange(0, 1.1, 0.05))
    ax1.set_yticklabels([f'{int(y * 100)}%' for y in np.arange(0, 1.1, 0.05)])

    # 设置坐标轴标签和标题
    ax1.set_xlabel('抛废率 (%)', fontsize=12)
    ax1.set_ylabel('回收率 (%)', fontsize=12)
    ax1.set_title(title, fontsize=14)

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

    ax1.legend(frameon = False)
    ax1.grid(True)
    plt.tight_layout()
    plt.show()
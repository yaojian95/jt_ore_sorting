import numpy as np
import pandas as pd
import os
import sys

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] 

import pickle
from multiprocessing import Pool
from multiprocessing import get_context

current_dir = os.getcwd()
code_path = os.path.abspath(os.path.join(current_dir, '..', '..', '..', 'codes'))
data_path = os.path.abspath(os.path.join(current_dir, '..', '..', '..', 'data'))

sys.path.append(os.path.join(code_path, 'jt_ore_sorting'))

from dataloader import path2truth, path2pixel, load_data

from classifiers.dual_thresh import DualThreshClassifier
from classifiers.r_method import RMethodClassifier
from classifiers.dual_all_parallel import ParallelClassifier
from classifiers.demo import Demo


# print('Success')
# exit()

def _tune_single_algorithm(input, algorithm_class, params, tuning_params):
    """单个算法的调优函数"""
    classifier = algorithm_class(**params)
    classifier.tuning(min_recovery_rate=0.95, min_scrap_rate=0.2, **tuning_params)
    return classifier

def tune(input_list, input_name = ['0219', '0224', 'both'], step_A = 5, include_Fe=False):
    m_name = ['dual_low', 'dual_high', 'R']
    res = {}
    
    for i, input in enumerate(input_list):
        # 定义每个算法的参数
        algorithms_params = [
            {
                'pixels': input[0][0],
                'truth': input[1],
                'pixel_kind': 'grayness'
            },
            {
                'pixels': input[0][1],
                'truth': input[1],
                'pixel_kind': 'grayness',
                'include_Fe': include_Fe
            },
            {
                'pixels': input[0],
                'truth': input[1],
                'I0_low': 195,
                'I0_high': 196,
                'input': 'pixels',
                'method': 'a',
                'const': [5, 20],
                'include_Fe': include_Fe
            }
        ]

        # 定义每个算法的调优参数
        tuning_params = [
            {
                'A_range': np.arange(0, 256, step_A),
                'step_B': 0.05,
                'grade_real_th': 0.1
            },
            {
                'A_range': np.arange(0, 256, step_A),
                'step_B': 0.05,
                'grade_real_th': 0.1
            },
            {
                'A_range': np.arange(0.5, 1.2, 0.01),
                'step_B': 0.05,
                'grade_real_th': 0.1
            }
        ]

        # 创建进程池并行执行
        with get_context('spawn').Pool(processes=4) as pool:
            results = pool.starmap(_tune_single_algorithm, [
                (input, DualThreshClassifier, algorithms_params[0], tuning_params[0]),
                (input, DualThreshClassifier, algorithms_params[1], tuning_params[1]),
                (input, RMethodClassifier, algorithms_params[2], tuning_params[2])
            ])

        # 处理结果

        res_save = []
        for j, result in enumerate(results):

            res, log = Demo(result).report_rates()
            print(log)
            res_save.append(result.tuning_results)

            # if len(method.best_params) == 3:
            #     params = {'grayness_th': method.best_params[0], 'ratio_th': method.best_params[1], 'mean_th': method.best_params[2]}
            # else:
            #     params = {'grayness_th': method.best_params[0], 'ratio_th': method.best_params[1]}

            # params.update(method.best_metrics)
            # res[input_name[i] + '_' + m_name[j]] = params
        

    return res_save     

with open('input_0219_0224_0225_0226_0227_contour_th_128', 'rb') as f:
    input_all = pickle.load(f)
pixels = input_all[0]
data = input_all[1]

dates = []
for t in range(5):
    index_0219 = data.loc[data['source'] == 'source_%s'%t].index.values
    input_0219 = [[pixels[0][index_0219], pixels[1][index_0219]], data.loc[index_0219]]
    dates.append(input_0219)   

res_all = tune(dates, input_name = ['0219', '0224', '0225', '0226', '0227'], step_A = 1)

with open('20250507_results_vs_thresholds.pkl', 'wb') as f:
    pickle.dump(res_all, f) 

# with open ('results.pkl', 'rb') as f:
#     res_all = pickle.load(f)
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


path_0219 = os.path.join(data_path, '20250219/160kV_4mA_forward.png')
excel_0219 = os.path.join(data_path, '化验结果/20250219化验结果.xlsx')

excel_0224 = os.path.join(data_path, '化验结果/20250224化验结果.xlsx')
path_0224 = os.path.join(data_path, '20250224/160kV_4mA_forward.png')

excel_0225 = os.path.join(data_path, '化验结果/20250225化验结果.xlsx')
path_0225_p1 = os.path.join(data_path, '20250225/160kV_4mA_1_105_forward.png')
path_0225_p2 = os.path.join(data_path, '20250225/160kV_4mA_106_200_forward.png')

z = path2pixel(path_0219, [[105, 1000, 400, 1200], [1950, 2844, 350, 1150]], max_len = [6, 9], length = [102, 35], 
           s_i = [64, 1], th_val = 105, save_rock_image = False, direction = 'ublr', save_rock_pixels=False)

consts = [5, 20]; th_val = 110

y = path2pixel(path_0224, [[105, 1000, 400, 1200], [1850,-100, 350,1150]], max_len = [6, 9], length=[102, 35], 
               s_i = [64, 1], th_val = 110, save_rock_image = False, direction = 'ublr', save_rock_pixels=False)

x1 = path2pixel(path_0225_p1, [[50, 1000, 350, 1150], [1850, -60, 350,1150]], max_len = [6, 9], length=[102, 35], 
               s_i = [64, 1], th_val = 115, save_rock_image = False, direction = 'ublr', save_rock_pixels=False)

x1[2][0].insert(55, np.array((0, 1)))
x1[2][1].insert(55, np.array((0, 1)))

x2 = path2pixel(path_0225_p2, [[30, 1000, 350, 1150], [1850, -60, 350,1150]], max_len = [7, 7], length=[35, 35], 
               s_i = [50, 1], th_val = 115, save_rock_image = False, direction = 'ublr', save_rock_pixels=False)

x2[2][0].insert(19, np.array((0, 1)))
x2[2][1].insert(19, np.array((0, 1)))

x2[2][0].insert(61, np.array((0, 1)))
x2[2][1].insert(61, np.array((0, 1)))

x  = [x1[2][0] + x2[2][0], x1[2][1] + x2[2][1]]

input_0224 = load_data(y[2], excel_0224)
input_0219 = load_data(z[2], excel_0219)
input_0225 = load_data(x, excel_0225)

input_both = [[],]
# pixels
input_both[0].append(input_0224[0][0].copy(deep = True))
input_both[0][0].index += 99

input_both[0].append(input_0224[0][1].copy(deep = True))
input_both[0][1].index += 99
# truth
input_0219[1]['source'] = '0219'
input_0224[1]['source'] = '0224'
input_both.append(input_0224[1].copy(deep = True))
input_both[1].index += 99

input_both[0][0] = pd.concat([input_0219[0][0], input_both[0][0]], axis = 0)
input_both[0][1] = pd.concat([input_0219[0][1], input_both[0][1]], axis = 0)

input_both[1]= pd.concat([input_0219[1], input_both[1]], axis = 0)

input_both[1]['Zn_Pb_grade'] = input_both[1]['Zn_grade'] + input_both[1]['Pb_grade']

# print('Success')
# exit()

def _tune_single_algorithm(input, algorithm_class, params, tuning_params):
    """单个算法的调优函数"""
    classifier = algorithm_class(**params)
    classifier.tuning(min_recovery_rate=0.95, min_scrap_rate=0.2, **tuning_params)
    return classifier

def tune(input_list, input_name = ['0219', '0224', 'both'], step_A = 5, include_Fe=False):
    m_name = ['dmp', 'dual', 'Rmp', 'R']
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
                'pixels': input[0][0],
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
                'mean_range': np.arange(20, 100),
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
                'mean_range': np.arange(0.5, 1.2, 0.01),
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
                (input, ParallelClassifier, algorithms_params[0], tuning_params[0]),
                (input, DualThreshClassifier, algorithms_params[1], tuning_params[1]),
                (input, RMethodClassifier, algorithms_params[2], tuning_params[2]),
                (input, RMethodClassifier, algorithms_params[3], tuning_params[3])
            ])

        # 处理结果
        for j, method in enumerate(results):
            if len(method.best_params) == 3:
                params = {'grayness_th': method.best_params[0], 'ratio_th': method.best_params[1], 'mean_th': method.best_params[2]}
            else:
                params = {'grayness_th': method.best_params[0], 'ratio_th': method.best_params[1]}

            params.update(method.best_metrics)
            res[input_name[i] + '_' + m_name[j]] = params

    return res        

res_all = tune([input_0219, input_0224, input_both, input_0225], input_name = ['0219', '0224', 'both','0225'], step_A = 1)

with open('20250414_results_inlucde_0225_v2.pkl', 'wb') as f:
    pickle.dump(res_all, f) 

# with open ('results.pkl', 'rb') as f:
#     res_all = pickle.load(f)
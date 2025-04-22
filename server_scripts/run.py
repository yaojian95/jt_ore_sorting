import numpy as np
import pandas as pd

import os
import sys

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] 
# %matplotlib inline

import pickle

current_dir = os.getcwd()
code_path = os.path.abspath(os.path.join(current_dir, '..', '..', '..', 'codes'))
data_path = os.path.abspath(os.path.join(current_dir, '..', '..', '..', 'data'))

sys.path.append(os.path.join(code_path, 'jt_ore_sorting'))

from utils import select_ores_greedy
from dataloader import path2truth, path2pixel, load_data

from classifiers.dual_thresh import DualThreshClassifier
from classifiers.r_method import RMethodClassifier
from classifiers.dual_all_parallel import ParallelClassifier
from classifiers.demo import Demo


with open('/home/yaojian/codes/jt_ore_sorting/input_0219_0224_0225.pkl', 'rb') as f:
    input_all = pickle.load(f)
pixels = input_all[0]
data = input_all[1]


index_0219 = data.loc[data['source'] == 'source_0'].index.values
input_0219 = [[pixels[0][index_0219], pixels[1][index_0219]], data.loc[index_0219]]

def tune(input_list, input_name = ['0219', '0224', 'both','0225'], step_A = 5, include_Fe = False):
    
    m_name = ['dmp', 'dual', 'Rmp', 'R']; 
    res = {}
    i = 0
    for input in input_list:
        dmp = ParallelClassifier(pixels=input[0][0], truth=input[1], pixel_kind= 'grayness')
        dmp.tuning(min_recovery_rate=0.95, min_scrap_rate=0.2, A_range=np.arange(0, 256, step_A), step_B=0.05, 
               mean_range= np.arange(20, 100), grade_real_th=0.1)
        
        dual = DualThreshClassifier(pixels=input[0][0], truth=input[1], pixel_kind= 'grayness', include_Fe = include_Fe)
        dual.tuning(min_recovery_rate=0.95, min_scrap_rate=0.2, A_range=np.arange(0, 256, step_A), step_B=0.05, 
                    grade_real_th= 0.1)
        
        #考虑三个阈值时，R_method.py内部使用的还是ParallelClassifier
        Rmp = RMethodClassifier(pixels=input[0], truth=input[1], I0_low=195, I0_high=196, input= 'pixels', method= 'a', const= [5, 20], include_Fe = include_Fe)
        Rmp.tuning(min_recovery_rate=0.95, min_scrap_rate=0.2, A_range=np.arange(0.5, 1.2, 0.01), step_B=0.05, 
                   mean_range= np.arange(0.5, 1.2, 0.01), grade_real_th= 0.1)
        
        R = RMethodClassifier(pixels=input[0], truth=input[1], I0_low=195, I0_high=196, input= 'pixels', method= 'a', const= [5, 20], include_Fe = include_Fe)
        R.tuning(min_recovery_rate=0.95, min_scrap_rate=0.2, A_range=np.arange(0.5, 1.2, 0.01), step_B=0.05, 
                 grade_real_th= 0.1)
        
        algorithms = [dmp, dual, Rmp, R]

        j = 0
        for method in algorithms:
            try:
                if len(method.best_params) == 3:
                    params = {'grayness_th': method.best_params[0], 'ratio_th': method.best_params[1], 'mean_th': method.best_params[2]}
                else:
                    params = {'grayness_th': method.best_params[0], 'ratio_th': method.best_params[1]}

                params.update(method.best_metrics)

            except:
                params = 'No best params'

            testtest = Demo(method)
            new_test = testtest.find_closest_point(target = [0.2, 0.95])
            params['close_scrap'] = [round(i, 4) for i in new_test[0][1:]]
            params['close_recov'] = [round(i, 4) for i in new_test[1][1:]]

            res[input_name[i] + '_' + m_name[j]] = params
            
            j += 1
        i += 1

    return res           

datasets = []
input_names = []
for i in range(7):

    grade_i = select_ores_greedy(data, 100, i+1)

    index = grade_i.index.values
    input_i= [[pixels[0][index], pixels[1][index]], data.loc[index]]

    datasets.append(input_i)
    input_names.append('MG_%s'%(i+1))

with open('20250422_7_mean_grade_data_v2.pkl', 'wb') as f:
    pickle.dump(datasets, f) 

res_all = tune(datasets, input_name = input_names, step_A = 1)

with open('20250422_results_7_mean_grade_data.pkl', 'wb') as f:
    pickle.dump(res_all, f) 

# with open ('results.pkl', 'rb') as f:
#     res_all = pickle.load(f)
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

from dataloader import path2truth, path2pixel, load_data

from classifiers.dual_thresh import DualThreshClassifier
from classifiers.r_method import RMethodClassifier
from classifiers.dual_all_parallel import ParallelClassifier
from classifiers.demo import Demo


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

res_all = tune([input_0219, input_0224, input_both, input_0225], input_name = ['0219', '0224', 'both','0225'], step_A = 1)

with open('20250414_results_inlucde_0225_v2.pkl', 'wb') as f:
    pickle.dump(res_all, f) 

# with open ('results.pkl', 'rb') as f:
#     res_all = pickle.load(f)
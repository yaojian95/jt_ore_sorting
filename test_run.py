import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] 
import pickle
from utils import merge_datasets
from multiprocessing import Pool, cpu_count

current_dir = os.getcwd()
code_path = os.path.abspath(os.path.join(current_dir, '..', '..', '..', 'codes'))
data_path = os.path.abspath(os.path.join(current_dir, '..', '..', '..', 'data'))

sys.path.append(os.path.join(code_path, 'jt_ore_sorting'))
from classifiers.r_method import RMethodClassifier

# Load data
with open('input_0219_0224_0225_0226_0227_contour_th_128.pkl', 'rb') as f:
    input_all = pickle.load(f)
pixels = input_all[0]
data = input_all[1]

input_all = []
for t in range(6):
    index_d = data.loc[data['source'] == 'source_%s'%t].index.values
    input_d = [[pixels[0][index_d], pixels[1][index_d]], data.loc[index_d]]
    input_all.append(input_d)
input_names = ['0219', '0224', '0225', '0226', '0227', '0604', 'all']

input_24_26_27 = merge_datasets([input_all[1], input_all[3], input_all[4]])

# Initialize classifier
R_0219 = RMethodClassifier(pixels=input_24_26_27[0], truth=input_24_26_27[1], I0_low=195, I0_high=196, 
                           input='pixels', method='a', const=[5, 20], include_Fe=False)

grade_ths = np.arange(0.1, 7, 0.1)

def process_grade_th(grade_th):
    """Function to process a single grade threshold"""
    R_0219.tuning(min_recovery_rate=0.95, min_scrap_rate=0.2, A_range=np.arange(0.5, 1.2, 0.005), step_B=0.02, 
                  grade_real_th=grade_th/100.0, score_on=True, score_weight=True, check_metrics=False)
    return R_0219.tuning_results

if __name__ == '__main__':
    # Determine number of CPUs to use
    num_processes = cpu_count()
    
    # Create a pool of workers
    with Pool(processes=num_processes) as pool:
        # Map the processing function to the grade thresholds
        all_tuning = pool.map(process_grade_th, grade_ths)
    
    # Save results
    with open('grades_0p1_7_ths_0p005_0p02.pkl', 'wb') as f:
        pickle.dump(all_tuning, f)
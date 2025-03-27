'''
Implementation of R_method.

created by:
    @author: Jian Yao 
    @date: 2025-03-20
'''
from re import A
import pandas as pd
import numpy as np
import sys
sys.path.append("..")

from preprocessing import get_contours
from dataloader import split_dual_xray_image

# classifiers/r_method.py
from .base_classifier import BaseClassifier
from .dual_thresh import DualThreshClassifier

class RMethodClassifier(BaseClassifier):
    def __init__(self):
        super().__init__()

    def __init__(self, pixels, truth = None, I0_low = 195, I0_high = 196, input = 'pixels', method = 'a', const = [5, 20]):
        '''
        初始化RMethodClassifier类。
        '''
        super().__init__(name = 'R_method', pixels = pixels, truth = truth)
        
        self.R_pixels = self.compute_R(self.pixels[0], self.pixels[1], I0_low, I0_high, input = input, method= method, const=const)
        # self.pixel_kind = pixel_kind

    def classify_ores(self):
        # 这里实现R_method分类的逻辑

        pass
    

    def tuning(self, 
               min_recovery_rate = 0.95,
               min_scrap_rate = 0.2,
               A_range = np.arange(0.5, 1.2, 0.01), 
               step_B= 0.05):
        
        test = DualThreshClassifier(truth = self.truth, pixels= self.R_pixels, pixel_kind= 'R')
        test.tuning(min_recovery_rate=min_recovery_rate, min_scrap_rate=min_scrap_rate, A_range=A_range, step_B=step_B)

        return test
    
    def compute_R(self, low, high, I0_low = 195, I0_high = 196, input = 'images', method = 'a', const = [5, 20]):

        '''
        input: 'images' or 'pixels', whole images of low and high energy or pixels of rocks
        
        '''

        # if isinstance(low, pd.Series):
        #     low = low.to_list()
        #     high = high.to_list()

        if input == 'images':
            if method == 'a':
                return np.log(I0_low/(low+1e-6) + const[0] )/np.log(I0_high/(high+1e-6) + const[1])

            elif method == 'b':
                return np.log((low + 1e-6))/(np.log(high+1e-6 + 200.0))
            
        elif input == 'pixels':
            R_values = []
            for i in low.index:

                if method == 'a':
                    R_i =np.log(I0_low/(low[i]+1e-6) + const[0] )/np.log(I0_high/(high[i]+1e-6) + const[1])

                elif method == 'b':
                    R_i = np.log((low[i] + 1e-6))/(np.log(high[i]+1e-6 + 200.0))   
                R_values.append(R_i)

            return pd.Series(R_values, index=low.index)
    
    def R_from_path(low, high, rock_pixels, I0_low, I0_high, input = 'images', method = 'a', const = [5, 20]):
        '''return (R_pixels) R_images, low, high, rock_pixels'''
            
        R_images = compute_R(low, high, I0_low, I0_high, input = 'images', method= method, const=const)
        
        if input == 'pixels':
            R_pixels = compute_R(rock_pixels[0], rock_pixels[1], I0_low, I0_high, input = input, method= method, const=const)
            return R_pixels, R_images    
        else:
            return R_images
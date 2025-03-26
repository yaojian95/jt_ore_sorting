'''
Implementation of R_method.

created by:
    @author: Jian Yao 
    @date: 2025-03-20
'''
import cv2
import numpy as np
import sys
sys.path.append("..")

from preprocessing import get_contours
from dataloader import split_dual_xray_image

# classifiers/r_method.py
from .base_classifier import BaseClassifier

class RMethodClassifier(BaseClassifier):
    def __init__(self):
        super().__init__()

    def __init__(self, pixels, truth = None):
        '''
        初始化RMethodClassifier类。
        '''
        super().__init__(name = 'R_method', pixels = pixels, truth = truth)
        # self.pixel_kind = pixel_kind

    def classify_ores(self, pixels):
        # 这里实现R_method分类的逻辑
        # 假设用某种阈值或算法对pixels进行处理，返回分类结果
        results = []
        for pixel in pixels:
            if pixel > 128:  # 举例阈值，实际需要根据算法调整
                results.append('精矿')
            else:
                results.append('废矿')
        return results

def R_from_path(low, high, rock_pixels, I0_low, I0_high, input = 'images', method = 'a', const = [5, 20]):
    '''return (R_pixels) R_images, low, high, rock_pixels'''
         
    R_images = compute_R(low, high, I0_low, I0_high, input = 'images', method= method, const=const)
    
    if input == 'pixels':
        R_pixels = compute_R(rock_pixels[0], rock_pixels[1], I0_low, I0_high, input = input, method= method, const=const)
        return R_pixels, R_images    
    else:
        return R_images

def compute_R(low, high, I0_low, I0_high, input = 'images', method = 'a', const = [5, 20]):

    '''
    input: 'images' or 'pixels', whole images of low and high energy or pixels of rocks
    
    '''

    if input == 'images':
        if method == 'a':
            return np.log(I0_low/(low+1e-6) + const[0] )/np.log(I0_high/(high+1e-6) + const[1])

        elif method == 'b':
            return np.log((low + 1e-6))/(np.log(high+1e-6 + 200.0))
         
    elif input == 'pixels':
        R_values = []
        for i in range(len(low)):

            if method == 'a':
                R_i =np.log(I0_low/(low[i]+1e-6) + const[0] )/np.log(I0_high/(high[i]+1e-6) + const[1])

            elif method == 'b':
                R_i = np.log((low[i] + 1e-6))/(np.log(high[i]+1e-6 + 200.0))   
            R_values.append(R_i)

        return R_values 
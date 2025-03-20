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
        super().__init__('R_method')

    def classify(self, pixels):
        # 这里实现R_method分类的逻辑
        # 假设用某种阈值或算法对pixels进行处理，返回分类结果
        results = []
        for pixel in pixels:
            if pixel > 128:  # 举例阈值，实际需要根据算法调整
                results.append('精矿')
            else:
                results.append('废矿')
        return results


def R_from_path(path, roi, I0_low, I0_high, input = 'images', method = 'a', const = [5, 20],
                max_len = 6, length = 100, s_i = 0, direction = 'ublr', th_val = 105,
                save_rock_image = False):
    
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
    # cv2.IMREAD_GRAYSCALE 或 0， 转换为0-255的灰度图
    # cv2.IMREAD_COLOR 或 1，转换为3通道的彩色图
    # cv2.IMREAD_UNCHANGED 或 -1， 保持原来的不变

    low, high = split_dual_xray_image(image.T) #为了与函数输入一致，先将图像transpose

    low, high = low.T, high.T # 先转置回来

    y1, y2, x1, x2 = roi

    low, high = cv2.flip(low[y1:y2, x1:x2], 0), cv2.flip(high[y1:y2, x1:x2], 0) 
    # 先选择感兴趣区域
    # X射线探测器成像与实际矿石摆放位置（俯视）差180°且左右相反
    # 等效为沿着垂直方向翻转

    low_contoured, rock_pixels, contours = get_contours(low, high, th_val = th_val, max_len = max_len, length=length, 
                                              direction = direction, path = path, s_i = s_i, save_rock_image=save_rock_image)
         
    R_images = compute_R(low, high, I0_low, I0_high, input = 'images', method= method, const=const)
    
    if input == 'pixels':
        R_pixels = compute_R(rock_pixels[0], rock_pixels[1], I0_low, I0_high, input = input, method= method, const=const)
        return R_pixels, R_images, low, high, low_contoured, rock_pixels, contours    
    else:
        return R_images, low, high, low_contoured, rock_pixels, contours

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
'''
Used to load raw png data and laboratory results (if any).
- set null data to zeros. 
- return: 
    low and high energy images; 
    true (laboratory) results;

- Add function to show the distribution of the grade
created on 2025-03-20

'''

import numpy as np
import pandas as pd
import pickle
import logging
from typing import Tuple
from preprocessing import draw_contours_yao, get_contours
import cv2
import matplotlib.pyplot as plt

def split_dual_xray_image(image, offset_up=0, offset_down=0):
    # Split the image into low and high energy parts
    height = image.shape[0]
    low_power_image = image[offset_up:int(height / 2) - offset_down, :]
    high_power_image = image[int(height / 2) + offset_up:height - offset_down, :]

    return low_power_image, high_power_image

def prepare_R(path, roi, max_len = 6, length = 100, s_i = 0, direction = 'ublr', th_val = 105,
                save_rock_image = False):
    '''
    return low, high, rock_pixels, low_contoured, contours
    '''
        
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

    return low, high, rock_pixels, low_contoured, contours

# 定义 load_data 函数
def load_data(
        pixel_file,
        date,
        annotation_file = None):
    """
    加载并预处理像素数据和注释数据。

    参数:
    - pixel_file (str): 包含像素数据的 pickle 文件路径。
    - annotation_file (str): 包含注释数据的 Excel 文件路径。

    返回:
    - Tuple 包含:
        - ann (pd.DataFrame): 预处理后的注释 DataFrame。
        - pixels (pd.Series): 预处理后的像素 Series。
    """

    if pixel_file is not str:
        pixels = pixel_file
    else:
        # 加载像素数据
        with open(pixel_file, 'rb') as f:
            pixels_data = pickle.load(f)
        pixels = pd.Series(pixels_data[date]['low'][:len(ann.iloc[:,0])])

    try:
        # 加载注释数据
        ann = pd.read_excel(annotation_file)
        ann = ann.loc[:, ~ann.columns.str.contains('^Unnamed')]
        ann.rename(columns={
            'Fe': "Fe_grade",
            'Zn': "Zn_grade",
            'Pb': "Pb_grade",
            'S': "S_grade",
            'Weight(g)': "weight"
        }, inplace=True)

        # 处理缺失数据
        none_indexes = pixels[pixels.isnull()].index.tolist()
        missing_indexes = ann[ann.isnull().any(axis=1)].index.tolist()
        indexes_to_remove = set(none_indexes).union(missing_indexes)
        pixels = pixels.drop(indexes_to_remove)
        ann = ann.drop(indexes_to_remove)

        logging.info(f"加载数据完成。清洗后的样本数量: {len(ann)}")
        return ann, pixels
    except Exception as e:
        logging.error(f"加载数据时出错: {e}")
        raise
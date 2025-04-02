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
from preprocessing import get_contours
import cv2
from utils import append_generic


def list_depth(lst):
    """
    此函数用于计算列表的深度

    参数:
    lst (list): 要计算深度的列表

    返回:
    int: 列表的深度
    """
    if not isinstance(lst, list):
        return 0
    if not lst:
        return 1
    # 递归计算每个子列表的深度
    depths = [list_depth(item) for item in lst]
    # 取最大深度并加 1 得到当前列表的深度
    return max(depths) + 1

def split_dual_xray_image(image, offset_up=0, offset_down=0):
    # Split the image into low and high energy parts
    height = image.shape[0]
    low_power_image = image[offset_up:int(height / 2) - offset_down, :]
    high_power_image = image[int(height / 2) + offset_up:height - offset_down, :]

    return low_power_image, high_power_image

def path2pixel(path, roi, max_len = 6, length = 100, s_i = 0, direction = 'ublr', th_val = 105,
                save_rock_image = False, save_rock_pixels = False):
    '''

    Returns:
    --------
    low, high, rock_pixels, low_contoured, contours
    '''
    
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
    # cv2.IMREAD_GRAYSCALE 或 0， 转换为0-255的灰度图
    # cv2.IMREAD_COLOR 或 1，转换为3通道的彩色图
    # cv2.IMREAD_UNCHANGED 或 -1， 保持原来的不变

    low, high = split_dual_xray_image(image.T) #为了与函数输入一致，先将图像transpose

    low, high = low.T, high.T # 先转置回来

    if list_depth(roi) == 1:
        y1, y2, x1, x2 = roi
        low_roi, high_roi = cv2.flip(low[y1:y2, x1:x2], 0), cv2.flip(high[y1:y2, x1:x2], 0) 
        # 先选择感兴趣区域
        # X射线探测器成像与实际矿石摆放位置（俯视）差180°且左右相反
        # 等效为沿着垂直方向翻转
        low_contoured, rock_pixels, contours = get_contours(low_roi, high_roi, th_val = th_val, max_len = max_len, length=length, 
                                              direction = direction, path = path, s_i = s_i, save_rock_image=save_rock_image)

        pre_combined = low_roi, high_roi, rock_pixels, low_contoured, contours
    
    elif list_depth(roi) == 2:

        # two set of stones in one image
        pre = []
        assert len(max_len) == len(s_i) == len(roi) == len(length), "roi, max_len, s_i, length should have the same length"

        for p in range(len(roi)):
            y1, y2, x1, x2 = roi[p]
            low_roi, high_roi = cv2.flip(low[y1:y2, x1:x2], 0), cv2.flip(high[y1:y2, x1:x2], 0)
            # print(low)
            low_contoured, rock_pixels, contours = get_contours(low_roi, high_roi, th_val = th_val, max_len = max_len[p], length=length[p], 
                                              direction = direction, path = path, s_i = s_i[p], save_rock_image=save_rock_image)

            pre.append([low_roi, high_roi, rock_pixels, low_contoured, contours])
        
        pre_combined =[append_generic(pre[1][i], pre[0][i]) for i in range(len(pre[0]))]

    if save_rock_pixels:
        date = path.split('/')[-2]

        if list_depth(roi) == 1:
            date = date + '_part'

        # pixel_data = {
        #     date: {
        #         'low': pre_combined[2][0],
        #         'high': pre_combined[2][1]
        #             },
        #             }
        
        with open('%s_pixels.pkl'%date, 'wb') as f:
            pickle.dump(pre_combined[2], f)

    logging.info(f"{list_depth(roi)}个盒子的图像转换rock_pixels完成。")

    return pre_combined

def path2truth(path):
    '''
    Returns:
    -------
    - true_results: pd.DataFrame
    
    '''
    # 加载注释数据
    ann = pd.read_excel(path)
    ann = ann.loc[:, ~ann.columns.str.contains('^Unnamed')]
    ann.rename(columns={
        'Fe': "Fe_grade",
        'Zn': "Zn_grade",
        'Pb': "Pb_grade",
        'S': "S_grade",
        'Weight(g)': "weight"
    }, inplace=True)

    # 处理缺失数据
    # none_indexes = pixels[pixels.isnull()].index.tolist()
    missing_indexes = ann[ann.isnull().any(axis=1)].index.tolist()
    indexes_to_remove = set(missing_indexes)
    # indexes_to_remove = set(none_indexes).union(missing_indexes)
    # pixels = pixels.drop(indexes_to_remove)
    ann = ann.drop(indexes_to_remove)

    ann.index = ann.iloc[:, 0].values - 1
    logging.info(f"加载数据完成。清洗后的样本数量: {len(ann)}")

    return ann, ann.iloc[:, 0]

def load_data(pixel_file, truth_path = None):
    '''
    Parameters:
    ----------
    - pixel_file: path of pixel data, str or list of low and high energy pixels, pd.Series
    - truth_path: path of truth data, str

    Returns:
    -------
    - pixels_full: list of low and high energy pixels, pd.Series
    - true_results: pd.DataFrame
    '''
    if pixel_file is not str:
        pixels_data = pixel_file

    else:
        with open(pixel_file, 'rb') as f:
            pixels_data = pickle.load(f)

    if truth_path is None:
        true_results = None
        pixels_full = [pd.Series(rock) for rock in pixels_data]
    else:
        true_results, rock_ids = path2truth(truth_path)
        pixels_full = [pd.Series(rock).iloc[rock_ids - 1] for rock in pixels_data]

    return pixels_full, true_results
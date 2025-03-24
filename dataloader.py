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

def split_dual_xray_image(image, offset_up=0, offset_down=0):
    # Split the image into low and high energy parts
    height = image.shape[0]
    low_power_image = image[offset_up:int(height / 2) - offset_down, :]
    high_power_image = image[int(height / 2) + offset_up:height - offset_down, :]

    return low_power_image, high_power_image

# 定义 load_data 函数
def load_data(
        pixel_file,
        date,
        annotation_file == None):
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
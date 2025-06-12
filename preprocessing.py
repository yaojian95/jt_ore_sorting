'''
Used to prepare the input for the model(s).
- find contour for each ore

'''

import numpy as np
import cv2
from pathlib import Path

def contour_detection(image,
                      th_val=185,
                      min_contour_area=50,
                      max_contour_area=None,
                      min_rect_width=None,
                      max_rect_width=None,
                      min_rect_height=None,
                      max_rect_height=None,
                      min_rect_ratio=None,
                      max_rect_ratio=None,
                      roi=None,
                      rescale_factor=1.0):
    """
    Detects contours in an image using binary thresholding and filters them based on the specified criteria.
    Supports detecting contours within a specified Region of Interest (ROI) while keeping all parameters
    relative to the original image.

    Parameters:
    image (numpy.ndarray): The input image in which contours are to be detected.
    th_val (int, optional): Threshold value for binary thresholding. Default is 185.
    min_contour_area (int, optional): Minimum area of a contour to be considered. Default is 50.
    max_contour_area (int, optional): Maximum area of a contour to be considered. Default is 50% of image area.
    roi (tuple, optional): Region of Interest specified as (x1, x2, y1, y2). If any of x1, x2, y1, y2 is None,
                           it will default to covering the entire width or height.
    rescale_factor (float, optional): Factor by which the image is rescaled. Default is 1.0.

    Returns:
    list: A list of contours that meet the specified criteria.
    """
    # Rescale default parameters if the image is rescaled
    if rescale_factor != 1.0:
        min_contour_area = int(min_contour_area * (rescale_factor ** 2))
        if max_contour_area is not None:
            max_contour_area = int(max_contour_area * (rescale_factor ** 2))
        if min_rect_width is not None:
            min_rect_width = int(min_rect_width * rescale_factor)
        if max_rect_width is not None:
            max_rect_width = int(max_rect_width * rescale_factor)
        if min_rect_height is not None:
            min_rect_height = int(min_rect_height * rescale_factor)
        if max_rect_height is not None:
            max_rect_height = int(max_rect_height * rescale_factor)

    # Set default max contour area to 50% of image area if not provided
    if max_contour_area is None:
        max_contour_area = int(0.5 * image.size)

    # Get image dimensions
    img_height, img_width = image.shape[:2]

    # Handle ROI (defaulting x1, x2 to full width, y1, y2 to full height if not specified)
    if roi is not None:
        x1 = roi[0] if roi[0] is not None else 0  # Default x1 to 0 if None
        x2 = roi[1] if roi[1] is not None else img_width  # Default x2 to image width if None
        y1 = roi[2] if roi[2] is not None else 0  # Default y1 to 0 if None
        y2 = roi[3] if roi[3] is not None else img_height  # Default y2 to image height if None
    else:
        x1, x2, y1, y2 = 0, img_width, 0, img_height  # No ROI means full image is considered

    # Create a mask for the ROI
    mask = np.ascontiguousarray(np.zeros_like(image, dtype=np.uint8))
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    masked_image = cv2.bitwise_and(image, mask)

    # Thresholding
    _, thresholded = cv2.threshold(masked_image, th_val, 255, cv2.THRESH_BINARY)

    # Find contours using cv2.RETR_TREE and cv2.CHAIN_APPROX_SIMPLE
    contours, hierarchy = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Remove contours that are too small or too large
    filtered_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_contour_area <= area <= max_contour_area:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = w / h if h != 0 else 0  # Avoid division by zero
            # Apply width, height, and ratio filters
            if ( (min_rect_width is None or w >= min_rect_width) and
                 (max_rect_width is None or w <= max_rect_width) and
                 (min_rect_height is None or h >= min_rect_height) and
                 (max_rect_height is None or h <= max_rect_height) and
                 (min_rect_ratio is None or ratio >= min_rect_ratio) and
                 (max_rect_ratio is None or ratio <= max_rect_ratio) ):
                filtered_contours.append(cnt)

    return filtered_contours

def get_contour_pixels(image, contour):
    # Create a mask for the contour
    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)  # Fill the contour

    # Get pixel values inside the contour
    contour_pixel_values = image[mask == 255]
    return contour_pixel_values

def get_contour_box_image(image, contour, margin=10):
    xywh = cv2.boundingRect(contour)
    box_image = image[max(xywh[1] - margin, 0):xywh[1] + xywh[3] + margin,
                max(xywh[0] - margin, 0):xywh[0] + xywh[2] + margin]
    return box_image

def get_contours(low, high, th_val = 100, max_len = 6, length = 100, direction = 'ublr', path = '', s_i = 0, save_rock_image = False):
        
        path = Path(path)

        _contours = contour_detection(low, roi=(None, None, None, None), th_val=th_val)
        
        low_contoured, contours = draw_contours_yao(low, _contours, thickness=2, show = 'index', max_len=max_len, length=length,
                                          indexes=s_i + np.arange(len(_contours)), direction=direction)
        # low_rock_images = []
        # high_rock_images = []
        rock_pixels = [[], []]
        rock_images = [[], []]
        for i in range(len(contours)):
                contour = contours[i]
                # low_rock_images.append(get_contour_box_image(low, contour, margin=10))
                # high_rock_images.append(get_contour_box_image(high, contour, margin=10))
                rock_pixels[0].append(get_contour_pixels(low, contour))
                rock_pixels[1].append(get_contour_pixels(high, contour))

                rock_images[0].append(get_contour_box_image(low, contour, margin=5))
                rock_images[1].append(get_contour_box_image(high, contour, margin=5))

                if save_rock_image:
                    cv2.imwrite(path.parent/('rocks/%s_%s_low.png'%(path.stem, s_i + i)), get_contour_box_image(low, contour, margin=10))
                    cv2.imwrite(path.parent/('rocks/%s_%s_high.png'%(path.stem, s_i + i)), get_contour_box_image(high, contour, margin=10))
        return low_contoured, rock_pixels, contours, rock_images

def draw_contours_yao(image, contours, color=(0, 0, 255), thickness=2, max_len = 6, 
                      length = 100, show= 'index', indexes=None, direction = 'ublr', grade = None):
    '''

    width: the width of each row. 当图像转置之后，将会行变列，所以width 应该是列宽 0304
    length: the width of each column
    direction: ublr or lrub， 从上到下再从左到右， 或先从左到右再从上到下
    '''
    # Create a copy of low_img for visualization
    contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR).copy()
    cv2.drawContours(contour_image, contours, -1, color, thickness)

    if show == 'index':

        if direction == 'ublr':
            # 从上到下再从左到右
            # contours = sorted(contours, key=lambda c: ((cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2]/2.0)//length, (cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3]) //width))
            contours = sort_contours(contours, tolerance=length, max_len=max_len, direaction='y')
        elif direction == 'lrub':
            # 先从左到右再从上到下
            # contours = sorted(contours, key=lambda c: ((cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3]/2.0) //width, cv2.boundingRect(c)[0] // length)) 
            contours = sort_contours(contours, tolerance=length, max_len=max_len, direaction='x')
        if indexes is None:
            indexes = range(len(contours))
            text = indexes
        else:
            text = [str(i) for i in indexes]
        
    elif show == 'grade':
        # in this case contours should already be sorted
        assert grade is not None, 'grade must be provided'
        text = np.around(grade, decimals=2)

    for contour, text_i in zip(contours, text):
        # Calculate the centroid of the contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        # Put the index number near the contour
        cv2.putText(contour_image, str(text_i), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    # print(cv2.boundingRect(contours[7]), cv2.boundingRect(contours[8]))
    return contour_image, contours

# 根据轮廓质心位置对阈值分割的轮廓进行排序
def sort_contours(contours, tolerance=35, max_len=9, direaction='y', reverse=False):
    """
    根据阈值分割的轮廓进行排序,实现的排序方式有两种,分别是沿x方向和y方向,以下注释以y方向为例
    参数说明：
    contours: 根据cv2阈值分割得到的轮廓;
    tolerance: 表示在x或y方向上的容忍值,在范围内表示同一行或同一列;
    输出结果：
    根据需求排序后的轮廓
    x方向: 从左上角开始,从左到右,从上到下排列轮廓序号结果；
    y方向: 从左上角开始,从上到下,从左到右排列轮廓序号结果;
    reverse: 根据图像采集成像左右翻转设置的参数
    """
    if direaction == 'x':
        sort_index = 1
        group_index = 0
    elif direaction == 'y':
        sort_index = 0
        group_index = 1
    else:
        raise TypeError("Must be sorted along the x or y direction.")
    # 根据轮廓计算质心位置
    centers = [get_contour_centroid(cnt) for cnt in contours]
    
    # 初步排序：根据x值或y值从小到大排列
    sorted_indices = sorted(range(len(centers)), key=lambda i: centers[i][sort_index])
    sorted_contours = [contours[i] for i in sorted_indices]
    sorted_centers = [centers[i] for i in sorted_indices]
    # print(sorted_contours)
    
    # 分组处理：首先按纵列分组，再在每个纵列内按横坐标分组和排序
    groups = []
    current_group = []
    previous = None
    for cnt, center in zip(sorted_contours, sorted_centers):
        # print(x,y)
        if previous is None:
            current_group.append((cnt, center[group_index]))
            previous = center[sort_index]
        else:
            # 检查是否在同一纵列
            if abs(center[sort_index] - previous) <= tolerance and len(current_group) < max_len:
                current_group.append((cnt, center[group_index]))
                previous = center[sort_index]
            else:
                # 对当前纵列内的轮廓按x排序，并添加到组列表中
                current_group_sorted = sorted(current_group, key=lambda item: item[1], reverse=reverse)
                groups.extend([item[0] for item in current_group_sorted])
                current_group = [(cnt, center[group_index])]
                previous = center[sort_index]
    
    # 处理最后一个纵列
    if current_group:
        current_group_sorted = sorted(current_group, key=lambda item: item[1], reverse=reverse)
        groups.extend([item[0] for item in current_group_sorted])
    
    return groups

def get_contour_centroid(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        # warnings.warn("Contour area is zero. Returning (0, 0) as centroid.")
        cX, cY = 0, 0

    return cX, cY
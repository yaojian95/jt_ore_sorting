'''
utility functions.
'''

import numpy as np

def append_generic(arr1, arr2):
    '''
    list1 = [1, 2] 
    list2 = [3, 4]
    print(append_generic(list1, list2))


    list1 = [[1, 2], [3,4]]
    list2 = [[5,6], [7,8]]
    print(append_generic(list1, list2))
    '''
    if isinstance(arr1, np.ndarray):
        return np.concatenate((arr1, arr2))
    elif isinstance(arr1, list):
        # 检查列表元素是否还是列表
        if all(isinstance(i, list) for i in arr1) and all(isinstance(i, list) for i in arr2):
            result = [sub_arr1 + sub_arr2 for sub_arr1, sub_arr2 in zip(arr1, arr2)]
            return result
        else:
            return arr1 + arr2
    else:
        raise TypeError("Unsupported type")
    
def find_closest_point(scrap_rates, recovery_rates, target= [20, 95]):
    '''
    找到回收率最接近目标回收率的抛废率， 或者最接近目标抛废率的回收率。

    :param scrap_rates: 抛废率列表
    :param recovery_rates: 回收率列表
    :param target: 目标回收率

    :return: 最接近目标回收率的抛废率, and indexes

    '''
    scrap_rates = np.array(scrap_rates)
    recovery_rates = np.array(recovery_rates)
    
    min_index_scrap = np.argmin(np.abs(scrap_rates - target[0]))
    min_index_recovery = np.argmin(np.abs(recovery_rates - target[1]))

    closest_scrap = [min_index_scrap, scrap_rates[min_index_scrap], recovery_rates[min_index_scrap]]
    closest_recovery = [min_index_recovery, scrap_rates[min_index_recovery], recovery_rates[min_index_recovery]]
    return [closest_scrap, closest_recovery]


    

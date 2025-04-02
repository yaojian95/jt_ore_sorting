# from classifiers.r_method import RMethodClassifier
# from classifiers.dual_thresh import DualThreshClassifier
# from the benefit of __init__.py file.
from classifiers import RMethodClassifier, DualThreshClassifier
from dataloader import path2pixel, load_data
import os
import numpy as np

class MineralClassifier:
    def __init__(self, pixels, ground_truth = None):
        '''
        Initialize the MineralClassifier class.

        Parameters:
        ----------
        Pixels: pixels of rocks. Can be str or list. If str, it should be the path of image file. If list, it should be the pixels of rocks.
        Ground_truth: excel file path. Can be None.
        '''

        self.input = load_data(pixels, ground_truth)


    def classify_minerals(self, method_name, mode = 'best', best_params = None):
        '''
        Classify minerals based on the specified method.

        Parameters:
        ----------
        method_name: str, the name of the classification method. Can be 'R_method' or 'Dual_thresh'.
        mode: str, the mode of the classification. Can be 'best' or 'tuning'. If 'best', the best parameters should be provided. If 'tuning', the ground_truth should be provided.
        best_params: tuple, the best parameters of the classification method. Should be provided when mode is 'best'.

        Returns:
        ----------
        results: array, the classification results.
        '''

        if method_name == 'R_method':
            classifier = RMethodClassifier(pixels=self.input[0], truth=self.input[1], I0_low=195, I0_high=196, input= 'pixels', method= 'a', const= [5, 20], include_Fe = False)

        elif method_name == 'Dual_thresh':
            classifier = DualThreshClassifier(pixels=self.input[0][0], truth=self.input[1], pixel_kind= 'grayness', include_Fe = False)

        else:
            raise ValueError("未知的分类方法")
        
        if mode == 'best':

            assert best_params is not None, "best_params must be provided when mode is 'best'"
            classifier.best_params = best_params
            results = classifier.classify_ores(best_params[0], best_params[1])
        
        elif mode == 'tuning':

            assert self.input[1] is not None, "ground_truth must be provided when mode is 'tuning'"

            para_range = np.arange(0, 256, 5) if method_name == 'Dual_thresh' else np.arange(0.5, 1.2, 0.01)
            classifier.tuning(min_recovery_rate=0.95, min_scrap_rate=0.2, A_range=para_range, step_B=0.05)
            best_params = classifier.best_params
            print(f"最佳参数: {best_params}")
            results = classifier.classify_ores(best_params[0], best_params[1])
        
        print(classifier.calculate_tuning_metrics(results))

        return results


        # # 计算准确率和召回率
        # accuracy, recall = classifier.calculate_rates(results, self.ground_truth)
        # print(f"准确率: {accuracy}, 召回率: {recall}")

if __name__ == "__main__":
    
    # path of image and excel files
    current_dir = os.getcwd()
    data_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'data'))
    image_path = os.path.join(data_path, '20250219/160kV_4mA_forward.png')
    excel_path = os.path.join(data_path, '化验结果/20250219化验结果.xlsx')

    # from image to pixels of rocks
    z = path2pixel(image_path, [[105, 1000, 400, 1200], [1950, 2844, 350, 1150]], max_len = [6, 9], length = [102, 35], 
            s_i = [64, 1], th_val = 105, save_rock_image = False, direction = 'ublr', save_rock_pixels=False)

    consts = [5, 20]; th_val = 110

    classifier = MineralClassifier(z[2], excel_path)
    # 选择分类方法
    # method = 'Dual_thresh' 
    method = 'R_method'  
    
    # result = classifier.classify_minerals(method, mode = 'tuning')
    result = classifier.classify_minerals(method, mode = 'best', best_params = [0.93, 0])
    print(f"使用{method}方法分类的结果: {result}")

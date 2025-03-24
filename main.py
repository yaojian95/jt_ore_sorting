# from classifiers.r_method import RMethodClassifier
# from classifiers.dual_thresh import DualThreshClassifier
# from the benefit of __init__.py file.
from classifiers import RMethodClassifier, DualThreshClassifier
from dataloader import load_data

class MineralClassifier:
    def __init__(self, path_pixels, path_ground_truth):

        self.pixels, self.ground_truth = load_data(path_pixels, path_ground_truth)


    def classify_minerals(self, method_name):
        if method_name == 'R_method':
            classifier = RMethodClassifier()
        elif method_name == 'Dual_thresh':
            classifier = DualThreshClassifier()
        else:
            raise ValueError("未知的分类方法")

        results = classifier.classify(self.pixels, mode='best', ground_truth=self.ground_truth)

        # 计算准确率和召回率
        accuracy, recall = classifier.calculate_rates(results, self.ground_truth)
        print(f"准确率: {accuracy}, 召回率: {recall}")

        return results

if __name__ == "__main__":
    # 假设我们有一些矿石的像素数据
    pixels = [120, 130, 160, 90, 180, 110, 140, 85, 200]
    yaoyao = MineralClassifier('path_pixels', 'path_ground_truth')
    # 选择分类方法
    method = 'Dual_thresh'  # 或者 'Dual_thresh'
    
    result = yaoyao.classify_minerals(method, mode = 'best')
    print(f"使用{method}方法分类的结果: {result}")

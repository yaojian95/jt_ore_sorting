# from classifiers.r_method import RMethodClassifier
# from classifiers.dual_thresh import DualThreshClassifier
# from the benefit of __init__.py file.
from classifiers import RMethodClassifier, DualThreshClassifier

def classify_minerals(method_name, pixels):
    if method_name == 'R_method':
        classifier = RMethodClassifier()
    elif method_name == 'Dual_thresh':
        classifier = DualThreshClassifier()
    else:
        raise ValueError("未知的分类方法")

    return classifier.classify(pixels)

if __name__ == "__main__":
    # 假设我们有一些矿石的像素数据
    pixels = [120, 130, 160, 90, 180, 110, 140, 85, 200]
    
    # 选择分类方法
    method = 'Dual_thresh'  # 或者 'Dual_thresh'
    
    result = classify_minerals(method, pixels)
    print(f"使用{method}方法分类的结果: {result}")

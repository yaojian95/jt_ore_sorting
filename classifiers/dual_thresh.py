# classifiers/dual_thresh.py
from .base_classifier import BaseClassifier

class DualThreshClassifier(BaseClassifier):
    def __init__(self):
        super().__init__('Dual_thresh')

    def classify(self, pixels):
        # 这里实现dual_thresh分类的逻辑
        results = []
        for pixel in pixels:
            if pixel > 100 and pixel < 150:
                results.append('精矿')
            else:
                results.append('废矿')
        return results

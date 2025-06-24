import numpy as np

def calculate_recovery_rates(concentrate_mass, concentrate_grade, 
                           tailings_mass, tailings_grade, 
                           feed_grade=None):
    """
    计算选矿回收率和抛废率
    
    参数:
    - concentrate_mass: 精矿质量 (吨)
    - concentrate_grade: 精矿品位 (%)
    - tailings_mass: 尾矿(废矿)质量 (吨)
    - tailings_grade: 尾矿品位 (%)
    - feed_grade: 可选，原矿品位 (%)。如不提供则自动计算
    
    返回:
    - 字典包含回收率、抛废率和富集比
    """
    # 计算金属量
    metal_concentrate = concentrate_mass * concentrate_grade / 100
    metal_tailings = tailings_mass * tailings_grade / 100
    total_metal = metal_concentrate + metal_tailings
    
    # 自动计算原矿平均品位(如未提供)
    if feed_grade is None:
        total_mass = concentrate_mass + tailings_mass
        feed_grade = total_metal / total_mass * 100 if total_mass > 0 else 0
    
    # 计算回收率和抛废率
    recovery_rate = metal_concentrate / total_metal * 100 if total_metal > 0 else 0
    discard_rate = tailings_mass / (concentrate_mass + tailings_mass) * 100
    enrichment_ratio = concentrate_grade / feed_grade if feed_grade > 0 else 0
    
    result = {
        '回收率(%)': round(recovery_rate, 2),
        '抛废率(%)': round(discard_rate, 2),
        '原矿品位(%)': round(feed_grade, 2),
        '富集比': round(enrichment_ratio, 2),
        '精矿金属量(kg)': round(metal_concentrate, 2),
        '尾矿金属量(kg)': round(metal_tailings, 2)
    }
    
    return result

if __name__ == "__main__":
    # 示例数据
    concentrate_mass = 38.1   
    concentrate_grade = 1.924+0.737 
    tailings_mass = 13.4     
    tailings_grade = 0.477 + 0.257  
    
    # 计算结果(不提供原矿品位，自动计算)
    result = calculate_recovery_rates(
        concentrate_mass, concentrate_grade,
        tailings_mass, tailings_grade
    )
    
    print("选矿指标计算结果:")
    for k, v in result.items():
        print(f"{k}: {v}")
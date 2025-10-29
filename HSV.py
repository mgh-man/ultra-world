import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class HSVAdjuster:
    """
    基于RandomHSV类的HSV颜色调整器
    
    Args:
        hgain (float): 色调调整幅度，范围 [0, 1]
        sgain (float): 饱和度调整幅度，范围 [0, 1] 
        vgain (float): 明度调整幅度，范围 [0, 1]
    """
    
    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5):
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain
    
    def adjust_hsv(self, img):
        """
        调整图像的HSV值
        
        Args:
            img (np.ndarray): 输入图像 (BGR格式)
            
        Returns:
            np.ndarray: 调整后的图像
        """
        if self.hgain != 0.5 or self.sgain != 0.5 or self.vgain != 0.5:
            # 获取图像数据类型
            dtype = img.dtype
            
            # 将[0,1]范围转换为[-1,1]范围进行计算
            r = np.array([(self.hgain - 0.5) * 2, (self.sgain - 0.5) * 2, (self.vgain - 0.5) * 2])
            
            # 创建查找表
            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x + r[0] * 180) % 180).astype(dtype)
            lut_sat = np.clip(x * (r[1] + 1), 0, 255).astype(dtype)
            lut_val = np.clip(x * (r[2] + 1), 0, 255).astype(dtype)
            lut_sat[0] = 0  # 防止纯白色改变颜色
            
            # 转换为HSV色彩空间
            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            
            # 应用查找表
            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), 
                               cv2.LUT(sat, lut_sat), 
                               cv2.LUT(val, lut_val)))
            
            # 转换回BGR色彩空间
            result = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)
            return result
        
        return img

def process_image(image_path, hgain=0.5, sgain=0.5, vgain=0.5, save_path=None):
    """
    处理单张图片的HSV调整
    
    Args:
        image_path (str): 输入图片路径
        hgain (float): 色调调整 [0, 1]
        sgain (float): 饱和度调整 [0, 1]
        vgain (float): 明度调整 [0, 1]
        save_path (str): 保存路径，为None时不保存
    """
    # 读取图像
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"无法读取图像: {image_path}")
        return None
    
    # 创建HSV调整器
    adjuster = HSVAdjuster(hgain=hgain, sgain=sgain, vgain=vgain)
    
    # 调整图像
    adjusted_img = adjuster.adjust_hsv(img)
    
    # 显示结果
    plt.figure(figsize=(15, 5))
    
    # 原图
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('原图')
    plt.axis('off')
    
    # 调整后的图
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(adjusted_img, cv2.COLOR_BGR2RGB))
    plt.title(f'调整后 (H:{hgain:.2f}, S:{sgain:.2f}, V:{vgain:.2f})')
    plt.axis('off')
    
    # HSV色彩空间可视化
    plt.subplot(1, 3, 3)
    hsv_img = cv2.cvtColor(adjusted_img, cv2.COLOR_BGR2HSV)
    plt.imshow(hsv_img)
    plt.title('HSV色彩空间')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 保存结果
    if save_path:
        cv2.imwrite(str(save_path), adjusted_img)
        print(f"结果已保存到: {save_path}")
    
    return adjusted_img

def interactive_hsv_demo(image_path):
    """
    交互式HSV调整演示
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"无法读取图像: {image_path}")
        return
    
    # 预设一些效果
    effects = [
        ("原图", 0.5, 0.5, 0.5),
        ("增强饱和度", 0.5, 0.8, 0.5),
        ("减少饱和度", 0.5, 0.2, 0.5),
        ("增加亮度", 0.5, 0.5, 0.7),
        ("减少亮度", 0.5, 0.5, 0.3),
        ("暖色调", 0.6, 0.7, 0.6),
        ("冷色调", 0.4, 0.7, 0.6),
        ("复古效果", 0.55, 0.3, 0.4),
        ("鲜艳效果", 0.5, 1.0, 0.7)
    ]
    
    plt.figure(figsize=(20, 12))
    
    for i, (name, h, s, v) in enumerate(effects):
        adjuster = HSVAdjuster(hgain=h, sgain=s, vgain=v)
        adjusted_img = adjuster.adjust_hsv(img)
        
        plt.subplot(3, 3, i + 1)
        plt.imshow(cv2.cvtColor(adjusted_img, cv2.COLOR_BGR2RGB))
        plt.title(f'{name}\nH:{h:.2f}, S:{s:.2f}, V:{v:.2f}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# 使用示例
if __name__ == "__main__":
    # 示例图片路径 (请替换为您的图片路径)
    image_path = "/mnt/datasets/manfenglin/ISDD/images/train/000513.png"  # 替换为您的图片路径

    # 创建一个示例图片 (如果没有测试图片)
    if not Path(image_path).exists():
        print("创建示例图片...")
        # 创建一个彩色示例图片
        sample_img = np.zeros((300, 400, 3), dtype=np.uint8)
        sample_img[:100, :, :] = [255, 100, 100]  # 红色区域
        sample_img[100:200, :, :] = [100, 255, 100]  # 绿色区域  
        sample_img[200:, :, :] = [100, 100, 255]  # 蓝色区域
        cv2.imwrite(image_path, sample_img)
    
    print("=== HSV颜色调整演示 ===")
    
    # 1. 单个效果演示
    print("\n1. 增加饱和度效果:")
    process_image(image_path, hgain=0.5, sgain=0.8, vgain=0.5)
    
    print("\n2. 调整色调效果:")
    process_image(image_path, hgain=0.7, sgain=0.5, vgain=0.5)
    
    print("\n3. 调整亮度效果:")
    process_image(image_path, hgain=0.5, sgain=0.5, vgain=0.8)
    
    print("\n4. 综合调整效果:")
    process_image(image_path, hgain=0.6, sgain=0.7, vgain=0.6, 
                 save_path="adjusted_image.jpg")
    
    # 2. 交互式演示
    print("\n5. 多种效果对比:")
    interactive_hsv_demo(image_path)
    
    print("\n=== 参数说明 ===")
    print("hgain: 色调调整 [0, 1]")
    print("  0.5: 无调整")
    print("  > 0.5: 向暖色调偏移")
    print("  < 0.5: 向冷色调偏移")
    print("\nsgain: 饱和度调整 [0, 1]")
    print("  0.5: 无调整")
    print("  > 0.5: 增加饱和度(更鲜艳)")
    print("  < 0.5: 减少饱和度(更灰暗)")
    print("\nvgain: 明度调整 [0, 1]")
    print("  0.5: 无调整")
    print("  > 0.5: 增加亮度")
    print("  < 0.5: 减少亮度")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
游戏图标分类器
自动检测和分类图像中的圆形图标
"""

import cv2
import numpy as np
from PIL import Image
import os
import json
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime

class IconClassifier:
    """图标分类器类"""
    
    def __init__(self):
        self.icon_types = {
            'yellow_leaf': '黄色叶子图标',      # 敏捷/速度类
            'red_spiral': '红色螺旋图标',       # 攻击/伤害类  
            'pink_flower': '粉色花朵图标',      # 治疗/恢复类
            'blue_shield': '蓝色盾牌图标',      # 防御/保护类
            'green_nature': '绿色自然图标',     # 自然/生命类
            'purple_magic': '紫色魔法图标',     # 魔法/能量类
            'orange_power': '橙色力量图标',     # 力量/强化类
            'unknown': '未知图标'
        }
        
        # 预定义的颜色范围 (HSV色彩空间)
        self.color_ranges = {
            'yellow': [(15, 50, 50), (35, 255, 255)],      # 黄色范围
            'red1': [(0, 50, 50), (10, 255, 255)],         # 红色范围1
            'red2': [(170, 50, 50), (180, 255, 255)],      # 红色范围2  
            'pink': [(140, 50, 50), (170, 255, 255)],      # 粉色范围
            'blue': [(100, 50, 50), (130, 255, 255)],      # 蓝色范围
            'green': [(40, 50, 50), (80, 255, 255)],       # 绿色范围
            'purple': [(130, 50, 50), (150, 255, 255)],    # 紫色范围
            'orange': [(10, 50, 50), (25, 255, 255)]       # 橙色范围
        }
    
    def detect_circular_icons(self, image_path: str, min_radius: int = 15, max_radius: int = 50) -> List[Dict]:
        """检测图像中的圆形图标"""
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")
        
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊减少噪声
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # 使用HoughCircles检测圆形
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,               # 累加器分辨率倍数
            minDist=30,         # 圆心之间的最小距离
            param1=50,          # Canny边缘检测高阈值
            param2=30,          # 累加器阈值
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        detected_icons = []
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for i, (x, y, r) in enumerate(circles):
                # 确保圆形在图像边界内
                if (x - r >= 0 and y - r >= 0 and 
                    x + r < img.shape[1] and y + r < img.shape[0]):
                    
                    # 提取圆形区域
                    icon_region = img[y-r:y+r, x-r:x+r]
                    
                    # 创建圆形掩码
                    mask = np.zeros((2*r, 2*r), dtype=np.uint8)
                    cv2.circle(mask, (r, r), r, 255, -1)
                    
                    # 应用掩码
                    masked_icon = cv2.bitwise_and(icon_region, icon_region, mask=mask)
                    
                    icon_info = {
                        'id': i,
                        'center': (x, y),
                        'radius': r,
                        'bbox': (x-r, y-r, x+r, y+r),
                        'icon_image': masked_icon,
                        'mask': mask
                    }
                    
                    detected_icons.append(icon_info)
        
        print(f"检测到 {len(detected_icons)} 个圆形图标")
        return detected_icons
    
    def extract_icon_features(self, icon_image: np.ndarray, mask: np.ndarray) -> Dict:
        """提取图标特征"""
        features = {}
        
        # 1. 颜色特征
        hsv_icon = cv2.cvtColor(icon_image, cv2.COLOR_BGR2HSV)
        
        # 计算主要颜色
        dominant_color = self.get_dominant_color(icon_image, mask)
        features['dominant_color_bgr'] = dominant_color
        features['dominant_color_name'] = self.classify_color(dominant_color)
        
        # 计算颜色直方图
        hist_h = cv2.calcHist([hsv_icon], [0], mask, [180], [0, 180])
        hist_s = cv2.calcHist([hsv_icon], [1], mask, [256], [0, 256])
        hist_v = cv2.calcHist([hsv_icon], [2], mask, [256], [0, 256])
        
        # 归一化直方图
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        
        features['color_hist'] = {
            'hue': hist_h.tolist(),
            'saturation': hist_s.tolist(),
            'value': hist_v.tolist()
        }
        
        # 2. 纹理特征
        gray_icon = cv2.cvtColor(icon_image, cv2.COLOR_BGR2GRAY)
        
        # 边缘检测
        edges = cv2.Canny(gray_icon, 50, 150)
        edge_pixels = np.sum(edges > 0)
        total_pixels = np.sum(mask > 0)
        features['edge_density'] = edge_pixels / total_pixels if total_pixels > 0 else 0
        
        # 纹理复杂度 (使用标准差)
        masked_gray = cv2.bitwise_and(gray_icon, gray_icon, mask=mask)
        features['texture_complexity'] = np.std(masked_gray[mask > 0]) if np.sum(mask > 0) > 0 else 0
        
        # 3. 形状特征
        # 轮廓检测
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            # 圆形度
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0
            features['circularity'] = circularity
            
            # 凸包比率
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                convexity = area / hull_area
            else:
                convexity = 0
            features['convexity'] = convexity
        else:
            features['circularity'] = 0
            features['convexity'] = 0
        
        return features
    
    def get_dominant_color(self, image: np.ndarray, mask: np.ndarray) -> Tuple[int, int, int]:
        """获取图像的主要颜色"""
        # 将图像重塑为像素数组
        pixels = image.reshape(-1, 3)
        mask_flat = mask.reshape(-1)
        
        # 只保留掩码内的像素
        valid_pixels = pixels[mask_flat > 0]
        
        if len(valid_pixels) == 0:
            return (0, 0, 0)
        
        # 使用K-means聚类找到主要颜色
        kmeans = KMeans(n_clusters=min(3, len(valid_pixels)), random_state=42, n_init=10)
        kmeans.fit(valid_pixels)
        
        # 返回最大簇的中心颜色
        colors = kmeans.cluster_centers_
        labels = kmeans.labels_
        label_counts = np.bincount(labels)
        dominant_color = colors[np.argmax(label_counts)]
        
        return tuple(map(int, dominant_color))
    
    def classify_color(self, bgr_color: Tuple[int, int, int]) -> str:
        """根据BGR颜色分类颜色名称"""
        # 转换BGR到HSV
        bgr_array = np.uint8([[bgr_color]])
        hsv_array = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2HSV)
        h, s, v = hsv_array[0][0]
        
        # 检查各个颜色范围
        for color_name, (lower, upper) in self.color_ranges.items():
            if (lower[0] <= h <= upper[0] and 
                lower[1] <= s <= upper[1] and 
                lower[2] <= v <= upper[2]):
                return color_name
        
        # 特殊处理红色（跨越0度）
        if (h <= 10 or h >= 170) and s >= 50 and v >= 50:
            return 'red'
        
        # 如果都不匹配，返回灰色或其他
        if s < 50:
            return 'gray'
        else:
            return 'other'
    
    def classify_icon_type(self, features: Dict) -> str:
        """根据特征分类图标类型"""
        color = features['dominant_color_name']
        edge_density = features['edge_density']
        texture_complexity = features['texture_complexity']
        circularity = features['circularity']
        
        # 基于颜色和特征的简单分类规则
        if color in ['yellow', 'orange']:
            if texture_complexity > 50:
                return 'yellow_leaf'  # 黄色叶子，通常有复杂纹理
            else:
                return 'orange_power'  # 橙色力量图标
        
        elif color in ['red', 'red1', 'red2']:
            if edge_density > 0.3:
                return 'red_spiral'  # 红色螺旋，边缘密度高
            else:
                return 'red_spiral'
        
        elif color == 'pink':
            return 'pink_flower'  # 粉色花朵图标
        
        elif color == 'blue':
            return 'blue_shield'  # 蓝色盾牌图标
        
        elif color == 'green':
            return 'green_nature'  # 绿色自然图标
        
        elif color == 'purple':
            return 'purple_magic'  # 紫色魔法图标
        
        else:
            return 'unknown'
    
    def process_image(self, image_path: str) -> Dict:
        """处理单张图像，检测并分类所有图标"""
        print(f"正在处理图像: {image_path}")
        
        # 检测圆形图标
        detected_icons = self.detect_circular_icons(image_path)
        
        classified_icons = []
        
        for icon in detected_icons:
            # 提取特征
            features = self.extract_icon_features(icon['icon_image'], icon['mask'])
            
            # 分类图标
            icon_type = self.classify_icon_type(features)
            
            # 合并信息
            classified_icon = {
                'id': icon['id'],
                'center': icon['center'],
                'radius': icon['radius'],
                'bbox': icon['bbox'],
                'type': icon_type,
                'type_name': self.icon_types.get(icon_type, '未知类型'),
                'features': {
                    'dominant_color': features['dominant_color_bgr'],
                    'color_name': features['dominant_color_name'],
                    'edge_density': features['edge_density'],
                    'texture_complexity': features['texture_complexity'],
                    'circularity': features['circularity'],
                    'convexity': features['convexity']
                }
            }
            
            classified_icons.append(classified_icon)
            
            print(f"  图标 {icon['id']}: {classified_icon['type_name']} "
                  f"(颜色: {features['dominant_color_name']}, "
                  f"位置: {icon['center']}, "
                  f"半径: {icon['radius']})")
        
        # 统计结果
        type_counts = {}
        for icon in classified_icons:
            icon_type = icon['type']
            type_counts[icon_type] = type_counts.get(icon_type, 0) + 1
        
        result = {
            'image_path': image_path,
            'total_icons': len(classified_icons),
            'classified_icons': classified_icons,
            'type_counts': type_counts,
            'type_distribution': {self.icon_types[k]: v for k, v in type_counts.items()},
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def save_visualization(self, image_path: str, result: Dict, output_path: str):
        """保存可视化结果"""
        # 读取原图
        img = cv2.imread(image_path)
        vis_img = img.copy()
        
        # 颜色映射
        type_colors = {
            'yellow_leaf': (0, 255, 255),      # 黄色
            'red_spiral': (0, 0, 255),         # 红色
            'pink_flower': (255, 0, 255),      # 品红色
            'blue_shield': (255, 0, 0),        # 蓝色
            'green_nature': (0, 255, 0),       # 绿色
            'purple_magic': (255, 0, 128),     # 紫色
            'orange_power': (0, 165, 255),     # 橙色
            'unknown': (128, 128, 128)         # 灰色
        }
        
        # 绘制检测结果
        for icon in result['classified_icons']:
            center = icon['center']
            radius = icon['radius']
            icon_type = icon['type']
            
            color = type_colors.get(icon_type, (128, 128, 128))
            
            # 绘制圆形边界
            cv2.circle(vis_img, center, radius, color, 2)
            
            # 绘制中心点
            cv2.circle(vis_img, center, 3, color, -1)
            
            # 添加文本标签
            label = f"{icon['id']}:{icon_type}"
            label_pos = (center[0] - radius, center[1] - radius - 10)
            cv2.putText(vis_img, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 1, cv2.LINE_AA)
        
        # 保存结果
        cv2.imwrite(output_path, vis_img)
        print(f"可视化结果已保存: {output_path}")


def main():
    """主函数 - 演示图标分类功能"""
    print("=" * 60)
    print("游戏图标分类器")
    print("=" * 60)
    
    # 创建分类器实例
    classifier = IconClassifier()
    
    # 处理123.png图像
    image_path = '/home/sim/code/BPSR_M_OCR/screenshot/123.png'
    
    if not os.path.exists(image_path):
        print(f"图像文件不存在: {image_path}")
        return
    
    try:
        # 分类图标
        result = classifier.process_image(image_path)
        
        # 显示结果
        print(f"\n分类结果:")
        print(f"总计检测到 {result['total_icons']} 个图标")
        print(f"\n类型分布:")
        for type_name, count in result['type_distribution'].items():
            print(f"  {type_name}: {count} 个")
        
        print(f"\n详细信息:")
        for icon in result['classified_icons']:
            print(f"  图标 {icon['id']}:")
            print(f"    类型: {icon['type_name']}")
            print(f"    位置: {icon['center']}")
            print(f"    半径: {icon['radius']}")
            print(f"    主色调: {icon['features']['color_name']}")
            print(f"    边缘密度: {icon['features']['edge_density']:.3f}")
            print(f"    纹理复杂度: {icon['features']['texture_complexity']:.1f}")
        
        # 保存可视化结果
        output_viz = '/home/sim/code/BPSR_M_OCR/icon_classification_result.png'
        classifier.save_visualization(image_path, result, output_viz)
        
        # 保存JSON结果
        output_json = '/home/sim/code/BPSR_M_OCR/icon_classification_result.json'
        # 移除不能序列化的图像数据
        clean_result = {k: v for k, v in result.items() if k != 'classified_icons'}
        clean_icons = []
        for icon in result['classified_icons']:
            clean_icon = {k: v for k, v in icon.items() if k not in ['features']}
            clean_icon['features'] = {k: v for k, v in icon['features'].items() 
                                    if not isinstance(v, (np.ndarray, np.integer, np.floating))}
            # 转换numpy类型为Python原生类型
            for key, value in clean_icon['features'].items():
                if isinstance(value, (np.integer, np.floating)):
                    clean_icon['features'][key] = float(value)
            clean_icons.append(clean_icon)
        clean_result['classified_icons'] = clean_icons
        
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(clean_result, f, ensure_ascii=False, indent=2)
        
        print(f"\nJSON结果已保存: {output_json}")
        print("分析完成！")
        
    except Exception as e:
        print(f"处理失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

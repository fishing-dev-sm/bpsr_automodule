#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模组OCR识别和最优组合计算器
识别游戏模组截图中的属性数值，并计算最优装备组合
"""

import cv2
import pytesseract
import numpy as np
import platform
from PIL import Image
import os
import re
from itertools import combinations
from typing import Dict, List, Tuple, Optional
import json

class ModuleOCR:
    """模组OCR识别类"""
    
    def __init__(self, engine: str = 'auto'):
        """初始化OCR引擎

        engine: 'auto' | 'paddle' | 'tesseract'
        - auto: 自动检测，优先使用 PaddleOCR，失败则回退 Tesseract
        - paddle: 仅使用 PaddleOCR（不可用则回退 Tesseract）
        - tesseract: 仅使用 Tesseract
        """
        # 引擎选择
        self.requested_engine = engine
        self.paddle_available = False
        self.paddle_ocr = None  # 惰性初始化

        # 配置Tesseract OCR路径（作为回退与备选）
        self.setup_tesseract()

        # 尝试加载 PaddleOCR
        self._try_init_paddle()
        
        # 游戏中的所有属性名称映射
        self.attribute_names = {
            '敏捷加持': 'agility_boost',
            '特攻伤害加持': 'special_attack_damage',
            '精英打击': 'elite_strike',
            '暴击专注': 'critical_focus',
            '极·伤害叠加': 'extreme_damage_stack',
            '极·灵活身法': 'extreme_agility',
            '极·生命波动': 'extreme_life_fluctuation',
            '力量加持': 'strength_boost',
            '智力加持': 'intelligence_boost',
            '特攻治疗加持': 'special_healing_boost',
            '专精治疗加持': 'specialized_healing',
            '抵御魔法': 'magic_resistance',
            '抵御物理': 'physical_resistance',
            '施法专注': 'spellcasting_focus',
            '攻速专注': 'attack_speed_focus'
        }
        
        # 设置tesseract配置
        self.tesseract_config = '--oem 3 --psm 6 -l chi_sim'

    def _try_init_paddle(self):
        """尝试初始化 PaddleOCR（仅在可用时）"""
        if self.requested_engine == 'tesseract':
            self.paddle_available = False
            return
        try:
            from paddleocr import PaddleOCR  # type: ignore
            # 惰性创建：第一次真正使用时再实例化，避免加载延迟
            self.PaddleOCRClass = PaddleOCR
            self.paddle_available = True
            print("PaddleOCR 可用，将优先使用中文OCR模型")
        except Exception as e:
            self.paddle_available = False
            if self.requested_engine in ('auto', 'paddle'):
                print(f"PaddleOCR 不可用，回退到 Tesseract（原因: {str(e)[:120]}）")
    
    def setup_tesseract(self):
        """配置Tesseract OCR路径"""
        if platform.system() == 'Windows':
            # Windows常见的Tesseract安装路径
            possible_paths = [
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                r'C:\Tesseract-OCR\tesseract.exe'
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    print(f"找到Tesseract: {path}")
                    return
            
            # 如果都没找到，尝试从PATH中查找
            try:
                import subprocess
                result = subprocess.run(['where', 'tesseract'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    pytesseract.pytesseract.tesseract_cmd = result.stdout.strip().split('\n')[0]
                    print(f"从PATH找到Tesseract: {pytesseract.pytesseract.tesseract_cmd}")
                    return
            except:
                pass
            
            print("警告: 未找到Tesseract OCR，请确保已正确安装")
        else:
            # Linux/Mac系统通常tesseract在PATH中
            print("Linux/Mac系统，使用默认tesseract命令")
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """预处理图像以提高OCR识别准确率"""
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")
        
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 应用高斯模糊减少噪声
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 应用阈值处理
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 形态学操作去除噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def extract_numbers_from_image(self, image_path: str) -> List[int]:
        """专门提取图像中的数字（优先使用 PaddleOCR）"""
        try:
            # 若 Paddle 可用，直接用 PaddleOCR 的文本结果来提取数字（更稳定）
            if self.paddle_available:
                text = self._paddle_extract_text(image_path)
                numbers = re.findall(r'\+?(\d+)', text)
            else:
                img = cv2.imread(image_path)
                numbers = []
                # Tesseract 策略集
                config_plus_digits = '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789+'
                processed_img = self.preprocess_image(image_path)
                digit_text1 = pytesseract.image_to_string(processed_img, config=config_plus_digits)
                numbers.extend(re.findall(r'\+?(\d+)', digit_text1))
                numbers.extend(re.findall(r'\d+', digit_text1))
                config_digits = '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789'
                digit_text2 = pytesseract.image_to_string(processed_img, config=config_digits)
                numbers.extend(re.findall(r'\d+', digit_text2))
                enhanced_img = self.enhance_for_digits(img)
                digit_text3 = pytesseract.image_to_string(enhanced_img, config=config_plus_digits)
                numbers.extend(re.findall(r'\+?(\d+)', digit_text3))
                for psm in [6, 7, 13]:
                    config_psm = f'--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789+'
                    try:
                        digit_text = pytesseract.image_to_string(processed_img, config=config_psm)
                        numbers.extend(re.findall(r'\+?(\d+)', digit_text))
                        numbers.extend(re.findall(r'\d+', digit_text))
                    except:
                        continue

            # 过滤到 1-10，并处理大于10取个位
            valid_numbers = []
            for num_str in numbers:
                try:
                    num = int(num_str)
                    if 1 <= num <= 10:
                        valid_numbers.append(num)
                    elif num > 10:
                        digit = num % 10
                        if 1 <= digit <= 10:
                            valid_numbers.append(digit)
                except:
                    continue

            print(f"  最终识别数字: {valid_numbers}")
            return valid_numbers
        except Exception as e:
            print(f"数字识别失败 {image_path}: {str(e)}")
            return []

    def enhance_for_digits(self, img: np.ndarray) -> np.ndarray:
        """专门为数字识别优化图像"""
        # 转换为灰度
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # 高斯模糊去噪
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 自适应阈值处理
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # 形态学操作清理噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # 增强对比度
        enhanced = cv2.equalizeHist(cleaned)
        
        return enhanced

    def extract_text_from_image(self, image_path: str) -> str:
        """从图像中提取文本（优先使用 PaddleOCR）"""
        try:
            if self.paddle_available:
                combined_text = self._paddle_extract_text(image_path)
                print(f"  OCR文本结果(Paddle): {repr(combined_text[:150])}")
                return combined_text.strip()
            else:
                # Tesseract 路径
                processed_img = self.preprocess_image(image_path)
                text1 = pytesseract.image_to_string(processed_img, config=self.tesseract_config)
                original_img = cv2.imread(image_path)
                text2 = pytesseract.image_to_string(original_img, config=self.tesseract_config)
                config_alt = '--oem 3 --psm 6 -l chi_sim'
                text3 = pytesseract.image_to_string(processed_img, config=config_alt)
                combined_text = f"{text1}\n{text2}\n{text3}"
                print(f"  OCR文本结果(Tesseract): {repr(combined_text[:150])}")
                return combined_text.strip()
        except Exception as e:
            print(f"OCR识别失败 {image_path}: {str(e)}")
            return ""

    def _ensure_paddle_instance(self):
        """确保 PaddleOCR 实例已创建"""
        if self.paddle_available and self.paddle_ocr is None:
            # 使用中文模型，开启方向分类
            self.paddle_ocr = self.PaddleOCRClass(use_angle_cls=True, lang='ch')

    def _paddle_extract_text(self, image_path: str) -> str:
        """使用 PaddleOCR 提取中文文本，返回合并后的文本"""
        self._ensure_paddle_instance()
        if not self.paddle_available or self.paddle_ocr is None:
            return ""
        # 不传 cls 参数，避免不同版本 API 差异
        result = self.paddle_ocr.ocr(image_path)
        lines = []
        try:
            # 兼容两种返回：
            # 1) 直接为行列表：[[box], (text, conf)]
            # 2) 外层包了一层：[[[box], (text, conf)], ...]
            lines_iter = None
            if isinstance(result, list) and result:
                # 情况2：外层包一层
                first = result[0]
                if isinstance(first, list) and first and isinstance(first[0], list) and len(first[0]) == 2:
                    # 形如 result = [[line, line, ...]]
                    lines_iter = first
                else:
                    # 情况1：直接为行列表
                    lines_iter = result

            if lines_iter is None:
                return ""

            for line in lines_iter:
                # line: [box, (text, score)]
                if isinstance(line, (list, tuple)) and len(line) >= 2:
                    meta = line[1]
                    if isinstance(meta, (list, tuple)) and meta and isinstance(meta[0], str):
                        text = meta[0]
                        if text.strip():
                            lines.append(text.strip())
        except Exception:
            pass
        return "\n".join(lines)
    
    def classify_module_quality(self, attributes: Dict[str, int], fallback_entry_count: Optional[int] = None) -> str:
        """根据词条数量分类模组品质

        规则：
        - 当已识别属性数为2时，直接判为紫色（保持原有正确逻辑，不做变更）
        - 当已识别属性数>=3时，判为金色
        - 当已识别属性数<2时，使用fallback_entry_count（由数字候选推断的词条数）进行回退判断：
          - >=3 判为金色
          - ==2 判为紫色
          - ==1 判为蓝色
          - 其他 判为白色
        """
        attr_count = len(attributes)
        
        # 若回退推断为三词条及以上，优先判定为金色（覆盖仅识别到2词条的情况）
        if fallback_entry_count is not None and fallback_entry_count >= 3:
            return 'legendary'

        # 已识别达到三词条（正常金色）
        if attr_count >= 3:
            return 'legendary'

        # 使用回退词条数进行判断
        if fallback_entry_count is not None:
            if fallback_entry_count >= 3:
                return 'legendary'
            if fallback_entry_count == 2:
                return 'epic'
            if fallback_entry_count == 1:
                return 'rare'
            return 'common'
        
        # 未提供回退时，保持原有紫色逻辑
        if attr_count == 2:
            return 'epic'

        # 无回退信息时，按已识别结果返回
        if attr_count == 1:
            return 'rare'
        return 'common'

    def parse_module_attributes(self, text: str, filename: str, image_path: str = None) -> Dict[str, any]:
        """解析模组属性和数值，并分类品质"""
        attributes = {}
        
        # 分行处理文本
        lines = text.split('\n')
        all_text = ' '.join(lines)  # 也尝试整体文本匹配
        
        print(f"  解析文本内容: {repr(all_text[:300])}")
        
        # 常见OCR误识别别名替换 -> 标准属性名
        alias_map = {
            '攻适专注': '攻速专注',   # 速->适
            '攻逝专注': '攻速专注',   # OCR相近
            '攻速专主': '攻速专注',
            '章迁专注': '施法专注',   # 施法->章迁（常见形近）
            '幸运专注': '暴击专注',   # 暴击->幸运（个别截图)
            '暴机专注': '暴击专注',
            '暴級专注': '暴击专注',
            '施法专主': '施法专注',
        }
        for alias, canonical in alias_map.items():
            if alias in all_text:
                all_text = all_text.replace(alias, canonical)
                print(f"    文本规范化: {alias} -> {canonical}")

        # 获取专门识别的数字
        valid_numbers = []
        if image_path:
            valid_numbers = self.extract_numbers_from_image(image_path)
        
        # 从文本中提取数字，处理多种格式
        all_numbers = []
        
        # 1. 匹配 "+数字" 格式
        plus_matches = re.findall(r'\+(\d+)', all_text)
        for match in plus_matches:
            num = int(match)
            if 1 <= num <= 10:
                all_numbers.append(num)
            elif num > 10:
                # 取个位数（如+19->9）
                digit = num % 10
                if 1 <= digit <= 10:
                    all_numbers.append(digit)
                    print(f"    数字修正: +{num} -> {digit}")
        
        # 2. 匹配 "属性名+数字" 或 "属性名数字" 格式（如"暴击专注19"）
        for attr_name in self.attribute_names.keys():
            # 匹配 "属性名+数字"
            pattern1 = rf'{re.escape(attr_name)}\+(\d+)'
            matches1 = re.findall(pattern1, all_text)
            for match in matches1:
                # 如果数字超过10，尝试取个位数
                num = int(match)
                if 1 <= num <= 10:
                    all_numbers.append(num)
                elif num > 10:
                    # 取个位数（如19->9, 28->8）
                    digit = num % 10
                    if 1 <= digit <= 10:
                        all_numbers.append(digit)
                        print(f"    数字修正: {attr_name}+{num} -> {digit}")
            
            # 匹配 "属性名数字"（紧贴，如"暴击专注19"）
            pattern2 = rf'{re.escape(attr_name)}(\d+)'
            matches2 = re.findall(pattern2, all_text)
            for match in matches2:
                # 如果数字超过10，尝试取个位数
                num = int(match)
                if 1 <= num <= 10:
                    all_numbers.append(num)
                elif num > 10:
                    # 取个位数（如19->9, 28->8）
                    digit = num % 10
                    if 1 <= digit <= 10:
                        all_numbers.append(digit)
                        print(f"    数字修正: {attr_name}{num} -> {digit}")
        
        # 3. 匹配单独的数字（但排除已经在属性名后的）
        #    允许数字前后出现分隔符或噪声字符（如'|'、'。'、引号等）
        standalone_numbers = []
        for match in re.finditer(r'(?<![A-Za-z\u4e00-\u9fa5])([1-9]\d?)(?![A-Za-z\u4e00-\u9fa5])', all_text):
            num = int(match.group(1))
            start_pos = match.start()
            before_text = all_text[max(0, start_pos-15):start_pos]
            is_after_attr = any(attr_name in before_text for attr_name in self.attribute_names.keys())
            
            if not is_after_attr:
                if 1 <= num <= 10:
                    standalone_numbers.append(num)
                elif num > 10:
                    # 取个位数
                    digit = num % 10
                    if 1 <= digit <= 10:
                        standalone_numbers.append(digit)
                        print(f"    数字修正: 独立{num} -> {digit}")
        
        all_numbers.extend(standalone_numbers)
        
        # 合并专门识别的数字和文本提取的数字
        text_numbers = all_numbers
        all_candidate_numbers = list(set(valid_numbers + text_numbers))
        
        # 基于数字候选推断词条数量（用于品质判断的回退逻辑）
        # 取值限制在1-10，计算不重复的个数
        unique_numeric_entries = []
        for n in all_candidate_numbers:
            try:
                iv = int(n)
                if 1 <= iv <= 10 and iv not in unique_numeric_entries:
                    unique_numeric_entries.append(iv)
            except Exception:
                continue
        inferred_entry_count = len(unique_numeric_entries)
        
        print(f"  候选数字: {all_candidate_numbers}")
        
        # 找到的属性名称
        found_attributes = []
        
        # 尝试匹配属性名称（在规范化文本后进行）
        for attr_name, attr_key in self.attribute_names.items():
            # 尝试精确匹配（包括带+数字的格式）
            if attr_name in all_text:
                found_attributes.append((attr_name, attr_key, 'exact'))
                continue
            
            # 匹配"属性名+数字"格式，如"暴击专注+9"
            pattern_plus = rf'{re.escape(attr_name)}\+\d+'
            if re.search(pattern_plus, all_text):
                found_attributes.append((attr_name, attr_key, 'plus_format'))
                continue
            
            # 匹配"属性名数字"格式，如"暴击专注19"（OCR误识别+号的情况）
            pattern_direct = rf'{re.escape(attr_name)}\d+'
            if re.search(pattern_direct, all_text):
                found_attributes.append((attr_name, attr_key, 'direct_number'))
                continue
            
            # 尝试部分匹配（去掉特殊字符）
            attr_simple = attr_name.replace('·', '').replace('特攻', '特殊攻击')
            if attr_simple in all_text or any(part in all_text for part in attr_simple.split() if len(part) > 1):
                found_attributes.append((attr_name, attr_key, 'partial'))
                continue
            
            # 尝试匹配"简化属性名+数字"格式
            pattern_simple_plus = rf'{re.escape(attr_simple)}\+\d+'
            if re.search(pattern_simple_plus, all_text):
                found_attributes.append((attr_name, attr_key, 'simple_plus_format'))
                continue
                
            # 尝试匹配"简化属性名数字"格式
            pattern_simple_direct = rf'{re.escape(attr_simple)}\d+'
            if re.search(pattern_simple_direct, all_text):
                found_attributes.append((attr_name, attr_key, 'simple_direct'))
                continue
        
        # 如果没找到精确属性，尝试关键词匹配
        if not found_attributes:
            keywords = {
                '敏捷': ('敏捷加持', 'agility_boost'),
                '力量': ('力量加持', 'strength_boost'), 
                '智力': ('智力加持', 'intelligence_boost'),
                '暴击': ('暴击专注', 'critical_focus'),
                '治疗': ('专精治疗加持', 'specialized_healing'),
                '伤害': ('极·伤害叠加', 'extreme_damage_stack'),
                '物理': ('抵御物理', 'physical_resistance'),
                '魔法': ('抵御魔法', 'magic_resistance'),
                '精英': ('精英打击', 'elite_strike'),
                '施法': ('施法专注', 'spellcasting_focus'),
                '攻速': ('攻速专注', 'attack_speed_focus')
            }
            
            for keyword, (attr_name, attr_key) in keywords.items():
                if keyword in all_text:
                    found_attributes.append((attr_name, attr_key, 'keyword'))
        
        print(f"  找到属性: {[(name, type_) for name, _, type_ in found_attributes]}")
        
        # 智能分配数字给属性
        if found_attributes:
            # 首先尝试从文本中精确提取每个属性对应的数字
            for attr_name, attr_key, match_type in found_attributes:
                assigned = False
                
                # 尝试从文本中提取这个属性紧邻的数字
                for pattern in [
                    rf'{re.escape(attr_name)}\+(\d+)',  # "属性名+数字"
                    rf'{re.escape(attr_name)}\s*\+(\d+)',  # 允许空格
                    rf'{re.escape(attr_name)}\s*(\d+)',    # 允许紧贴或空格
                ]:
                    matches = re.findall(pattern, all_text)
                    if matches:
                        for match in matches:
                            raw_value = int(match)
                            if 1 <= raw_value <= 10:
                                attributes[attr_key] = raw_value
                                print(f"    精确分配: {attr_name} = {raw_value} (从文本提取)")
                                assigned = True
                                break
                            elif raw_value > 10:
                                # 取个位数
                                digit = raw_value % 10
                                if 1 <= digit <= 10:
                                    attributes[attr_key] = digit
                                    print(f"    精确分配(修正): {attr_name} = {digit} (原值{raw_value}->个位数)")
                                    assigned = True
                                    break
                    if assigned:
                        break
                
                # 如果无法精确分配，使用候选数字
                if not assigned and all_candidate_numbers:
                    # 使用还未被分配的数字
                    used_numbers = set(attributes.values())
                    available_numbers = [n for n in all_candidate_numbers if n not in used_numbers]
                    
                    if available_numbers:
                        attributes[attr_key] = available_numbers[0]
                        print(f"    候选分配: {attr_name} = {available_numbers[0]} ({match_type})")
                    elif all_candidate_numbers:
                        # 如果没有未使用的数字，使用第一个候选数字
                        attributes[attr_key] = all_candidate_numbers[0]
                        print(f"    重复分配: {attr_name} = {all_candidate_numbers[0]} ({match_type})")
        
        # 分类模组品质（当识别不足时，使用数字候选数量作为回退）
        quality = self.classify_module_quality(attributes, fallback_entry_count=inferred_entry_count)
        quality_names = {
            'legendary': '金色',
            'epic': '紫色', 
            'rare': '蓝色',
            'common': '白色'
        }
        
        # 返回包含属性和品质的字典
        return {
            'attributes': attributes,
            'quality': quality,
            'quality_name': quality_names.get(quality, '未知'),
            # 展示的词条数优先使用回退推断结果，以反映真实词条个数
            'attribute_count': max(len(attributes), inferred_entry_count),
            'inferred_entry_count': inferred_entry_count,
            'inferred_numbers': unique_numeric_entries
        }
    
    def scan_all_modules(self, screenshot_dir: str) -> Dict[str, Dict]:
        """扫描所有模组截图"""
        modules = {}
        
        # 获取所有图片文件
        image_files = [f for f in os.listdir(screenshot_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"发现 {len(image_files)} 张截图，开始OCR识别...")
        
        for filename in image_files:
            file_path = os.path.join(screenshot_dir, filename)
            print(f"正在处理: {filename}")
            
            # OCR识别
            text = self.extract_text_from_image(file_path)
            
            # 解析属性和品质（传递图像路径用于数字识别）
            module_data = self.parse_module_attributes(text, filename, file_path)
            
            if module_data and module_data['attributes']:
                module_name = filename.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
                modules[module_name] = module_data
                print(f"  识别到属性: {module_data['attributes']}")
                print(f"  模组品质: {module_data['quality_name']} ({module_data['attribute_count']}种属性)")
            else:
                print(f"  未识别到有效属性")
        
        return modules


class ModuleCombinationOptimizer:
    """模组组合优化器"""
    
    def __init__(self, modules: Dict[str, Dict[str, int]]):
        self.modules = modules
        self.max_modules = 4  # 最多装备4个模组
        self.max_attribute_value = 20  # 单属性最大有效值
        self.min_threshold = 16  # 最小阈值，低于此值的组合被视为垃圾
    
    def calculate_combination_score(self, module_combination: List[str]) -> Dict[str, int]:
        """计算模组组合的属性总和"""
        total_attributes = {}
        
        for module_name in module_combination:
            module_data = self.modules[module_name]
            module_attrs = module_data['attributes']  # 获取属性字典
            for attr, value in module_attrs.items():
                total_attributes[attr] = total_attributes.get(attr, 0) + value
        
        # 不再应用20点上限限制，保留真实数值用于排序
        return total_attributes
    
    def evaluate_combination(self, total_attributes: Dict[str, int]) -> Tuple[int, int, List[str]]:
        """评估组合的总体价值"""
        # 计算达到20+的属性数量（主要排序依据）
        maxed_count = len([attr for attr, value in total_attributes.items() 
                          if value >= self.max_attribute_value])
        
        # 计算总分数（次要排序依据）
        total_score = sum(total_attributes.values())
        
        # 找出达到20+的属性
        maxed_attributes = [attr for attr, value in total_attributes.items() 
                           if value >= self.max_attribute_value]
        
        return maxed_count, total_score, maxed_attributes
    
    def find_optimal_combinations(self) -> List[Dict]:
        """找出所有最优组合"""
        all_combinations = []
        
        module_names = list(self.modules.keys())
        
        # 生成所有可能的模组组合（1-4个模组）
        for r in range(1, min(len(module_names) + 1, self.max_modules + 1)):
            for combination in combinations(module_names, r):
                # 计算组合属性
                total_attrs = self.calculate_combination_score(list(combination))
                
                # 过滤掉所有属性都低于阈值的组合
                if not any(value >= self.min_threshold for value in total_attrs.values()):
                    continue
                
                # 评估组合
                maxed_count, total_score, maxed_attrs = self.evaluate_combination(total_attrs)
                
                combination_info = {
                    'modules': list(combination),
                    'attributes': total_attrs,
                    'total_score': total_score,
                    'maxed_count': maxed_count,
                    'maxed_attributes': maxed_attrs,
                    'module_count': len(combination)
                }
                
                all_combinations.append(combination_info)
        
        # 按达到20+的属性数量排序（主要），然后按总分数排序（次要）
        all_combinations.sort(key=lambda x: (x['maxed_count'], x['total_score']), reverse=True)
        
        return all_combinations
    
    def group_by_maxed_attributes(self, combinations: List[Dict]) -> Dict[str, List[Dict]]:
        """按照最大化的属性分组"""
        grouped = {}
        
        for combo in combinations:
            # 创建最大化属性的组合作为键
            maxed_key = tuple(sorted(combo['maxed_attributes']))
            if not maxed_key:
                maxed_key = ('无最大化属性',)
            
            key_str = '+'.join(maxed_key)
            if key_str not in grouped:
                grouped[key_str] = []
            
            grouped[key_str].append(combo)
        
        return grouped


def format_attribute_name(attr_key: str) -> str:
    """将属性键转换为中文显示名称"""
    name_mapping = {
        'agility_boost': '敏捷加持',
        'special_attack_damage': '特攻伤害加持',
        'elite_strike': '精英打击',
        'critical_focus': '暴击专注',
        'extreme_damage_stack': '极·伤害叠加',
        'extreme_agility': '极·灵活身法',
        'extreme_life_fluctuation': '极·生命波动',
        'strength_boost': '力量加持',
        'intelligence_boost': '智力加持',
        'special_healing_boost': '特攻治疗加持',
        'specialized_healing': '专精治疗加持',
        'magic_resistance': '抵御魔法',
        'physical_resistance': '抵御物理',
        'spellcasting_focus': '施法专注',
        'attack_speed_focus': '攻速专注'
    }
    return name_mapping.get(attr_key, attr_key)


def main():
    """主函数"""
    screenshot_dir = '/home/sim/code/BPSR_M_OCR/screenshot'
    
    print("=" * 60)
    print("模组OCR识别和最优组合计算器")
    print("=" * 60)
    
    # 第一步：OCR识别所有模组
    print("\n第一步：OCR识别模组属性...")
    ocr = ModuleOCR()
    modules = ocr.scan_all_modules(screenshot_dir)
    
    if not modules:
        print("未识别到任何有效模组，请检查截图质量或OCR配置")
        return
    
    print(f"\n成功识别 {len(modules)} 个模组:")
    for module_name, module_data in modules.items():
        print(f"  {module_name} [{module_data['quality_name']}品质]:")
        for attr_key, value in module_data['attributes'].items():
            attr_name = format_attribute_name(attr_key)
            print(f"    {attr_name}: {value}")
    
    # 第二步：计算最优组合
    print("\n第二步：计算最优组合...")
    optimizer = ModuleCombinationOptimizer(modules)
    combinations = optimizer.find_optimal_combinations()
    
    if not combinations:
        print("未找到任何有效组合")
        return
    
    # 第三步：按最大化属性分组并显示结果
    print(f"\n第三步：分析结果（共找到 {len(combinations)} 个有效组合）...")
    grouped = optimizer.group_by_maxed_attributes(combinations)
    
    print("\n" + "=" * 80)
    print("最优组合方案（按属性最大化分组，按总收益排序）")
    print("=" * 80)
    
    for group_name, group_combos in grouped.items():
        print(f"\n【{group_name} 最大化组合】")
        print("-" * 50)
        
        for i, combo in enumerate(group_combos[:5]):  # 每组只显示前5个最优方案
            print(f"\n方案 {i+1}:")
            print(f"  模组组合: {' + '.join(combo['modules'])}")
            print(f"  总收益: {combo['total_score']} 点")
            print(f"  使用模组数: {combo['module_count']}/4")
            
            print(f"  属性详情:")
            for attr_key, value in combo['attributes'].items():
                attr_name = format_attribute_name(attr_key)
                status = " (MAX)" if value == 20 else f" ({value}/20)"
                print(f"    {attr_name}: {value}{status}")
    
    # 保存结果到JSON文件
    output_file = '/home/sim/code/BPSR_M_OCR/optimization_results.json'
    result_data = {
        'modules': modules,
        'combinations': combinations[:50],  # 保存前50个最优组合
        'summary': {
            'total_modules': len(modules),
            'total_combinations': len(combinations),
            'max_score': combinations[0]['total_score'] if combinations else 0
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {output_file}")
    print("\n分析完成！")


if __name__ == "__main__":
    main()

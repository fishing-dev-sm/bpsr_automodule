#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模组OCR优化器 Web UI
基于Flask的Web界面，提供美观的结果展示和交互功能
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import os
import json
import threading
import time
from datetime import datetime
from module_ocr_optimizer import ModuleOCR, ModuleCombinationOptimizer, format_attribute_name

def convert_group_name_to_chinese(group_name):
    """将英文属性键组合转换为中文属性名组合"""
    if group_name == '无最大化属性':
        return '无最大化属性'
    
    # 分割属性键（用+连接）
    attr_keys = group_name.split('+')
    chinese_names = []
    
    for attr_key in attr_keys:
        attr_key = attr_key.strip()
        chinese_name = format_attribute_name(attr_key)
        chinese_names.append(chinese_name)
    
    return '+'.join(chinese_names)

app = Flask(__name__)
CORS(app)

# 全局变量存储计算状态和结果
calculation_status = {
    'running': False,
    'progress': 0,
    'current_step': '',
    'error': None,
    'completed': False,
    'start_time': None,
    'end_time': None
}

calculation_results = {
    'modules': {},
    'combinations': [],
    'summary': {},
    'grouped_results': {}
}

def run_ocr_calculation():
    """在后台线程中运行OCR计算"""
    global calculation_status, calculation_results
    
    try:
        calculation_status.update({
            'running': True,
            'progress': 0,
            'current_step': '初始化...',
            'error': None,
            'completed': False,
            'start_time': datetime.now()
        })
        
        screenshot_dir = os.path.join(os.path.dirname(__file__), 'screenshot')
        
        # 第一步：OCR识别
        calculation_status.update({'progress': 10, 'current_step': 'OCR识别模组属性...'})
        ocr = ModuleOCR()
        modules = ocr.scan_all_modules(screenshot_dir)
        
        if not modules:
            raise Exception("未识别到任何有效模组，请检查截图质量")
        
        calculation_status.update({'progress': 40, 'current_step': f'识别到 {len(modules)} 个模组，计算组合...'})
        
        # 第二步：计算最优组合
        optimizer = ModuleCombinationOptimizer(modules)
        combinations = optimizer.find_optimal_combinations()
        
        calculation_status.update({'progress': 70, 'current_step': '分析最优组合...'})
        
        # 第三步：分组结果
        grouped = optimizer.group_by_maxed_attributes(combinations)
        
        calculation_status.update({'progress': 90, 'current_step': '准备结果数据...'})
        
        # 更新全局结果
        calculation_results.update({
            'modules': modules,
            'combinations': combinations,
            'summary': {
                'total_modules': len(modules),
                'total_combinations': len(combinations),
                'max_score': combinations[0]['total_score'] if combinations else 0,
                'calculation_time': None  # 将在完成时计算
            },
            'grouped_results': grouped
        })
        
        # 保存结果文件
        output_file = os.path.join(os.path.dirname(__file__), 'optimization_results.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'modules': modules,
                'combinations': combinations[:50],
                'summary': calculation_results['summary']
            }, f, ensure_ascii=False, indent=2)
        
        calculation_status.update({
            'running': False,
            'progress': 100,
            'current_step': '计算完成！',
            'completed': True,
            'end_time': datetime.now()
        })
        
        # 计算用时
        if calculation_status['start_time'] and calculation_status['end_time']:
            duration = (calculation_status['end_time'] - calculation_status['start_time']).total_seconds()
            calculation_results['summary']['calculation_time'] = f"{duration:.2f}秒"
        
    except Exception as e:
        calculation_status.update({
            'running': False,
            'error': str(e),
            'completed': False,
            'end_time': datetime.now()
        })

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/api/start_calculation', methods=['POST'])
def start_calculation():
    """启动OCR计算"""
    global calculation_status
    
    if calculation_status['running']:
        return jsonify({'error': '计算正在进行中，请等待完成'}), 400
    
    # 重置状态
    calculation_status.update({
        'running': False,
        'progress': 0,
        'current_step': '',
        'error': None,
        'completed': False,
        'start_time': None,
        'end_time': None
    })
    
    # 在后台线程启动计算
    thread = threading.Thread(target=run_ocr_calculation)
    thread.daemon = True
    thread.start()
    
    return jsonify({'message': '计算已启动'})

@app.route('/api/calculation_status')
def get_calculation_status():
    """获取计算状态"""
    return jsonify(calculation_status)

@app.route('/api/results')
def get_results():
    """获取计算结果"""
    if not calculation_status['completed']:
        return jsonify({'error': '计算尚未完成'}), 400
    
    # 格式化结果用于前端显示
    formatted_results = {
        'summary': calculation_results['summary'],
        'modules': {},
        'grouped_combinations': {}
    }
    
    # 格式化模组数据
    for module_name, module_data in calculation_results['modules'].items():
        formatted_attrs = {}
        for attr_key, value in module_data['attributes'].items():
            formatted_attrs[format_attribute_name(attr_key)] = value
        
        formatted_results['modules'][module_name] = {
            'attributes': formatted_attrs,
            'quality': module_data['quality'],
            'quality_name': module_data['quality_name'],
            'attribute_count': module_data.get('attribute_count', len(module_data['attributes'])),
            'inferred_entry_count': module_data.get('inferred_entry_count', len(module_data['attributes']))
        }
    
    # 格式化分组组合数据
    for group_name, combos in calculation_results['grouped_results'].items():
        # 将英文属性键转换为中文属性名
        chinese_group_name = convert_group_name_to_chinese(group_name)
        
        formatted_combos = []
        for combo in combos[:10]:  # 每组最多显示10个方案
            # 生成模组详细信息
            module_details = []
            for module_name in combo['modules']:
                if module_name in calculation_results['modules']:
                    module_data = calculation_results['modules'][module_name]
                    # 格式化为中文属性名
                    chinese_attrs = {}
                    for attr_key, value in module_data['attributes'].items():
                        chinese_name = format_attribute_name(attr_key)
                        chinese_attrs[chinese_name] = value
                    
                    module_details.append({
                        'name': module_name,
                        'attributes': chinese_attrs,
                        'quality': module_data['quality'],
                        'quality_name': module_data['quality_name'],
                        'attribute_count': module_data.get('attribute_count', len(module_data['attributes'])),
                        'inferred_entry_count': module_data.get('inferred_entry_count', len(module_data['attributes']))
                    })
                else:
                    # 如果找不到模组数据，只显示名称
                    module_details.append({
                        'name': module_name,
                        'attributes': {},
                        'quality': 'unknown',
                        'quality_name': '未知',
                        'attribute_count': 0
                    })
            
            formatted_combo = {
                'modules': combo['modules'],  # 保留原始模组名列表
                'module_details': module_details,  # 新增详细信息
                'total_score': combo['total_score'],
                'maxed_count': combo.get('maxed_count', 0),  # 达到20+的属性数量
                'module_count': combo['module_count'],
                'attributes': {}
            }
            
            # 格式化属性名称
            for attr_key, value in combo['attributes'].items():
                attr_name = format_attribute_name(attr_key)
                formatted_combo['attributes'][attr_name] = {
                    'value': value,
                    'is_maxed': value >= 20,  # 20+都标记为maxed
                    'efficiency': f"{min(value, 20)}/20" if value <= 20 else f"{value}(超出)"
                }
            
            formatted_combos.append(formatted_combo)
        
        formatted_results['grouped_combinations'][chinese_group_name] = formatted_combos
    
    return jsonify(formatted_results)

@app.route('/api/screenshot_info')
def get_screenshot_info():
    """获取截图文件夹信息"""
    try:
        screenshot_dir = os.path.join(os.path.dirname(__file__), 'screenshot')
        print(f"检查截图文件夹: {screenshot_dir}")  # 调试信息
        
        if not os.path.exists(screenshot_dir):
            print(f"截图文件夹不存在: {screenshot_dir}")
            return jsonify({'error': '截图文件夹不存在'}), 404
        
        files = [f for f in os.listdir(screenshot_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"找到 {len(files)} 个图片文件")  # 调试信息
        
        return jsonify({
            'total_files': len(files),
            'files': files,
            'directory': screenshot_dir
        })
    except Exception as e:
        print(f"检查截图文件夹时出错: {str(e)}")
        return jsonify({'error': f'检查截图文件夹失败: {str(e)}'}), 500

if __name__ == '__main__':
    # 确保templates和static文件夹存在
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("=" * 60)
    print("模组OCR优化器 Web UI")
    print("=" * 60)
    print("启动Web服务器...")
    print("访问地址: http://localhost:5000")
    print("按 Ctrl+C 停止服务")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)

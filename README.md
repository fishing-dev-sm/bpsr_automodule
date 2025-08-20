# 模组OCR识别和最优组合计算器

这是一个用于游戏模组属性识别和最优装备组合计算的Python程序。

## 功能特性

1. **OCR自动识别**：自动识别截图中的模组属性和数值
2. **智能组合计算**：计算所有可能的模组组合（最多4个模组）
3. **最优解分析**：找出各属性收益最大化的组合方案
4. **垃圾过滤**：自动过滤属性总和低于16点的无效组合
5. **结果分组**：按照最大化属性分组显示结果

## 游戏规则

- 每个模组可以有1-3种属性，每个属性1-10点
- 角色最多装备4个模组
- 单一属性叠加到20点收益最大，超过无效
- 低于16点的组合被视为垃圾组合

## 安装依赖

```bash
pip install -r requirements.txt
```

**注意**：还需要安装tesseract OCR引擎：

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-chi-sim

# macOS
brew install tesseract tesseract-lang

# Windows
# 下载并安装 https://github.com/UB-Mannheim/tesseract/wiki
```

## 使用方法

1. 将模组截图放在 `screenshot/` 文件夹中
2. 运行程序：

```bash
python module_ocr_optimizer.py
```

## 输出结果

程序会输出：

1. **OCR识别结果**：每个模组的属性和数值
2. **最优组合方案**：按属性最大化分组的最佳组合
3. **详细分析**：包括总收益、模组使用数量、属性详情等
4. **JSON文件**：完整结果保存在 `optimization_results.json`

## 输出示例

```
【力量加持+智力加持 最大化组合】
--------------------------------------------------

方案 1:
  模组组合: 模组A + 模组B + 模组C
  总收益: 55 点
  使用模组数: 3/4
  属性详情:
    力量加持: 20 (MAX)
    智力加持: 20 (MAX)
    敏捷加持: 15 (15/20)
```

## 支持的属性

- 敏捷加持
- 特攻伤害加持
- 精英打击
- 暴击专注
- 极·伤害叠加
- 极·灵活身法
- 极·生命波动
- 力量加持
- 智力加持
- 特攻治疗加持
- 专精治疗加持
- 抵御魔法
- 抵御物理
- 施法专注
- 攻速专注

## 注意事项

1. 截图需要清晰，属性名称和数值完整可见
2. 支持PNG、JPG、JPEG格式的图片
3. OCR识别准确率取决于图片质量
4. 程序会自动过滤识别错误的数据

## 文件结构

```
BPSR_M_OCR/
├── module_ocr_optimizer.py    # 主程序
├── requirements.txt           # 依赖列表
├── README.md                 # 说明文档
├── screenshot/               # 截图文件夹
│   ├── Screenshot 2025-08-19 132840.png
│   └── ...
└── optimization_results.json # 输出结果
```

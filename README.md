# BPSR 星痕共鸣 模组 OCR 优化器

<div align="center">

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

</div>

一个专为 BPSR（星痕共鸣）游戏设计的模组属性自动识别和最优装备组合计算工具。支持命令行模式和现代化 Web UI 界面。

## ✨ 功能特性

### 🔍 智能 OCR 识别
- **多策略识别**：采用多种 OCR 策略确保识别准确性
- **数字格式处理**：智能处理 "+9" 格式和数字误识别问题
- **属性名称匹配**：支持精确匹配、关键词匹配等多种模式
- **图像预处理**：自动图像增强提升识别质量

### 🧮 强大的组合优化算法
- **智能组合计算**：计算所有可能的模组组合（最多4个模组）
- **多目标优化**：同时最大化多个属性的收益
- **收益评估**：精确计算每种组合的总收益和效率
- **智能过滤**：自动过滤低价值组合，节省计算时间

### 🎨 现代化 Web 界面
- **响应式设计**：支持桌面和移动设备
- **实时预览**：拖拽上传，实时显示识别结果
- **交互式结果**：可视化组合方案，一目了然
- **数据导出**：支持 JSON 格式结果导出

### 💻 多平台支持
- **命令行版本**：适合批处理和自动化
- **Web UI 版本**：用户友好的图形界面
- **Windows 一键部署**：提供完整的 Windows 部署脚本

## 🎮 游戏规则

| 规则项目 | 说明 |
|---------|------|
| 模组属性 | 每个模组可以有 1-3 种属性，每个属性 1-10 点 |
| 装备限制 | 角色最多装备 4 个模组 |
| 属性上限 | 单一属性叠加到 20 点收益最大，超过无效 |
| 过滤阈值 | 低于 16 点总和的组合被视为垃圾组合 |

## 🚀 快速开始

### Windows 用户（推荐）

1. 下载项目文件
2. 双击运行 `install_windows.bat` 自动安装依赖
3. 选择运行方式：
   - **Web UI 版本**：双击 `start_webui.bat`，浏览器访问 `http://localhost:5000`
   - **命令行版本**：双击 `run_ocr.bat`

### Linux/macOS 用户

#### 1. 安装依赖

```bash
# 安装 Python 依赖
pip install -r requirements.txt

# 安装 Tesseract OCR 引擎
# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-chi-sim

# macOS
brew install tesseract tesseract-lang
```

#### 2. 运行程序

**Web UI 版本（推荐）**：
```bash
python web_app.py
```
然后在浏览器中访问 `http://localhost:5000`

**命令行版本**：
```bash
# 将截图放在 screenshot/ 文件夹中
python module_ocr_optimizer.py
```

## 📊 结果展示

### Web UI 界面功能

- **📁 文件上传**：支持拖拽上传，批量处理多个截图
- **👁️ 实时预览**：上传后立即显示识别结果
- **📈 可视化结果**：直观的组合方案展示
- **💾 数据导出**：一键导出 JSON 格式结果

### 输出内容

1. **OCR 识别结果**：每个模组的属性和数值
2. **最优组合方案**：按属性最大化分组的最佳组合
3. **详细分析**：包括总收益、模组使用数量、属性详情等
4. **JSON 文件**：完整结果保存在 `optimization_results.json`

### 命令行输出示例

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

### Web UI 结果展示

<div align="center">

| 组合方案 | 总收益 | 使用模组 | 属性详情 |
|---------|--------|----------|----------|
| 方案 1 | 55 点 | 3/4 | 力量+智力 MAX |
| 方案 2 | 52 点 | 4/4 | 均衡发展 |

</div>

## 🎯 支持的属性

<div align="center">

| 攻击属性 | 防御属性 | 专精属性 | 特殊属性 |
|---------|---------|---------|---------|
| 敏捷加持 | 抵御魔法 | 暴击专注 | 极·伤害叠加 |
| 力量加持 | 抵御物理 | 施法专注 | 极·灵活身法 |
| 智力加持 | | 攻速专注 | 极·生命波动 |
| 特攻伤害加持 | | 精英打击 | |
| 特攻治疗加持 | | | |
| 专精治疗加持 | | | |

</div>

## ⚠️ 注意事项

| 项目 | 说明 |
|------|------|
| 🖼️ **图片质量** | 截图需要清晰，属性名称和数值完整可见 |
| 📁 **支持格式** | PNG、JPG、JPEG 格式的图片 |
| 🎯 **识别准确率** | OCR 识别准确率取决于图片质量 |
| 🔧 **错误处理** | 程序会自动过滤识别错误的数据 |
| 💾 **数据安全** | 截图文件夹已加入 .gitignore，保护隐私 |

## 📁 项目结构

```
BPSR_M_OCR/
├── 📄 核心文件
│   ├── module_ocr_optimizer.py    # 主程序（命令行版本）
│   ├── web_app.py                 # Flask Web 应用
│   ├── icon_classifier.py         # 图标分类器
│   └── requirements.txt           # Python 依赖列表
├── 🌐 Web UI 组件
│   ├── templates/
│   │   └── index.html             # HTML 模板
│   └── static/
│       ├── css/style.css          # CSS 样式
│       └── js/app.js              # JavaScript 逻辑
├── 🪟 Windows 部署
│   ├── install_windows.bat        # 自动安装脚本
│   ├── run_ocr.bat                # 命令行启动脚本
│   └── start_webui.bat            # Web UI 启动脚本
├── 📂 数据文件夹
│   ├── screenshot/                # 截图存放目录（已忽略）
│   └── optimization_results.json # 计算结果输出
└── 📚 文档
    ├── README.md                  # 项目说明
    ├── WebUI使用指南.md           # Web UI 使用指南
    └── Windows使用指南.md         # Windows 部署指南
```

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

## 📄 开源协议

本项目基于 MIT 协议开源 - 查看 [LICENSE](LICENSE) 文件了解详情

## 🏷️ 版本历史

- **v1.0.0** - 初始版本
  - ✅ 基础 OCR 识别功能
  - ✅ 组合优化算法
  - ✅ 命令行界面
  - ✅ Web UI 界面
  - ✅ Windows 一键部署

---

<div align="center">

**🎮 专为 BPSR 玩家打造的模组优化工具 🎮**

如果这个项目对你有帮助，请给个 ⭐ Star 支持一下！

</div>

# Windows环境使用指南

## 前置要求

### 1. 安装Python
- 下载并安装 Python 3.8 或更高版本
- 下载地址: https://www.python.org/downloads/
- **重要**: 安装时勾选 "Add Python to PATH"

### 2. 安装Tesseract OCR引擎
- 下载地址: https://github.com/UB-Mannheim/tesseract/wiki
- 选择Windows版本 (推荐: tesseract-ocr-w64-setup-v5.x.x.exe)
- **重要设置**:
  - 安装时勾选 "Chinese (Simplified)" 语言包
  - 记住安装路径 (通常是 `C:\Program Files\Tesseract-OCR`)
  - 将安装路径添加到系统环境变量PATH中

#### 添加环境变量步骤:
1. 右键"此电脑" → "属性"
2. 点击"高级系统设置"
3. 点击"环境变量"
4. 在"系统变量"中找到"Path"，点击"编辑"
5. 点击"新建"，添加: `C:\Program Files\Tesseract-OCR`
6. 确定保存

## 快速安装

1. **运行自动安装脚本**:
   ```cmd
   双击运行 install_windows.bat
   ```

2. **手动安装** (如果自动安装失败):
   ```cmd
   pip install opencv-python==4.8.1.78
   pip install pytesseract==0.3.10
   pip install Pillow==10.0.1
   pip install numpy==1.24.3
   ```

## 使用步骤

### 1. 准备截图
- 将模组截图文件放入 `screenshot` 文件夹
- 支持格式: PNG, JPG, JPEG
- 确保截图清晰，属性名称和数值完整可见

### 2. 运行程序
```cmd
双击运行 run_ocr.bat
```

或者手动运行:
```cmd
python module_ocr_optimizer.py
```

### 3. 查看结果
- 程序会在控制台显示实时结果
- 详细数据保存在 `optimization_results.json` 文件中

## 故障排除

### 问题1: "python不是内部或外部命令"
**解决**: Python未正确安装或未添加到PATH
- 重新安装Python，确保勾选"Add Python to PATH"
- 或手动添加Python安装目录到系统PATH

### 问题2: "tesseract不是内部或外部命令"
**解决**: Tesseract OCR未正确安装或配置
- 重新安装Tesseract OCR
- 确保安装路径已添加到系统PATH
- 重启命令提示符或电脑

### 问题3: OCR识别准确率低
**解决**: 
- 确保截图清晰度足够
- 检查是否安装了中文语言包
- 尝试调整图片亮度/对比度

### 问题4: "ModuleNotFoundError"
**解决**: Python依赖包未安装
```cmd
pip install --upgrade pip
pip install -r requirements.txt
```

### 问题5: 程序运行但无结果
**可能原因**:
- 截图中文字过小或模糊
- 属性名称不在支持列表中
- 数值格式不标准

## 支持的属性列表

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

## 输出文件说明

### console输出
- 实时显示OCR识别进度
- 展示最优组合方案
- 按属性分组显示结果

### optimization_results.json
```json
{
  "modules": {
    "模组名称": {
      "属性名": 数值
    }
  },
  "combinations": [
    {
      "modules": ["模组A", "模组B"],
      "attributes": {"属性名": 数值},
      "total_score": 总分,
      "maxed_attributes": ["最大化属性列表"]
    }
  ],
  "summary": {
    "total_modules": 识别模组数,
    "total_combinations": 有效组合数,
    "max_score": 最高分数
  }
}
```

## 技术参数

- **最大模组数**: 4个
- **属性数值范围**: 1-10点
- **单属性上限**: 20点 (超出无效)
- **垃圾组合阈值**: 低于16点总和
- **支持图片格式**: PNG, JPG, JPEG
- **OCR语言**: 中文简体

## 联系支持

如遇到问题无法解决，请提供:
1. 错误信息截图
2. Python版本 (`python --version`)
3. 操作系统版本
4. 模组截图样例

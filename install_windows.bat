@echo off
chcp 65001
echo ======================================
echo 模组OCR优化器 Windows安装脚本
echo ======================================
echo.

echo 正在检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python环境
    echo 请先安装Python 3.8或更高版本
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Python环境检查通过
echo.

echo 正在安装Python依赖包...
pip install opencv-python==4.8.1.78
pip install pytesseract==0.3.10
pip install Pillow==10.0.1
pip install numpy==1.24.3

echo.
echo 正在检查Tesseract OCR引擎...
tesseract --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo 警告: 未找到Tesseract OCR引擎
    echo 请手动安装Tesseract OCR:
    echo 1. 下载地址: https://github.com/UB-Mannheim/tesseract/wiki
    echo 2. 选择Windows版本下载安装
    echo 3. 安装时记得勾选"中文语言包"
    echo 4. 将安装目录添加到系统PATH环境变量
    echo.
    echo 典型安装路径: C:\Program Files\Tesseract-OCR
    echo 需要添加到PATH: C:\Program Files\Tesseract-OCR
    echo.
    echo 安装完成后重新运行此脚本
    pause
    exit /b 1
)

echo Tesseract OCR引擎检查通过
echo.

echo ======================================
echo 安装完成！
echo ======================================
echo.
echo 使用方法:
echo 1. 将模组截图放入 screenshot 文件夹
echo 2. 双击运行 run_ocr.bat
echo 3. 查看结果文件 optimization_results.json
echo.
pause

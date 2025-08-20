@echo off
chcp 65001
echo ======================================
echo 模组OCR识别和最优组合计算器
echo ======================================
echo.

:: 检查screenshot文件夹是否存在
if not exist "screenshot" (
    echo 创建screenshot文件夹...
    mkdir screenshot
    echo.
    echo 请将模组截图文件放入 screenshot 文件夹中
    echo 支持格式: PNG, JPG, JPEG
    echo.
    pause
    exit /b 1
)

:: 检查是否有图片文件
dir /b screenshot\*.png screenshot\*.jpg screenshot\*.jpeg >nul 2>&1
if errorlevel 1 (
    echo 错误: screenshot文件夹中没有找到图片文件
    echo 请将模组截图放入 screenshot 文件夹
    echo 支持格式: PNG, JPG, JPEG
    echo.
    pause
    exit /b 1
)

echo 开始运行OCR识别和优化计算...
echo.

python module_ocr_optimizer.py

if errorlevel 1 (
    echo.
    echo 程序运行出错，请检查:
    echo 1. Python环境是否正确安装
    echo 2. 所有依赖包是否已安装
    echo 3. Tesseract OCR是否正确配置
    echo 4. 截图文件是否清晰完整
    echo.
) else (
    echo.
    echo ======================================
    echo 计算完成！
    echo ======================================
    echo.
    echo 结果文件已生成:
    echo - optimization_results.json (详细数据)
    echo.
    if exist optimization_results.json (
        echo 打开结果文件？[Y/N]
        set /p choice=
        if /i "%choice%"=="Y" (
            start optimization_results.json
        )
    )
)

echo.
pause

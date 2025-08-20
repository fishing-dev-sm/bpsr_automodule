@echo off
chcp 65001
echo ======================================
echo 模组OCR优化器 Web UI 启动脚本
echo ======================================
echo.

:: 检查Python环境
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python环境
    echo 请先运行 install_windows.bat 安装环境
    pause
    exit /b 1
)

:: 检查Flask是否安装
python -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo 正在安装Web UI依赖...
    pip install Flask==2.3.3 Flask-CORS==4.0.0
)

:: 检查screenshot文件夹
if not exist "screenshot" (
    echo 创建screenshot文件夹...
    mkdir screenshot
    echo.
    echo 请将模组截图文件放入 screenshot 文件夹中
    echo 然后重新运行此脚本
    pause
    exit /b 1
)

:: 检查是否有图片文件
dir /b screenshot\*.png screenshot\*.jpg screenshot\*.jpeg >nul 2>&1
if errorlevel 1 (
    echo 警告: screenshot文件夹中没有找到图片文件
    echo 请确保已将模组截图放入文件夹中
    echo.
)

echo 启动Web UI服务器...
echo.
echo ======================================
echo Web界面将在浏览器中打开
echo 地址: http://localhost:5000
echo 按 Ctrl+C 停止服务器
echo ======================================
echo.

:: 等待2秒后打开浏览器
timeout /t 2 /nobreak >nul
start http://localhost:5000

:: 启动Flask应用
python web_app.py

pause

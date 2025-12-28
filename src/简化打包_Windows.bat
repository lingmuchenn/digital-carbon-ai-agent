@echo off
chcp 65001 >nul
echo ====================================
echo 数字脱碳管家 - Windows 打包脚本
echo ====================================
echo.

REM 清理旧文件
echo [1] 清理旧环境...
if exist build_env rmdir /s /q build_env
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist release rmdir /s /q release
echo 完成
echo.

REM 创建虚拟环境
echo [2] 创建虚拟环境...
python -m venv build_env
if %errorlevel% neq 0 (
    echo 创建虚拟环境失败！
    pause
    exit /b 1
)
echo 完成
echo.

REM 激活虚拟环境
echo [3] 激活虚拟环境...
call build_env\Scripts\activate.bat
echo 完成
echo.

REM 安装依赖
echo [4] 安装依赖包...
pip install --upgrade pip
pip install pyinstaller
pip install flask flask-cors
pip install pywebview
pip install openai
pip install numpy
pip install python-docx PyPDF2 pypdfium2
pip install Pillow
pip install dashscope
pip install tqdm
echo 完成
echo.

REM 打包
echo [5] 开始打包...
python -m PyInstaller --name 数字脱碳管家 --windowed --onefile --add-data "page;page" --add-data "config.json.example;." --hidden-import flask --hidden-import flask_cors --hidden-import webview --hidden-import openai --hidden-import config_loader --hidden-import numpy --hidden-import numpy.core._multiarray_umath --hidden-import docx --hidden-import PyPDF2 --hidden-import pypdfium2 --hidden-import PIL --hidden-import PIL.Image --hidden-import dashscope --hidden-import tqdm --collect-all flask --collect-all numpy --copy-metadata numpy --copy-metadata openai --copy-metadata dashscope run_gui.py

if %errorlevel% neq 0 (
    echo 打包失败！
    call build_env\Scripts\deactivate.bat
    pause
    exit /b 1
)
echo 完成
echo.

REM 创建发布包
echo [6] 创建发布包...
mkdir release 2>nul
copy config.json.example dist\
copy Windows安装说明.txt dist\README.txt
powershell -Command "Compress-Archive -Path 'dist\数字脱碳管家.exe', 'dist\config.json.example', 'dist\README.txt' -DestinationPath 'release\数字脱碳管家_Windows.zip' -Force"
echo 完成
echo.

REM 清理
call build_env\Scripts\deactivate.bat

echo ====================================
echo 打包完成！
echo ====================================
echo.
echo 应用文件: dist\数字脱碳管家.exe
echo 发布包: release\数字脱碳管家_Windows.zip
echo.
pause


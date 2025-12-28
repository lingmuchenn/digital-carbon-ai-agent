#!/bin/bash

echo "===================================================="
echo "数字脱碳管家 - Mac 打包脚本"
echo "===================================================="
echo ""

echo "此脚本将："
echo "1. 创建干净的虚拟环境"
echo "2. 安装必要的依赖"
echo "3. 执行打包"
echo "4. 创建发布包"
echo ""
read -p "按 Enter 继续..."

echo ""
echo "===================================================="
echo "[1/7] 清理旧环境..."
echo "===================================================="
rm -rf build_env
rm -rf build
rm -rf dist
rm -rf release
rm -f *.spec

echo ""
echo "===================================================="
echo "[2/7] 创建虚拟环境..."
echo "===================================================="
python3 -m venv build_env
if [ $? -ne 0 ]; then
    echo "错误: 创建虚拟环境失败！"
    echo "请确保已安装 Python 3.9-3.11"
    exit 1
fi
echo "完成: 虚拟环境创建成功"

echo ""
echo "===================================================="
echo "[3/7] 激活虚拟环境..."
echo "===================================================="
source build_env/bin/activate

echo ""
echo "===================================================="
echo "[4/7] 升级 pip..."
echo "===================================================="
pip install --upgrade pip --quiet

echo ""
echo "===================================================="
echo "[5/7] 安装依赖包（这需要几分钟）..."
echo "===================================================="
echo "正在安装 PyInstaller..."
pip install pyinstaller --quiet
echo "正在安装 Flask..."
pip install flask flask-cors --quiet
echo "正在安装 PyWebView..."
pip install pywebview --quiet
echo "正在安装 OpenAI..."
pip install openai --quiet
echo "正在安装 NumPy..."
pip install numpy --quiet
echo "正在安装文档处理库..."
pip install python-docx PyPDF2 pypdfium2 --quiet
echo "正在安装图像处理库..."
pip install Pillow --quiet
echo "正在安装其他依赖..."
pip install dashscope tqdm --quiet
echo "完成: 依赖安装完成"

echo ""
echo "===================================================="
echo "[6/7] 开始打包（这需要10-20分钟）..."
echo "===================================================="

# 设置临时环境变量
export DASHSCOPE_API_KEY=dummy-key-for-build
export DEEPSEEK_API_KEY=dummy-key-for-build

# 检查图标文件是否存在
ICON_ARG=""
if [ -f "page/image/logo.icns" ]; then
    ICON_ARG="--icon=page/image/logo.icns"
    echo "发现图标文件，将使用自定义图标"
fi

pyinstaller \
  --name "数字脱碳管家" \
  --windowed \
  --onefile \
  $ICON_ARG \
  --add-data "page:page" \
  --add-data "config.json.example:." \
  --hidden-import flask \
  --hidden-import flask_cors \
  --hidden-import webview \
  --hidden-import openai \
  --hidden-import config_loader \
  --hidden-import numpy \
  --hidden-import numpy.core._multiarray_umath \
  --hidden-import docx \
  --hidden-import PyPDF2 \
  --hidden-import pypdfium2 \
  --hidden-import PIL \
  --hidden-import PIL.Image \
  --hidden-import dashscope \
  --hidden-import tqdm \
  --collect-all flask \
  --collect-all numpy \
  --copy-metadata numpy \
  --copy-metadata openai \
  --copy-metadata dashscope \
  --osx-bundle-identifier "com.carbon.manager" \
  run_gui.py

if [ $? -ne 0 ]; then
    echo ""
    echo "错误: 打包失败！"
    deactivate
    exit 1
fi
echo "完成: 打包完成"

echo ""
echo "===================================================="
echo "[7/7] 创建发布包..."
echo "===================================================="
mkdir -p release
cp config.json.example dist/

# 检查安装说明文件
if [ -f "Mac安装说明.txt" ]; then
    cp Mac安装说明.txt dist/README.txt
else
    echo "Mac 安装说明" > dist/README.txt
    echo "=============" >> dist/README.txt
    echo "" >> dist/README.txt
    echo "1. 将 数字脱碳管家.app 拖到应用程序文件夹" >> dist/README.txt
    echo "2. 复制 config.json.example 为 config.json" >> dist/README.txt
    echo "3. 编辑 config.json 填入您的 API 密钥" >> dist/README.txt
    echo "4. 双击运行 数字脱碳管家.app" >> dist/README.txt
fi

# 移除隔离属性
xattr -cr "dist/数字脱碳管家.app" 2>/dev/null

# 创建 ZIP
cd dist
zip -r -q ../release/数字脱碳管家_Mac.zip *
cd ..
echo "完成: 发布包创建完成"

echo ""
echo "===================================================="
echo "[清理] 退出虚拟环境..."
echo "===================================================="
deactivate
echo "完成: 虚拟环境已退出"

echo ""
echo "===================================================="
echo "           打包完成！"
echo "===================================================="
echo ""
echo "应用文件: dist/数字脱碳管家.app"
echo "发布包: release/数字脱碳管家_Mac.zip"
echo ""

if [ -d "dist/数字脱碳管家.app" ]; then
    echo "打包信息:"
    du -sh "dist/数字脱碳管家.app" | awk '{print "   文件大小: "$1}'
    echo ""
fi

echo "===================================================="
echo "下一步:"
echo "===================================================="
echo "1. 测试运行: open dist/数字脱碳管家.app"
echo "2. 配置 config.json（在 dist 目录）"
echo "3. 分发 release/数字脱碳管家_Mac.zip 给用户"
echo ""
echo "提示:"
echo "- 虚拟环境在 build_env/ 目录"
echo "- 可以删除 build_env/ 以节省空间"
echo "- 保留 dist/ 和 release/ 目录"
echo ""
read -p "按 Enter 退出..."


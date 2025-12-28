#!/usr/bin/env python3
"""
应用启动脚本
适用于 Mac 打包成 .app 应用
"""
import sys
import os
from pathlib import Path

# 设置工作目录为脚本所在目录
if getattr(sys, 'frozen', False):
    # 如果是打包后的应用
    application_path = Path(sys._MEIPASS)
else:
    # 如果是直接运行脚本
    application_path = Path(__file__).parent

os.chdir(application_path)
sys.path.insert(0, str(application_path))

# 导入并运行 GUI
from run_gui import main

if __name__ == '__main__':
    main()



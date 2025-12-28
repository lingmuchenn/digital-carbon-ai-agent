import sys
import threading
import time
import webbrowser
import os
from pathlib import Path
import logging

# Try to import pywebview
try:
    import webview
except ImportError:
    webview = None

# Import the Flask app
sys.path.append(str(Path(__file__).parent))
from server import app

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class Api:
    """pywebview JS bridge: provide native folder picker to bypass browser path restrictions."""
    def select_folder(self):
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            try:
                root.attributes('-topmost', True)
            except Exception:
                pass
            path = filedialog.askdirectory()
            try:
                root.destroy()
            except Exception:
                pass
            return path or ""
        except Exception as e:
            try:
                print(f"❌ select_folder 失败: {e}")
            except Exception:
                pass
            return ""

def start_server():
    """Start the Flask server"""
    try:
        print("🔧 正在启动 Flask 服务器...")
        print(f"📁 当前工作目录: {os.getcwd()}")
        print(f"🔑 检查配置文件...")
        
        # 检查 config.json 是否存在
        config_path = Path("config.json")
        if config_path.exists():
            print(f"✅ 找到配置文件: {config_path.absolute()}")
        else:
            print(f"⚠️ 未找到 config.json，将使用环境变量")
            
        app.run(host='127.0.0.1', port=5000, use_reloader=False, debug=False)
    except Exception as e:
        print(f"❌ Flask 服务器启动失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("🚀 启动数字脱碳管家前端...")
    
    # Start server in a daemon thread
    t = threading.Thread(target=start_server)
    t.daemon = True
    t.start()
    
    # Wait for server to start
    time.sleep(1.5)
    
    url = "http://127.0.0.1:5000"
    
    if webview:
        print("📱 使用 pywebview 启动桌面悬浮球...")
        # Create a transparent window
        # frameless=True removes standard title bar
        # transparent=True makes background transparent (requires css support)
        # on_top=True keeps it visible like a floating ball
        window = webview.create_window(
            "数字脱碳管家", 
            url, 
            width=600, 
            height=850, 
            frameless=True, 
            transparent=True,
            on_top=True,
            easy_drag=True,  # Allow dragging by clicking background
            js_api=Api()
        )
        webview.start()
    else:
        print("⚠️ 未检测到 pywebview，将在默认浏览器中打开...")
        print("💡 提示：安装 pywebview 可获得最佳悬浮球体验 (pip install pywebview)")
        webbrowser.open(url)
        
        # Keep the script running since we don't have webview loop
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("🛑 停止服务")

if __name__ == '__main__':
    main()

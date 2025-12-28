"""
配置加载模块
从 config.json 或环境变量中加载 API keys 和其他配置
"""
import os
import json
from pathlib import Path

def load_config():
    """加载配置文件或环境变量"""
    config = {}
    # 兼容：源码运行 / 打包运行（PyInstaller onefile/onedir）
    # 优先级：当前工作目录 > 可执行文件目录 > 模块目录
    candidates = []
    try:
        candidates.append(Path.cwd() / "config.json")
    except Exception:
        pass
    try:
        # PyInstaller: sys.executable 指向 exe；源码运行时也可用
        import sys
        candidates.append(Path(sys.executable).parent / "config.json")
    except Exception:
        pass
    candidates.append(Path(__file__).parent / "config.json")
    config_file = next((p for p in candidates if p.exists()), candidates[-1])
    
    # 尝试从 config.json 加载
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"✅ 成功从 config.json 加载配置")
        except Exception as e:
            print(f"⚠️ 读取 config.json 失败: {e}")
    
    # 环境变量优先级更高
    if os.getenv('DASHSCOPE_API_KEY'):
        config['DASHSCOPE_API_KEY'] = os.getenv('DASHSCOPE_API_KEY')
    if os.getenv('DEEPSEEK_API_KEY'):
        config['DEEPSEEK_API_KEY'] = os.getenv('DEEPSEEK_API_KEY')
    
    # 设置默认值
    config.setdefault('EMBED_MODEL', 'text-embedding-v3')
    config.setdefault('CHAT_MODEL', 'deepseek-chat')
    config.setdefault('SIMILARITY_THRESHOLD', 0.6)
    # 预计时间（秒/文件 & 基础秒数）
    config.setdefault('ESTIMATE_PER_FILE_DUPLICATE', 0.15)
    config.setdefault('ESTIMATE_PER_FILE_SIMILAR', 6.0)
    config.setdefault('ESTIMATE_PER_FILE_IMAGE', 5.0)
    # 相似文件对：LLM 深度分析（秒/对）。该步骤通常耗时显著，需单独计入进度条
    config.setdefault('ESTIMATE_PER_SIMILAR_PAIR_LLM', 3.5)
    config.setdefault('ESTIMATE_BASE_SECONDS', 15)
    
    return config

def get_api_key(key_name):
    """获取指定的 API key"""
    config = load_config()
    return config.get(key_name)



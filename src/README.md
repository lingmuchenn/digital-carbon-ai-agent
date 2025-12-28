# 💻 源码核心实现 (Source Code)

此文件夹包含了“数字脱碳管家”的所有核心源代码、前端交互界面以及工程化打包脚本。开发者可以在此基础上进行本地调试。

## 🛠️ 本地运行指引

1. **环境准备**：
   建议使用 Python 3.11+ 环境。在 `src/` 目录下运行以下命令安装依赖：
   pip install -r requirements.txt
      *注：本项目使用了 `pywebview` 提供桌面窗口体验，若在 Linux 环境下运行可能需要安装额外的系统库。*

2. **获取 API Key**：
   本项目依赖大语言模型及向量模型，请确保您拥有以下平台的 API Key：
   - **阿里云 DashScope** (用于文本嵌入与 Qwen-VL)
   - **DeepSeek** (用于语义差异分析)

3. **启动应用**：
   配置好下方的 `config.json` 后，执行以下命令即可启动：
   
   python run_gui.py
   ## 📂 项目结构与文件说明

```text
.
├── server.py               # 后端核心：提供 Flask API 接口，协调各分析模块
├── find_similar_files.py   # 语义分析模块：负责文档相似度计算、OCR 逻辑及分析流程
├── find_duplicate_by_hash.py # 重复检测模块：基于哈希算法的快速去重逻辑
├── image_analysis.py       # 视觉分析模块：封装 Qwen-VL 与 CV 质量评估算法
├── config_loader.py        # 配置管理：支持从 config.json 或环境变量加载 API Key
├── run_gui.py              # GUI 启动入口：基于 pywebview 的桌面窗口封装
├── run_app.py              # 启动脚本：适配不同环境的工作目录设置
├── config.json.example     # 配置模板：供用户配置 API Key 的参考文件
├── requirements.txt        # 依赖清单：项目运行所需的 Python 库
└── page/                   # 前端资源：HTML/CSS/JS 构建的交互界面
```


## ⚙️ 配置说明

在使用前，需将 `config.json.example` 重命名为 `config.json` 并填写相应的 API Key：

```text
{
    "DASHSCOPE_API_KEY": "你的阿里云通义千问 API Key",
    "DEEPSEEK_API_KEY": "你的 DeepSeek API Key",
    "CHAT_MODEL": "deepseek-chat",
    "EMBED_MODEL": "text-embedding-v4"
}
```

## 📦 打包与分发

项目支持使用 PyInstaller 打包为 Windows (.exe) 或 Mac (.app) 可执行文件：
*   **Windows**: 运行 `简化打包_Windows.bat`。
*   **Mac**: 运行 `简化打包_Mac.sh`。

打包后的程序将自动包含所有 `page/` 下的静态资源。

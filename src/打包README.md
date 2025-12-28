# 数字脱碳管家 - 打包指南

## 📦 快速打包

### Windows 打包

```bash
# 进入目录
cd document_sort

# 执行打包脚本
build_windows.bat
```

打包完成后：
- `dist/数字脱碳管家.exe` - 可执行文件
- `release/数字脱碳管家_Windows.zip` - 分发包

### Mac 打包

```bash
# 进入目录
cd document_sort

# 添加执行权限
chmod +x build_mac.sh

# 执行打包脚本
./build_mac.sh
```

打包完成后：
- `dist/数字脱碳管家.app` - 应用程序
- `release/数字脱碳管家_Mac.zip` - 分发包

## 📋 必要文件

打包前确保这些文件存在：

```
document_sort/
├── run_app.py                 # 应用入口
├── config_loader.py           # 配置加载器
├── server.py                  # Flask 后端
├── find_similar_files.py      # 相似文件分析
├── find_duplicate_by_hash.py  # 重复文件分析  
├── config.json.example        # 配置模板
├── page/                      # 前端文件夹
│   ├── index.html
│   ├── style.css
│   ├── script.js
│   └── image/logo.svg
├── build_windows.bat          # Windows 打包脚本
├── build_mac.sh               # Mac 打包脚本
├── Windows安装说明.txt       # Windows 用户说明
└── Mac安装说明.txt           # Mac 用户说明
```

## 🎯 分发给用户

将 `release/` 目录下的 ZIP 文件分发给用户。

用户只需：
1. 解压 ZIP
2. 配置 `config.json`
3. 双击运行应用

## ⚠️ 注意事项

- 打包前需要安装 PyInstaller: `pip install pyinstaller`
- Mac 需要在 macOS 系统上打包
- Windows 需要在 Windows 系统上打包
- 跨平台打包需要使用虚拟机或 CI/CD

## 🔧 故障排查

### Windows
- 如果提示 `pyinstaller: 未找到命令`，请确保已添加到 PATH
- 杀毒软件可能拦截打包，需要临时关闭

### Mac
- 如果提示权限错误：`chmod +x build_mac.sh`
- 需要 Xcode Command Line Tools: `xcode-select --install`



import os
import sys
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS
import json
import time
import threading
import queue
import datetime
from config_loader import load_config
try:
    from send2trash import send2trash
except Exception:
    send2trash = None

# Ensure we can import modules
sys.path.append(str(Path(__file__).parent))
import find_similar_files
import find_duplicate_by_hash
import find_history_files
import find_process_files

app = Flask(__name__, static_folder='page')

# Queue for log messages
log_queue = queue.Queue()
# 防止同一进程内并发/重复启动分析任务（打包后 UI 误触发时很常见）
analysis_lock = threading.Lock()

def count_files(directory):
    """Recursively count files in directory"""
    count = 0
    try:
        for root, dirs, files in os.walk(directory):
            count += len(files)
    except Exception:
        pass
    return count

def get_folder_size(directory):
    """Get total size of folder in bytes"""
    total_size = 0
    try:
        for root, dirs, files in os.walk(directory):
            for file in files:
                try:
                    file_path = os.path.join(root, file)
                    if os.path.exists(file_path):
                        total_size += os.path.getsize(file_path)
                except Exception:
                    continue
    except Exception:
        pass
    return total_size

@app.route('/')
def index():
    return send_from_directory('page', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('page', path)

@app.route('/api/image')
def serve_image():
    """提供本地图片文件的访问（用于缩略图显示）"""
    image_path = request.args.get('path')
    if not image_path:
        return jsonify({"error": "Path parameter required"}), 400
    
    try:
        file_path = Path(image_path)
        if not file_path.exists() or not file_path.is_file():
            return jsonify({"error": "File not found"}), 404
        
        # 检查是否是图片文件
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tif', '.tiff'}
        if file_path.suffix.lower() not in image_extensions:
            return jsonify({"error": "Not an image file"}), 400
        
        # 返回图片文件
        return send_from_directory(file_path.parent, file_path.name, mimetype=f'image/{file_path.suffix[1:]}')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/get_folder_size', methods=['POST'])
def get_folder_size_api():
    """获取文件夹大小"""
    try:
        data = request.json
        folder_path = data.get('folder_path')
        
        if not folder_path:
            return jsonify({"error": "Folder path is required"}), 400
        
        path_obj = Path(folder_path)
        if not path_obj.exists() or not path_obj.is_dir():
            return jsonify({"error": f"Invalid directory: {folder_path}"}), 400
        
        # 获取文件夹大小
        size = get_folder_size(folder_path)
        
        return jsonify({
            "size": size,
            "path": folder_path
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        print(f"📥 收到分析请求: {data}")
        
        target_path = data.get('path')
        mode = data.get('mode', 'similar')
        
        if not target_path:
            return jsonify({"error": "Path is required"}), 400
        
        path_obj = Path(target_path)
        if not path_obj.exists() or not path_obj.is_dir():
            return jsonify({"error": f"Invalid directory: {target_path}"}), 400
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # 若已有任务在运行，直接拒绝，避免共享全局状态导致第二次运行报错
    if not analysis_lock.acquire(blocking=False):
        return jsonify({"error": "已有分析任务正在运行，请等待完成后再开始新的分析"}), 409
    
    def run_analysis():
        session_logs = []
        start_time = datetime.datetime.now()
        
        def log_callback(message):
            """Capture log, send to queue, and save to memory"""
            # print(message) # Optional console output
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            formatted_msg = f"[{timestamp}] {message}"
            session_logs.append(formatted_msg)
            log_queue.put(message)

        try:
            # 读取配置（支持打包后从 exe 目录/工作目录读取 config.json）
            cfg = load_config()

            # 清理上一轮遗留消息，避免前端收到混杂事件
            try:
                while True:
                    log_queue.get_nowait()
            except queue.Empty:
                pass
            log_queue.put({"type": "reset"})

            log_callback(f"🚀 开始分析: {target_path} (模式: {mode})")
            
            # 1. Estimate time
            log_callback("⏳ 正在扫描文件并计算预估时间...")
            total_files = count_files(target_path)
            
            # Estimate logic（可配置，默认偏保守，尤其图片处理）
            per_file_time = float(cfg.get('ESTIMATE_PER_FILE_DUPLICATE', 0.15))
            if mode == 'similar':
                per_file_time = float(cfg.get('ESTIMATE_PER_FILE_SIMILAR', 6.0))
            elif mode == 'image':
                per_file_time = float(cfg.get('ESTIMATE_PER_FILE_IMAGE', 5.0))
            elif mode == 'history':
                # 历史文件：仅元信息判定 + 层级剪枝，通常非常快
                per_file_time = float(cfg.get('ESTIMATE_PER_FILE_HISTORY', 0.03))
            elif mode == 'process':
                # 过程文件：主要基于文件名/元信息规则，通常很快
                per_file_time = float(cfg.get('ESTIMATE_PER_FILE_PROCESS', 0.06))
                
            estimated_seconds = int(total_files * per_file_time)
            # 添加基础时间（初始化、扫描、模型 warmup 等）
            estimated_seconds += int(cfg.get('ESTIMATE_BASE_SECONDS', 15))
            if estimated_seconds < 5: estimated_seconds = 5
            
            # Send estimate event
            log_queue.put({
                "type": "estimate", 
                "seconds": estimated_seconds,
                "total_files": total_files
            })
            log_callback(f"📄 共发现 {total_files} 个文件，预计耗时 {estimated_seconds} 秒")

            results = []
            if mode == 'duplicate':
                results = find_duplicate_by_hash.process_directory(path_obj, log_callback=log_callback)
            elif mode == 'history':
                results = find_history_files.process_directory(path_obj, log_callback=log_callback)
            elif mode == 'process':
                results = find_process_files.process_directory(path_obj, log_callback=log_callback)
            elif mode == 'image': # New mode for images
                # Directly call the image logic only (reusing find_similar_files with filter?)
                # Actually find_similar_files.process_directory now does BOTH if images exist.
                # To be cleaner, we might want to tell it to ONLY do images.
                # But for now, let's reuse process_directory and rely on its internal logic
                # Maybe we can add a 'mode' param to process_directory later.
                # For now, let's just use it as is, it handles images.
                results = find_similar_files.process_directory(path_obj, log_callback=log_callback)
            else:
                # 'similar' mode
                results = find_similar_files.process_directory(path_obj, log_callback=log_callback)
            
            if results is None:
                results = []
            
            log_callback(f"✅ 分析完成. 找到 {len(results)} 组结果。")
            log_queue.put({"type": "result", "data": results, "mode": mode})
            
            # 3. Save Log to Markdown
            log_dir = path_obj / "logs"
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / f"analysis_log_{start_time.strftime('%Y%m%d_%H%M%S')}.md"
            
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f"# 分析日志 - {mode}\n\n")
                f.write(f"**时间**: {start_time}\n")
                f.write(f"**路径**: {target_path}\n")
                f.write(f"**文件总数**: {total_files}\n")
                f.write(f"**发现结果**: {len(results)} 组\n\n")
                f.write("## 详细日志\n\n")
                for line in session_logs:
                    f.write(f"- {line}\n")
            
            log_callback(f"📝 日志已保存至: {log_file.name}")

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            log_callback(f"❌ 错误: {str(e)}")
            log_callback(f"详细错误: {error_trace}")
            log_queue.put({"type": "error", "message": str(e)})
        finally:
            # 释放锁，允许下一次分析
            try:
                analysis_lock.release()
            except Exception:
                pass

    thread = threading.Thread(target=run_analysis)
    thread.start()
    
    return jsonify({"status": "started", "message": "Analysis started"})


@app.route('/api/delete', methods=['POST'])
def delete_files():
    """将用户勾选的文件移到回收站/废纸篓（前端勾选 → 后端执行）"""
    try:
        data = request.json or {}
        root = data.get('root')
        paths = data.get('paths') or []

        if not root:
            return jsonify({"error": "root is required"}), 400
        if not isinstance(paths, list) or not paths:
            return jsonify({"error": "paths must be a non-empty list"}), 400

        root_path = Path(root).resolve()
        if not root_path.exists() or not root_path.is_dir():
            return jsonify({"error": f"Invalid root directory: {root}"}), 400

        deleted = []
        failed = []

        for p in paths:
            try:
                fp = Path(p).resolve()
                # 安全校验：必须在 root 目录内
                if root_path not in fp.parents and fp != root_path:
                    failed.append({"path": p, "error": "path is outside root"})
                    continue
                if not fp.exists() or not fp.is_file():
                    failed.append({"path": p, "error": "file not found"})
                    continue
                if send2trash is None:
                    failed.append({"path": p, "error": "send2trash not available"})
                    continue
                # 移到回收站/废纸篓（跨平台）
                send2trash(str(fp))
                deleted.append(p)
            except Exception as e:
                failed.append({"path": p, "error": str(e)})

        return jsonify({"deleted": deleted, "failed": failed}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Custom JSON encoder to handle datetime and other objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)

@app.route('/api/stream')
def stream():
    def event_stream():
        while True:
            try:
                message = log_queue.get(timeout=30)
                if isinstance(message, dict):
                    # Use custom encoder
                    yield f"data: {json.dumps(message, cls=CustomJSONEncoder)}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'log', 'message': message}, cls=CustomJSONEncoder)}\n\n"
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'ping'})}\n\n"
    
    return Response(stream_with_context(event_stream()), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(port=5000, debug=True)

import os
import hashlib
from collections import defaultdict
from datetime import datetime


def compute_md5(file_path, block_size=8192):
    """计算文件的 MD5 哈希（分块读取以节省内存）"""
    md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(block_size), b""):
                md5.update(chunk)
        return md5.hexdigest()
    except Exception:
        return None


def get_file_info(file_path):
    """获取文件基本信息（含创建时间）"""
    try:
        stat = os.stat(file_path)
        return {
            "path": file_path,
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "atime": stat.st_atime,
            "ctime": stat.st_ctime,
        }
    except Exception:
        return None


def find_duplicate_by_hash(folder_path):
    """扫描文件夹并找出哈希值相同（内容完全相同）的文件。"""
    file_infos = []
    for root, _, files in os.walk(folder_path):
        for name in files:
            full_path = os.path.join(root, name)
            info = get_file_info(full_path)
            if info:
                file_infos.append(info)

    size_groups = defaultdict(list)
    for info in file_infos:
        size_groups[info["size"]].append(info)

    hash_dict = defaultdict(list)
    for size, group in size_groups.items():
        if len(group) < 2:
            continue
        for item in group:
            md5 = compute_md5(item["path"])
            if md5:
                hash_dict[md5].append(item)

    duplicates = [group for group in hash_dict.values() if len(group) > 1]
    duplicates.sort(key=lambda g: g[0]["size"], reverse=True)
    return duplicates


def suggest_cleanup_for_duplicates(
    duplicates,
    inactive_days=180,
    active_days=30,
    large_file_mb=20
):
    """
    生成清理建议。
    - 组级“需清理”：体积≥large_file_mb 且最近修改时间距今>inactive_days
    - 组内保留：优先最近访问(atime)；若相同用创建时间(ctime)作tiebreaker。
    """
    now = datetime.now()
    results = []

    for idx, group in enumerate(duplicates, start=1):
        for f in group:
            f["size_mb"] = round(f["size"] / (1024 * 1024), 2)
            f["last_modify"] = datetime.fromtimestamp(f["mtime"])
            f["last_access"] = datetime.fromtimestamp(f["atime"])
            f["ctime_dt"] = datetime.fromtimestamp(f.get("ctime", f["mtime"]))
            f["days_since_access"] = (now - f["last_access"]).days
            f["days_since_modify"] = (now - f["last_modify"]).days
            f["is_active"] = f["days_since_access"] <= active_days

        group_size_mb = round(group[0]["size"] / (1024 * 1024), 2)
        newest_modify_dt = max(f["last_modify"] for f in group)
        days_since_newest_modify = (now - newest_modify_dt).days

        group_needs_cleanup = (group_size_mb >= large_file_mb) and (days_since_newest_modify > inactive_days)

        keep = max(group, key=lambda x: (x["last_access"].timestamp(), x["ctime_dt"].timestamp(), x["path"]))

        for f in group:
            if f is keep:
                f["suggestion"] = "✅ 保留"
            else:
                reasons = []
                if not f["is_active"]:
                    reasons.append(f"未访问 {f['days_since_access']} 天")
                if f["size_mb"] >= large_file_mb:
                    reasons.append(f"体积≥{large_file_mb}MB")
                reason_text = ", ".join(reasons) if reasons else "冗余副本"
                f["suggestion"] = f"🗑 强烈删除（{reason_text}）" if group_needs_cleanup else f"🗑 删除（{reason_text}）"

        # 格式化文件对象以匹配前端期望的格式
        formatted_files = []
        for f in group:
            formatted_files.append({
                "path": str(f["path"]),
                "name": os.path.basename(str(f["path"])),
                "size": f["size"],
                "mtime": f["mtime"],
                "suggestion": f["suggestion"]
            })
        
        results.append({
            "type": "duplicate",
            "group_id": idx,
            "fileSize": group_size_mb,
            "file_size_mb": group_size_mb,  # 保留兼容性
            "last_modify": newest_modify_dt.strftime("%Y-%m-%d"),
            "days_since_modify": days_since_newest_modify,
            "need_cleanup": group_needs_cleanup,
            "files": formatted_files,
            "analysis": f"这些文件内容相同，最近修改时间：{newest_modify_dt.strftime('%Y-%m-%d')}"
        })

    return results


def process_directory(target_dir, log_callback=None):
    """
    主处理函数，接收目标目录并返回结果。
    """
    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    folder_path = str(target_dir)
    log(f"🚀 开始扫描重复文件: {folder_path}")
    
    try:
        duplicates = find_duplicate_by_hash(folder_path)
        if not duplicates:
            log("✅ 未发现重复文件")
            return []
            
        log(f"🔍 发现 {len(duplicates)} 组重复文件，正在生成分析报告...")
        
        results = suggest_cleanup_for_duplicates(duplicates, inactive_days=180, active_days=30, large_file_mb=20)
        
        # 简单统计
        total_waste = sum(g['file_size_mb'] * (len(g['files']) - 1) for g in results)
        log(f"📊 预计可释放空间: {total_waste:.2f} MB")
        
        return results
        
    except Exception as e:
        log(f"❌ 发生错误: {str(e)}")
        return []

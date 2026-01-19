import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple


"""
find_history_files.py

扫描用户输入的文件夹，依据文件元信息判定“历史文件”：
1) 超过 N 个月未修改（mtime 距今超过阈值）
   且 创建至今超过 N 个月（ctime 距今超过阈值）
   - 默认 6 个月；阈值可由用户参数修改
   - 注意：Windows 上 st_ctime 通常代表创建时间；Linux/macOS 上可能代表 inode/change time（受系统影响）
2) 文件路径层级 <= max_depth（包含文件名层级），顶层为用户输入目录
   - 例如 max_depth=3：
     - 根目录下 file.txt => 1 层 ✅
     - 子目录/file.txt => 2 层 ✅
     - 子/子/file.txt => 3 层 ✅
     - 子/子/子/file.txt => 4 层 ❌

本文件提供两种使用方式：
- 作为模块：process_directory(target_dir, log_callback=...) -> 返回前端可用的结果列表
- 命令行：python find_history_files.py <target_dir> --inactive-months 6 --age-months 6 --max-depth 3
"""


SECONDS_PER_DAY = 86400


@dataclass(frozen=True)
class HistoryRule:
    inactive_months: int = 6
    age_months: int = 6
    max_depth: int = 3  # 包含文件名层级
    chunk_size: int = 50  # 结果按卡片分组，避免单组过大

    @property
    def inactive_days(self) -> int:
        return max(0, int(self.inactive_months) * 30)

    @property
    def age_days(self) -> int:
        return max(0, int(self.age_months) * 30)


def _safe_print(msg: str) -> None:
    """避免 Windows 控制台编码问题导致打印崩溃。"""
    try:
        print(msg)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        try:
            safe_msg = msg.encode(enc, errors="replace").decode(enc, errors="replace")
        except Exception:
            safe_msg = msg.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        print(safe_msg)


def _format_date(ts: float) -> str:
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
    except Exception:
        return ""


def _iter_files_limited_depth(base_dir: Path, max_depth_including_filename: int) -> Iterable[Path]:
    """
    只遍历 base_dir 下“相对层级（含文件名）<= max_depth”的文件。
    使用 os.walk(topdown=True) 进行目录剪枝，避免深层扫描浪费时间。
    """
    base_dir = base_dir.resolve()
    # 允许的最大“目录深度”（不含文件名）：max_depth-1
    max_dir_depth = max(0, int(max_depth_including_filename) - 1)

    for root, dirs, files in os.walk(str(base_dir), topdown=True):
        try:
            rel_root = Path(root).resolve().relative_to(base_dir)
            dir_depth = 0 if str(rel_root) == "." else len(rel_root.parts)
        except Exception:
            # 无法计算相对路径时，保守处理：不剪枝但仍继续
            dir_depth = 0

        # 目录本身深度已经超过允许范围：直接剪枝
        if dir_depth > max_dir_depth:
            dirs[:] = []
            continue

        # 当前目录刚好在最大目录深度：不再深入子目录
        if dir_depth == max_dir_depth:
            dirs[:] = []

        for name in files:
            path = Path(root) / name
            # 文件层级 = 目录深度 + 1（文件名）
            file_depth = dir_depth + 1
            if file_depth <= max_depth_including_filename:
                yield path


def _is_history_file(st: os.stat_result, now_ts: float, rule: HistoryRule) -> Tuple[bool, Dict[str, int]]:
    """
    历史文件判定：超过 rule.inactive_days 未修改，且创建至今超过 rule.age_days。
    返回：(是否命中, 诊断信息)
    """
    try:
        days_since_modify = int((now_ts - st.st_mtime) / SECONDS_PER_DAY)
    except Exception:
        days_since_modify = 0
    try:
        days_since_create = int((now_ts - st.st_ctime) / SECONDS_PER_DAY)
    except Exception:
        days_since_create = 0

    ok_modify = days_since_modify >= rule.inactive_days
    ok_create = days_since_create >= rule.age_days
    return (ok_modify and ok_create), {
        "days_since_modify": days_since_modify,
        "days_since_create": days_since_create,
    }


def _chunk_list(items: List[Dict], chunk_size: int) -> List[List[Dict]]:
    if chunk_size <= 0:
        return [items]
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def find_history_files(
    target_dir: Path,
    rule: HistoryRule,
    log: Optional[Callable[[str], None]] = None,
) -> List[Dict]:
    """
    扫描并返回历史文件列表（未分组）。
    每个元素为前端可用的 file 对象：{path,name,size,mtime,suggestion,...}
    """
    def _log(msg: str) -> None:
        if log:
            log(msg)
        else:
            _safe_print(msg)

    target_dir = Path(target_dir)
    if not target_dir.exists() or not target_dir.is_dir():
        raise ValueError(f"Invalid directory: {target_dir}")

    now_ts = time.time()
    _log(
        f"🚀 开始扫描历史文件: {str(target_dir)} "
        f"(规则: 未修改≥{rule.inactive_months}个月 & 创建≥{rule.age_months}个月, 最大层级≤{rule.max_depth})"
    )

    matched_files: List[Dict] = []
    scanned = 0

    for path in _iter_files_limited_depth(target_dir, rule.max_depth):
        scanned += 1
        try:
            st = path.stat()
        except OSError:
            continue

        ok, diag = _is_history_file(st, now_ts, rule)
        if not ok:
            continue

        size_bytes = int(getattr(st, "st_size", 0) or 0)
        suggestion = "🗂 建议归档/清理（长期未修改）"
        matched_files.append(
            {
                "path": str(path),
                "name": path.name,
                "size": size_bytes,
                "mtime": float(getattr(st, "st_mtime", 0.0) or 0.0),
                "suggestion": suggestion,
                # 便于调试/解释（前端不一定会展示）
                "days_since_modify": diag["days_since_modify"],
                "days_since_create": diag["days_since_create"],
            }
        )

    matched_files.sort(key=lambda x: (x.get("mtime", 0.0), x.get("path", "")))
    _log(f"📄 扫描文件数（受层级限制）: {scanned}")
    _log(f"✅ 命中历史文件: {len(matched_files)}")
    return matched_files


def process_directory(target_dir, log_callback=None, rule: Optional[HistoryRule] = None) -> List[Dict]:
    """
    与其它模块保持一致的入口：接收目标目录并返回结果列表。
    返回结构设计为“组卡片”形式，方便前端像重复文件一样渲染。
    """
    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            _safe_print(msg)

    try:
        rule = rule or HistoryRule()
        folder_path = Path(target_dir)
        files = find_history_files(folder_path, rule=rule, log=log)
        if not files:
            log("✅ 未发现历史文件")
            return []

        chunks = _chunk_list(files, rule.chunk_size)
        results: List[Dict] = []
        for idx, chunk in enumerate(chunks, start=1):
            total_mb = round(sum(f.get("size", 0) for f in chunk) / (1024 * 1024), 2)
            newest_mtime = max((f.get("mtime", 0.0) for f in chunk), default=0.0)
            oldest_mtime = min((f.get("mtime", 0.0) for f in chunk), default=0.0)

            results.append(
                {
                    "type": "history",
                    "group_id": idx,
                    "fileSize": total_mb,
                    "file_size_mb": total_mb,  # 兼容 duplicate 的字段命名
                    "need_cleanup": True,
                    "files": chunk,
                    "analysis": (
                        f"命中规则：未修改≥{rule.inactive_months}个月 且 创建≥{rule.age_months}个月，"
                        f"路径层级≤{rule.max_depth}。"
                        f"本组共 {len(chunk)} 个文件，修改时间范围：{_format_date(oldest_mtime)} ~ {_format_date(newest_mtime)}。"
                    ),
                    "criteria": {
                        "inactive_months": rule.inactive_months,
                        "age_months": rule.age_months,
                        "max_depth": rule.max_depth,
                        "chunk_size": rule.chunk_size,
                    },
                }
            )

        log(f"📦 已生成 {len(results)} 组历史文件结果（每组≤{rule.chunk_size}个）")
        return results

    except Exception as e:
        log(f"❌ 发生错误: {str(e)}")
        return []


def parse_args():
    parser = argparse.ArgumentParser(description="查找文件夹内的历史文件（基于元信息 + 层级限制）")
    parser.add_argument("target_dir", type=str, help="要扫描的目录")
    parser.add_argument("--inactive-months", type=int, default=6, help="未修改阈值（月），默认 6")
    parser.add_argument("--age-months", type=int, default=6, help="创建至今阈值（月），默认 6")
    parser.add_argument("--max-depth", type=int, default=3, help="最大路径层级（含文件名），默认 3")
    parser.add_argument("--chunk-size", type=int, default=50, help="输出分组每组最多文件数，默认 50")
    return parser.parse_args()


def main():
    args = parse_args()
    target_dir = Path(args.target_dir)
    if not target_dir.exists():
        _safe_print(f"❌ 目录不存在：{target_dir}")
        sys.exit(1)

    rule = HistoryRule(
        inactive_months=args.inactive_months,
        age_months=args.age_months,
        max_depth=args.max_depth,
        chunk_size=args.chunk_size,
    )
    results = process_directory(target_dir, rule=rule)
    _safe_print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


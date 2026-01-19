import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# =========================
# 基础配置
# =========================
ARCHIVE_EXTS = {".zip", ".rar", ".7z"}
INSTALLER_EXTS = {".dmg", ".pkg", ".exe", ".msi", ".apk"}
EDITABLE_SOURCE_EXTS = {".docx", ".doc", ".pptx", ".ppt", ".xls", ".xlsx", ".png", ".jpg"}
OPEN_WITHIN_MINUTES = 30


@dataclass
class FileInfo:
    path: Path
    name: str
    stem: str
    ext: str
    parent: Path
    size: int
    ctime: float
    mtime: float
    atime: float


def _safe_print(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        try:
            safe_msg = msg.encode(enc, errors="replace").decode(enc, errors="replace")
        except Exception:
            safe_msg = msg.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        print(safe_msg)


def get_default_download_dirs() -> List[Path]:
    """根据系统推断默认下载目录。"""
    candidates = []
    home = Path.home()
    candidates.append(home / "Downloads")
    candidates.append(home / "下载")

    userprofile = os.environ.get("USERPROFILE")
    if userprofile:
        candidates.append(Path(userprofile) / "Downloads")
        candidates.append(Path(userprofile) / "下载")

    # 只保留存在的路径
    return [p for p in candidates if p.exists()]


def is_under_path(path: Path, root: Path) -> bool:
    try:
        return path.resolve().is_relative_to(root.resolve())
    except Exception:
        # 兼容老版本写法
        try:
            path.resolve().relative_to(root.resolve())
            return True
        except Exception:
            return False


def normalize_name(name: str) -> str:
    """去掉符号并统一大小写，用于名称相似度判断。"""
    name = name.lower()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", name)


_DATE_PATTERNS = [
    # 20250115 / 2025-01-15 / 2025_1_5 / 2025.01.15
    re.compile(r"20\d{2}[-_\.]?\d{1,2}[-_\.]?\d{1,2}"),
    # 2025Q1 / 2025q4
    re.compile(r"20\d{2}\s*[qQ]\s*[1-4]"),
]
_VERSION_PATTERNS = [
    # v1 / v1.2 / v2025.01
    re.compile(r"^v\d+([._-]\d+)*$", re.IGNORECASE),
    re.compile(r"^(ver|version|rev|release)\d*([._-]\d+)*$", re.IGNORECASE),
    # 1.2.3 / 2025.01.15 等纯数字点分
    re.compile(r"^\d+(\.\d+){1,4}$"),
]
_NOISE_TOKENS = {
    # 英文常见噪声
    "final", "draft", "copy", "temp", "tmp", "new", "untitled",
    "export", "output", "converted", "scan", "scanned",
    # 中文常见噪声
    # 注意：不要把可能具有语义的词（如“归档”）当作噪声剔除，否则会导致相似匹配失败
    "副本", "拷贝", "备份", "最终版", "终版", "草稿", "定稿", "新建", "未命名",
}


def _split_tokens(name: str) -> List[str]:
    """
    将文件名拆成 token：兼容中英文、数字、下划线/空格/括号等，以及 camelCase。
    """
    s = name.strip()
    # camelCase 边界插空格
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    # 统一分隔符为空格
    s = re.sub(r"[^\w\u4e00-\u9fff]+", " ", s, flags=re.UNICODE)
    raw = [t for t in s.split() if t]

    # 进一步拆分“中文/英文 + 数字”粘连的 token，例如：归档16、report2025
    tokens: List[str] = []
    for tok in raw:
        parts = re.findall(r"[A-Za-z\u4e00-\u9fff]+|\d+", tok, flags=re.UNICODE)
        tokens.extend(parts if parts else [tok])
    return tokens


def _is_date_like(token: str) -> bool:
    t = token.strip()
    for pat in _DATE_PATTERNS:
        if pat.fullmatch(t) or pat.search(t):
            return True
    return False


def _is_version_like(token: str) -> bool:
    t = token.strip()
    for pat in _VERSION_PATTERNS:
        if pat.fullmatch(t):
            return True
    return False


def _clean_tokens_for_similarity(name: str) -> List[str]:
    tokens = _split_tokens(name)
    cleaned: List[str] = []
    for tok in tokens:
        t = tok.strip().lower()
        if not t:
            continue
        # 去括号包裹的纯数字：(1) / [2]
        if re.fullmatch(r"[\(\[\{]?\d+[\)\]\}]?", t):
            continue
        if _is_date_like(t):
            continue
        if _is_version_like(t):
            continue
        # 回退到上一个版本：纯数字按噪声处理（用于更强的“语义名 + 时间尾巴”匹配）
        if t.isdigit():
            continue
        # 中文 token 也可能是“副本”等
        if t in _NOISE_TOKENS:
            continue
        # 诸如 x64/arm64/win64 这类平台信息通常是噪声
        if re.fullmatch(r"(x64|x86|arm64|amd64|win\d*|mac|linux)", t):
            continue
        cleaned.append(t)
    return cleaned


def _core_string(tokens: List[str]) -> str:
    """
    将 token 合成“核心串”，用于编辑距离；中文/英文统一小写，移除符号。
    """
    if not tokens:
        return ""
    joined = " ".join(tokens)
    return normalize_name(joined)


def name_similarity(a: str, b: str) -> float:
    """
    优化后的文件名相似度：
    - 先移除日期/版本/副本等噪声 token
    - 再综合 token Jaccard + 核心串编辑距离
    """
    ta = _clean_tokens_for_similarity(a)
    tb = _clean_tokens_for_similarity(b)

    ca = _core_string(ta) or normalize_name(a)
    cb = _core_string(tb) or normalize_name(b)
    if not ca or not cb:
        return 0.0

    seq = SequenceMatcher(None, ca, cb).ratio()

    set_a = set(ta)
    set_b = set(tb)
    if not set_a or not set_b:
        token_j = 0.0
    else:
        inter = len(set_a & set_b)
        union = len(set_a | set_b)
        token_j = inter / union if union else 0.0

    # 子串加成：比如“报告” vs “报告更新”
    substr_bonus = 0.0
    if ca in cb or cb in ca:
        substr_bonus = 0.05

    score = 0.6 * seq + 0.4 * token_j + substr_bonus
    return min(1.0, max(0.0, score))


def is_recently_accessed(atime: float, days: int) -> bool:
    return (time.time() - atime) <= days * 86400


def is_recently_accessed_hours(atime: float, hours: float) -> bool:
    return (time.time() - atime) <= hours * 3600


def last_activity_ts_by_mtime_ctime(ctime: float, mtime: float) -> float:
    """
    过程文件判定不再依赖 atime（访问时间）。
    统一用 max(mtime, ctime) 作为“最近活动时间”。
    """
    return max(ctime, mtime)


def is_inactive_for_days_by_mtime_ctime(ctime: float, mtime: float, days: float) -> bool:
    """今天 - 修改日期/创建日期 > N天（用 max(mtime, ctime) 计算）"""
    ts = last_activity_ts_by_mtime_ctime(ctime, mtime)
    return (time.time() - ts) > days * 86400


def used_soon_after_create_by_mtime_ctime(ctime: float, mtime: float, minutes: int) -> bool:
    """
    下载类过程文件的“创建后 ≤30分钟内被使用”：
    - (修改日期 - 创建日期) ≤ 30分钟
      OR
    - 创建日期 > 修改日期（时间顺序异常/拷贝行为等）
    """
    if ctime > mtime:
        return True
    return (mtime - ctime) <= minutes * 60


def opened_within_minutes(ctime: float, atime: float, minutes: int) -> bool:
    if atime < ctime:
        return False
    return (atime - ctime) <= minutes * 60


def recently_used_within_minutes(ctime: float, atime: float, mtime: float, minutes: int) -> bool:
    """创建后短时间内被使用（访问或修改）"""
    # “被使用”的定义：访问时间或修改时间任一满足即可
    threshold = minutes * 60
    ok_access = atime >= ctime and (atime - ctime) <= threshold
    ok_modify = mtime >= ctime and (mtime - ctime) <= threshold
    return ok_access or ok_modify


def list_editable_sources_nearby(file_path: Path, max_siblings: int = 5) -> List[Path]:
    """在同目录或相邻目录寻找可编辑源文件。"""
    candidates = []
    parent = file_path.parent

    # 同目录
    for p in parent.glob("*"):
        if p.is_file() and p.suffix.lower() in EDITABLE_SOURCE_EXTS:
            candidates.append(p)

    # 父目录下的相邻目录（限制数量避免扫描太大）
    parent_parent = parent.parent
    if parent_parent.exists():
        siblings = [d for d in parent_parent.iterdir() if d.is_dir()]
        for d in siblings[:max_siblings]:
            for p in d.glob("*"):
                if p.is_file() and p.suffix.lower() in EDITABLE_SOURCE_EXTS:
                    candidates.append(p)

    return candidates


def find_similar_named_folder(file_path: Path) -> Tuple[Optional[Path], float]:
    """在同一父目录下查找同名/相似文件夹。"""
    parent = file_path.parent
    best_match = None
    best_score = 0.0
    for p in parent.iterdir():
        if p.is_dir():
            score = name_similarity(p.name, file_path.stem)
            if score > best_score:
                best_score = score
                best_match = p
    return best_match, best_score


def build_file_info(path: Path) -> FileInfo:
    st = path.stat()
    return FileInfo(
        path=path,
        name=path.name,
        stem=path.stem,
        ext=path.suffix.lower(),
        parent=path.parent,
        size=st.st_size,
        ctime=st.st_ctime,
        mtime=st.st_mtime,
        atime=st.st_atime,
    )


def evaluate_archive_file(
    info: FileInfo,
) -> Optional[Dict]:
    if info.ext not in ARCHIVE_EXTS:
        return None

    folder_match, score = find_similar_named_folder(info.path)
    has_extract_folder = bool(folder_match and score >= 0.9)
    if not has_extract_folder:
        return None

    # 最后条件：大于 1 天未访问（今天 - 修改日期/创建日期 > 1天）
    if not is_inactive_for_days_by_mtime_ctime(info.ctime, info.mtime, 1):
        return None

    return {
        "category": "archive_container",
        "path": str(info.path),
        "evidence": {
            "extract_folder": str(folder_match) if folder_match else "",
            "folder_match_score": round(score, 3),
        },
    }


def evaluate_installer_file(
    info: FileInfo,
) -> Optional[Dict]:
    if info.ext not in INSTALLER_EXTS:
        return None

    # 最后条件：大于 1 天未访问（今天 - 修改日期/创建日期 > 1天）
    if not is_inactive_for_days_by_mtime_ctime(info.ctime, info.mtime, 1):
        return None

    return {
        "category": "installer",
        "path": str(info.path),
    }


def evaluate_document_export(
    info: FileInfo,
) -> Optional[Dict]:
    if info.ext != ".pdf":
        return None

    # 必要条件：存在相同或相似名称的可编辑源文件
    similar_source = None
    similar_score = 0.0
    sources = list_editable_sources_nearby(info.path)
    for src in sources:
        score = name_similarity(info.stem, src.stem)
        if score > similar_score:
            similar_score = score
            similar_source = src

    if similar_score < 0.9:
        return None

    # 最后条件：大于 3 天未访问（今天 - 修改日期/创建日期 > 3天）
    if not is_inactive_for_days_by_mtime_ctime(info.ctime, info.mtime, 3):
        return None

    return {
        "category": "document_export",
        "path": str(info.path),
        "evidence": {
            "similar_source": str(similar_source) if similar_source else "",
            "name_similarity": round(similar_score, 3)
            },
    }


def evaluate_single_use_download(
    info: FileInfo,
) -> Optional[Dict]:
    if info.ext in ARCHIVE_EXTS or info.ext in INSTALLER_EXTS:
        return None

    # 必要条件 1：创建后 ≤ 30 分钟内被使用（基于 mtime/ctime）
    if not used_soon_after_create_by_mtime_ctime(info.ctime, info.mtime, OPEN_WITHIN_MINUTES):
        return None

    # 必要条件 2：大于 3 天未访问（今天 - 修改日期/创建日期 > 3天）
    if not is_inactive_for_days_by_mtime_ctime(info.ctime, info.mtime, 1):
        return None

    return {
        "category": "downloads",
        "path": str(info.path),
    }


def scan_files(target_dir: Path, recursive: bool = False) -> List[FileInfo]:
    if recursive:
        files = [p for p in target_dir.rglob("*") if p.is_file()]
    else:
        files = [p for p in target_dir.iterdir() if p.is_file()]
    return [build_file_info(p) for p in files]


def find_process_files(
    target_dir: Path,
    recursive: bool = False,
    debug: bool = False,
) -> List[Dict]:
    results = []
    infos = scan_files(target_dir, recursive=recursive)
    if debug:
        _safe_print(f"[DEBUG] 目标目录: {target_dir.resolve()}")
        _safe_print(f"[DEBUG] 递归扫描: {'是' if recursive else '否'}")
        _safe_print(f"[DEBUG] 扫描到文件数: {len(infos)}")
        if len(infos) == 0:
            _safe_print("[DEBUG] 未扫描到任何文件（可能目录为空，或只有子文件夹且无文件）")

    for info in infos:
        if debug:
            now = time.time()
            cdt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(info.ctime))
            mdt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(info.mtime))
            adt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(info.atime))
            dm_min = (info.mtime - info.ctime) / 60.0
            last_ts = last_activity_ts_by_mtime_ctime(info.ctime, info.mtime)
            last_dt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_ts))
            inactive_days = (now - last_ts) / 86400.0
            _safe_print(
                f"\n[DEBUG] {info.path}\n"
                f"  ctime={cdt}  mtime={mdt}  atime={adt}\n"
                f"  last_activity(ctime/mtime)={last_dt}  inactive={inactive_days:.2f}d  (mtime-ctime)={dm_min:.2f}min"
            )

        matched = False
        # 1) 压缩包类
        res = evaluate_archive_file(info)
        if res:
            results.append(res)
            matched = True
        elif debug and info.ext in ARCHIVE_EXTS:
            folder_match, score = find_similar_named_folder(info.path)
            _safe_print(
                f"  [archive_container] folder_match_score={score:.3f} "
                f"has_folder={'Y' if (folder_match and score >= 0.9) else 'N'} "
                f"inactive_>1d={'Y' if is_inactive_for_days_by_mtime_ctime(info.ctime, info.mtime, 1) else 'N'}"
            )

        if matched:
            continue

        # 2) 安装包类
        res = evaluate_installer_file(info)
        if res:
            results.append(res)
            matched = True
        elif debug and info.ext in INSTALLER_EXTS:
            _safe_print(
                f"  [installer] inactive_>1d={'Y' if is_inactive_for_days_by_mtime_ctime(info.ctime, info.mtime, 1) else 'N'}"
            )

        if matched:
            continue

        # 3) 文档导出类
        res = evaluate_document_export(info)
        if res:
            results.append(res)
            matched = True
        elif debug and info.ext == ".pdf":
            similar_source = None
            similar_score = 0.0
            sources = list_editable_sources_nearby(info.path)
            for src in sources:
                score = name_similarity(info.stem, src.stem)
                if score > similar_score:
                    similar_score = score
                    similar_source = src
            _safe_print(
                f"  [document_export] best_name_similarity={similar_score:.3f} "
                f"has_source={'Y' if similar_source else 'N'} "
                f"inactive_>3d={'Y' if is_inactive_for_days_by_mtime_ctime(info.ctime, info.mtime, 3) else 'N'} "
                f"best_source={similar_source}"
            )

        if matched:
            continue

        # 4) 下载即用类
        res = evaluate_single_use_download(info)
        if res:
            results.append(res)
            matched = True
        elif debug:
            used_30m = used_soon_after_create_by_mtime_ctime(info.ctime, info.mtime, OPEN_WITHIN_MINUTES)
            inactive_3d = is_inactive_for_days_by_mtime_ctime(info.ctime, info.mtime, 3)
            _safe_print(
                f"  [downloads] used_within_30m={'Y' if used_30m else 'N'} "
                f"inactive_>3d={'Y' if inactive_3d else 'N'}"
            )
    return results


def process_directory(target_dir, log_callback=None, recursive: bool = True, debug: bool = False) -> List[Dict]:
    """
    与其它模块保持一致的入口：接收目标目录并返回“组卡片”结果列表，供前端渲染。

    输出结构（每组）：
    - type: "process"
    - group_id
    - label: 分类名称
    - file_size_mb / fileSize
    - need_cleanup
    - files: [{path,name,size,mtime,suggestion,category,evidence}]
    - analysis: 解释文本
    """
    def log(msg: str):
        if log_callback:
            log_callback(msg)
        else:
            _safe_print(msg)

    base = Path(target_dir)
    if not base.exists() or not base.is_dir():
        log(f"❌ 目录不存在或不可访问：{base}")
        return []

    CATEGORY_LABEL = {
        "archive_container": "压缩包（已解压可删除压缩包）",
        "installer": "安装包（安装后可删除）",
        "document_export": "导出文件（有源文件可删除导出件）",
        "downloads": "下载即用（短期用过、近期未再使用）",
    }
    CATEGORY_SUGGESTION = {
        "archive_container": "🗑 建议删除（已存在解压文件夹）",
        "installer": "🗑 建议删除（安装包）",
        "document_export": "🗑 建议删除（存在可编辑源文件）",
        "downloads": "🗑 建议清理（下载即用）",
    }

    log(f"🚀 开始扫描过程文件: {str(base)} (递归: {'是' if recursive else '否'})")
    raw = find_process_files(base, recursive=recursive, debug=debug)
    if not raw:
        log("✅ 未发现过程文件")
        return []

    # 补齐前端需要的 file 字段
    groups: Dict[str, List[Dict]] = {}
    for item in raw:
        category = item.get("category") or "process"
        path_str = item.get("path") or ""
        if not path_str:
            continue

        p = Path(path_str)
        try:
            st = p.stat()
            size = int(getattr(st, "st_size", 0) or 0)
            mtime = float(getattr(st, "st_mtime", 0.0) or 0.0)
        except OSError:
            size = 0
            mtime = 0.0

        file_obj = {
            "path": str(p),
            "name": p.name,
            "size": size,
            "mtime": mtime,
            "suggestion": CATEGORY_SUGGESTION.get(category, "🗑 建议清理（过程文件）"),
            "category": category,
            "evidence": item.get("evidence", {}),
        }
        groups.setdefault(category, []).append(file_obj)

    # 固定输出顺序（更符合用户理解）
    ordered_categories = ["archive_container", "installer", "document_export", "downloads"]
    # 补充未知分类
    for c in groups.keys():
        if c not in ordered_categories:
            ordered_categories.append(c)

    results: List[Dict] = []
    gid = 1
    for category in ordered_categories:
        files = groups.get(category)
        if not files:
            continue
        total_mb = round(sum(f.get("size", 0) for f in files) / (1024 * 1024), 2)
        label = CATEGORY_LABEL.get(category, "过程文件")
        results.append(
            {
                "type": "process",
                "group_id": gid,
                "label": label,
                "fileSize": total_mb,
                "file_size_mb": total_mb,
                "need_cleanup": True,
                "files": files,
                "analysis": f"识别到 {len(files)} 个“{label}”类型的过程文件，建议按需清理或归档。",
                "criteria": {
                    "recursive": bool(recursive),
                    "category": category,
                },
            }
        )
        gid += 1

    log(f"✅ 过程文件扫描完成：{sum(len(v) for v in groups.values())} 个文件，{len(results)} 组")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="查找文件夹内的过程文件")
    parser.add_argument("target_dir", type=str, help="要扫描的目录")
    parser.add_argument("--recursive", action="store_true", help="递归扫描子文件夹（默认不递归）")
    parser.add_argument("--debug", action="store_true", help="输出每个文件的判定细节（用于排查未命中原因）")
    return parser.parse_args()


def main():
    args = parse_args()
    target_dir = Path(args.target_dir)
    if not target_dir.exists():
        _safe_print(f"❌ 目录不存在：{target_dir}")
        sys.exit(1)

    results = find_process_files(
        target_dir=target_dir,
        recursive=args.recursive,
        debug=args.debug,
    )

    _safe_print(f"✅ 共发现 {len(results)} 个过程文件")
    _safe_print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""
image_analysis.py

图像理解与整理模块（独立模块，按你要求的最优架构）：
- EXIF 初筛（判断是否为截图 / 照片）
- 基于 Qwen-VL（通义千问）生成 JSON 标签与描述
- 计算传统 CV 质量分（清晰度/亮度/曝光/噪声）用于 BestShot
- 将图像描述转换为 embedding（供聚类/相似度匹配）
- 截图类图片可调用 DeepSeek（或其他 LLM）做高阶分类与整理建议

设计目标：把核心函数做成独立、可复用、无全局副作用的接口，便于在你已有的 find_similar_files.py 中 import 并使用。

依赖（可选）：Pillow, numpy, requests(or openai client), sklearn, piexif, cv2
注：模块对缺失库做了优雅回退；实际部署建议安装： pillow numpy scikit-learn opencv-python piexif

用法示例：
    from image_analysis import ImageAnalyzer
    analyzer = ImageAnalyzer(emb_client=emb_client, vl_model='qwen3-vl-plus', llm_client=chat_client)
    info = analyzer.analyze_image_file('/path/to/img.jpg')

返回：一个 dict，包含 exif, type, vl_json, quality_scores, embedding, bestshot_score, suggestion 等字段。

"""

from __future__ import annotations
import io
import json
import math
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

# Optional heavy deps
try:
    from PIL import Image, ExifTags, ImageStat
except Exception:
    Image = None

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

# clustering
try:
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    KMeans = None
    def cosine_similarity(a,b):
        a = np.atleast_2d(a)
        b = np.atleast_2d(b)
        denom = (np.linalg.norm(a, axis=1, keepdims=True) * np.linalg.norm(b, axis=1, keepdims=True).T) + 1e-10
        return (a @ b.T) / denom

# EXIF parsing helper
try:
    import piexif
except Exception:
    piexif = None

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def safe_open_image(path: str):
    if Image is None:
        raise RuntimeError('Pillow not installed')
    return Image.open(path)


def _clean_exif_value(val):
    """Helper to make EXIF values JSON serializable"""
    if isinstance(val, bytes):
        if len(val) > 50:
            return f"<bytes len={len(val)}>"
        try:
            return val.decode('utf-8', errors='ignore').strip('\x00')
        except:
            return str(val)
    if isinstance(val, str):
        return val.strip('\x00')
    if hasattr(val, 'numerator') and hasattr(val, 'denominator'):
        return float(val)
    # Handle tuples/lists recursively
    if isinstance(val, (tuple, list)):
        return [_clean_exif_value(x) for x in val]
    if isinstance(val, dict):
        return {str(k): _clean_exif_value(v) for k, v in val.items()}
    # Check for other PIL specific types if necessary, or just rely on str() for unknowns
    try:
        json.dumps(val)
        return val
    except (TypeError, OverflowError):
        return str(val)

def read_exif(path: str) -> Dict[str, Any]:
    """读取图片 EXIF 元数据，返回 dict（对常用字段做友好化）"""
    exif_data = {}
    if Image is None:
        return exif_data
    try:
        with Image.open(path) as img:
            exif = img._getexif() if hasattr(img, '_getexif') else None
            if exif:
                for k, v in exif.items():
                    tag = ExifTags.TAGS.get(k, k)
                    exif_data[str(tag)] = _clean_exif_value(v)
            # piexif 可读更多结构化信息
            if piexif is not None:
                try:
                    # piexif load needs path, not image object usually
                    raw = piexif.load(path)
                    # 同样清理 piexif 数据
                    if raw:
                         exif_data['piexif'] = _clean_exif_value({k: dict(v) for k, v in raw.items() if v})
                except Exception:
                    pass
    except Exception:
        pass
    return exif_data


def is_likely_screenshot_basic(path: str, exif: Dict[str, Any]) -> bool:
    """基于扩展名、EXIF、文件名做初步判断（在 VL 分析前）。
    
    这是初步筛选，返回：
    - True: 很可能是截图（文件名明确、PNG无EXIF等）
    - False: 需要进一步判断（由 VL 模型辅助）
    
    注意：此函数会被 VL 模型结果覆盖，所以采用保守策略。
    """
    p = Path(path)
    ext = p.suffix.lower()
    name = p.stem.lower()

    # 强信号：如果 EXIF 包含相机信息，肯定是照片
    cam_keys = ['Make', 'Model', 'LensModel']
    if any(k in exif for k in cam_keys):
        return False

    # 强信号：文件名明确包含截图关键词
    screenshot_keywords = ['screenshot', 'screen', '截屏', '截图', 'screencapture']
    if any(keyword in name for keyword in screenshot_keywords):
        return True

    # 中等信号：PNG 且无 EXIF，但不确定（很多照片经过处理后也会变成这样）
    # 所以这里只作为弱信号，等待 VL 模型判断
    if ext == '.png' and not exif:
        return None  # 不确定，交给 VL 模型判断
    
    # 默认倾向认为是照片（因为 JPG/JPEG 更常见于照片）
    return False


# ---------------- CV Quality Metrics ------------------------------------------------

def variance_of_laplacian_gray(img_gray: np.ndarray) -> float:
    """使用 OpenCV 计算 Laplacian 方差作为模糊度指标。若无 cv2，则用 FFT 高频能量近似。"""
    if cv2 is not None:
        lap = cv2.Laplacian(img_gray, cv2.CV_64F)
        return float(lap.var())
    # fallback: FFT high-frequency energy
    try:
        f = np.fft.fft2(img_gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.abs(fshift)
        h, w = img_gray.shape
        # take corners as high-frequency proxy
        pad = max(4, int(min(h,w)*0.1))
        corners = magnitude_spectrum[:pad,:pad].sum() + magnitude_spectrum[-pad:,-pad:].sum()
        return float(corners / (h*w))
    except Exception:
        return 0.0


def brightness_score(img: Image.Image) -> float:
    """返回 0-1 的亮度分数（0 太暗，1 太亮/正常）这里仅返回基于均值的归一化得分。"""
    if Image is None:
        return 0.5
    try:
        s = ImageStat.Stat(img)
        # 对RGB取平均亮度
        if len(s.mean) >= 3:
            mean = float(sum(s.mean[:3]) / 3.0)
        else:
            mean = float(s.mean[0])
        # 0-255 映射到 0-1，但理想亮度约在 100-180
        # 将 0->0, 255->1 并给中间区域较高score
        return max(0.0, min(1.0, mean / 255.0))
    except Exception:
        return 0.5


def exposure_score(img: Image.Image) -> float:
    """简单用直方图分布判断过曝/欠曝，返回 0-1 代表曝光合理度（1最佳）。"""
    try:
        gray = img.convert('L')
        arr = np.array(gray).flatten()
        # fraction of pixels in very dark and very bright
        dark_frac = np.mean(arr < 10)
        bright_frac = np.mean(arr > 245)
        bad = dark_frac + bright_frac
        score = max(0.0, 1.0 - bad*10)  # 放大惩罚
        return float(score)
    except Exception:
        return 0.5


def compute_quality_scores(path: str) -> Dict[str, float]:
    """对图片做一组传统 CV 质量度量，返回字典。
    包含: blur_var, brightness, exposure, noise_est
    """
    out = {
        'blur_var': 0.0,
        'brightness': 0.5,
        'exposure': 0.5,
        'noise': 0.0
    }
    if Image is None:
        return out
    try:
        with Image.open(path) as im:
            # convert to gray numpy
            gray = im.convert('L')
            # keep as uint8 for cv2 stability
            arr_u8 = np.array(gray)
            arr_f32 = arr_u8.astype(np.float32)
            
            out['blur_var'] = variance_of_laplacian_gray(arr_u8)
            out['brightness'] = brightness_score(im)
            out['exposure'] = exposure_score(im)
            # noise estimate: local std deviation mean
            try:
                from scipy.ndimage import uniform_filter
                # use float for noise calc
                mean = uniform_filter(arr_f32, size=3)
                sq_mean = uniform_filter(arr_f32*arr_f32, size=3)
                variance = sq_mean - mean*mean
                out['noise'] = float(max(0.0, np.mean(np.sqrt(np.maximum(variance,0)))))
            except Exception:
                out['noise'] = 0.0
    except Exception as e:
        print(f"⚠️ compute_quality_scores error: {e}")
        pass
    return out


# ---------------- Qwen-VL 调用封装 (视觉理解) --------------------------------------
class ImageAnalyzer:
    def __init__(self,
                 emb_client=None,
                 vl_client=None,
                 llm_client=None,
                 vl_model: str = 'qwen3-vl-plus',
                 embedding_model: str = 'text-embedding-v4',
                 debug: bool = False):
        """
        emb_client: 用于 embeddings 的客户端（通义或 OpenAI compatible）
        vl_client: 用于视觉理解（chat.completions）客户端（可与 emb_client 相同）
        llm_client: 高阶逻辑分析（DeepSeek）客户端
        """
        self.emb_client = emb_client
        self.vl_client = vl_client or emb_client
        self.llm_client = llm_client
        self.vl_model = vl_model
        self.embedding_model = embedding_model
        self.debug = debug

    def _call_vl_json(self, image_path: str) -> Dict[str, Any]:
        """调用视觉模型，让它返回 JSON 格式的标签与描述。"""
        # fallback: 如果没有客户端，返回空结构
        if self.vl_client is None:
            return {}
        
        # 检查 Image 是否可用
        if Image is None:
            if self.debug:
                print('⚠️ _call_vl_json: Pillow not installed')
            return {}

        # 读取图片并 encode 为 base64 data url
        try:
            with Image.open(image_path) as im:
                buffer = io.BytesIO()
                im.convert('RGB').save(buffer, format='JPEG', quality=90)
                buffer.seek(0)
                img_bytes = buffer.read()
                import base64
                b64 = base64.b64encode(img_bytes).decode('utf-8')
                image_url_obj = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        except Exception as e:
            if self.debug:
                print('⚠️ _call_vl_json: 读取图片失败', e)
            return {}

        prompt_text = (
            "Analyze this image carefully and return JSON ONLY using ENGLISH for every field.\n\n"
            "CRITICAL: Distinguish between real photos and screenshots/documents:\n"
            "- 'photo': ONLY real-world captured images (people in real environments, nature, physical objects, travel, events)\n"
            "  → Look for: natural lighting, depth of field, outdoor/indoor scenes, people in real settings\n"
            "- 'screenshot': Computer/phone screen captures including:\n"
            "  → Software UI, apps, web pages, chat interfaces, code editors\n"
            "  → **Document screenshots** (Word, PDF, presentations, spreadsheets, text pages)\n"
            "  → Browser content, notifications, menus\n"
            "  → Any digitally rendered interface or text-heavy content\n"
            "- 'scanned': Physical documents scanned (forms, papers, receipts)\n"
            "- 'drawing': Hand-drawn or digital illustrations\n\n"
            "⚠️ Be strict: If the image shows ANY software interface elements, text documents, or digital content → it's 'screenshot', NOT 'photo'.\n\n"
            "JSON format:\n"
            "{\n"
            "  \"type\": \"photo|screenshot|scanned|drawing\",\n"
            "  \"scene\": \"short English scene description (e.g. 'outdoor portrait', 'document page', 'chat interface')\",\n"
            "  \"faces\": integer or null,\n"
            "  \"eyes_open\": true|false|null,\n"
            "  \"blur_score\": 0.0-1.0,\n"
            "  \"brightness_score\": 0.0-1.0,\n"
            "  \"composition_score\": 0.0-1.0,\n"
            "  \"is_screenshot_like\": true|false,\n"
            "  \"app_name\": null or app name if screenshot (e.g. 'WeChat', 'Browser', 'Word', 'PDF'),\n"
            "  \"short_description\": \"one English sentence describing the image content\"\n"
            "}\n"
            "If unsure about a field, use null. No additional text besides JSON."
        )

        try:
            response = self.vl_client.chat.completions.create(
                model=self.vl_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            image_url_obj,
                            {"type": "text", "text": prompt_text}
                        ]
                    }
                ],
                temperature=0.0
            )
            raw = response.choices[0].message.content.strip()
            # 尝试解析 JSON
            try:
                parsed = json.loads(raw)
                return parsed
            except Exception:
                # 有时候模型会输出带代码块或多余文本，尝试抽取第一个 { } json
                import re
                m = re.search(r"(\{[\s\S]*\})", raw)
                if m:
                    try:
                        return json.loads(m.group(1))
                    except Exception:
                        return {"_raw": raw}
                return {"_raw": raw}
        except Exception as e:
            if self.debug:
                print('⚠️ _call_vl_json: 调用视觉模型失败', e)
            return {}

    def embed_text(self, text: str) -> Optional[List[float]]:
        if not text:
            if self.debug:
                print('⚠️ embed_text: text is empty')
            return None
        if self.emb_client is None:
            if self.debug:
                print('⚠️ embed_text: emb_client is None')
            return None
        try:
            resp = self.emb_client.embeddings.create(model=self.embedding_model, input=text)
            return resp.data[0].embedding
        except Exception as e:
            if self.debug:
                print(f'⚠️ embed_text failed for text: "{text[:50]}..." - Error: {e}')
            return None

    def analyze_image_file(self, path: str) -> Dict[str, Any]:
        """主入口：分析单张图片并返回结构化信息"""
        p_obj = Path(path)
        stat = p_obj.stat()
        result: Dict[str, Any] = {
            "path": str(path),
            "size": stat.st_size,
            "mtime": stat.st_mtime
        }
        # 1. exif
        exif = read_exif(path)
        result['exif'] = exif

        # 2. 初步判断（基于 EXIF 和文件名）
        basic_check = is_likely_screenshot_basic(path, exif)
        result['likely_screenshot'] = basic_check if basic_check is not None else False
        result['_classification_source'] = 'basic'  # 跟踪分类来源（调试用）

        # 3. cv quality scores
        q = compute_quality_scores(path)
        # 归一化 blur_var（将方差映射到 0-1，阈值可调）
        blur_norm = 1.0 - (1.0 / (1.0 + math.log1p(q.get('blur_var', 0) + 1e-9)))
        result['quality_scores'] = {
            'blur_var': q.get('blur_var', 0),
            'blur_norm': float(blur_norm),
            'brightness': float(q.get('brightness', 0.5)),
            'exposure': float(q.get('exposure', 0.5)),
            'noise': float(q.get('noise', 0.0))
        }

        # 4. call VL model for JSON labels & description (更准确的判断)
        vl_json = self._call_vl_json(path)
        result['vl'] = vl_json
        
        # 5. 使用 VL 模型的判断覆盖初步判断（VL 模型更准确）
        if isinstance(vl_json, dict):
            vl_type = str(vl_json.get('type') or '').lower()
            vl_is_screenshot = vl_json.get('is_screenshot_like')
            
            # VL 模型明确判断为 photo → 照片
            if vl_type == 'photo':
                result['likely_screenshot'] = False
                result['_classification_source'] = 'vl_type:photo'
            # VL 模型明确判断为 screenshot/scanned/drawing → 截图类
            elif vl_type in ('screenshot', 'scanned', 'drawing'):
                result['likely_screenshot'] = True
                result['_classification_source'] = f'vl_type:{vl_type}'
            # VL 模型的 is_screenshot_like 字段
            elif vl_is_screenshot is not None:
                result['likely_screenshot'] = bool(vl_is_screenshot)
                result['_classification_source'] = f'vl_is_screenshot:{vl_is_screenshot}'
            # 如果 basic_check 返回 None（不确定），且 VL 也不确定，用场景辅助判断
            elif basic_check is None:
                scene = str(vl_json.get('scene') or '').lower()
                short_desc = str(vl_json.get('short_description') or '').lower()
                combined_text = scene + ' ' + short_desc
                
                # 自然场景关键词 → 照片（提高标准，需要更强的信号）
                natural_keywords = ['outdoor', 'landscape', 'nature', 'sky', 'people', 
                                   'portrait', 'animal', 'pet', 'street', 'beach',
                                   'mountain', 'forest', 'garden', 'sunset', 'sunrise', 'travel',
                                   'vacation', 'family', 'friends', 'selfie']
                
                # 界面/文档元素关键词 → 截图（扩充关键词）
                ui_keywords = ['interface', 'window', 'menu', 'button', 'dialog', 'toolbar',
                              'browser', 'webpage', 'application', 'screen', 'desktop',
                              'document', 'text', 'article', 'page', 'spreadsheet', 'table',
                              'chart', 'graph', 'slide', 'presentation', 'code', 'editor',
                              'terminal', 'console', 'form', 'input', 'search', 'notification',
                              'chat', 'message', 'email', 'pdf', 'word']
                
                # 统计关键词出现次数
                natural_count = sum(1 for kw in natural_keywords if kw in combined_text)
                ui_count = sum(1 for kw in ui_keywords if kw in combined_text)
                
                # 需要更强的信号才认定为真实照片
                if natural_count >= 2 and ui_count == 0:
                    # 至少2个自然关键词，且无UI关键词
                    result['likely_screenshot'] = False
                    result['_classification_source'] = f'scene_strong_natural:{natural_count}kw'
                elif ui_count >= 1:
                    # 只要有UI关键词，就倾向于截图
                    result['likely_screenshot'] = True
                    result['_classification_source'] = f'scene_ui:{ui_count}kw'
                elif natural_count == 1:
                    # 只有1个自然关键词，不够确定，检查文件格式
                    ext = Path(path).suffix.lower()
                    # 需要是 JPG 且有人脸才认定为照片
                    has_faces = vl_json.get('faces', 0) > 0
                    if ext in ('.jpg', '.jpeg') and has_faces:
                        result['likely_screenshot'] = False
                        result['_classification_source'] = 'weak_natural+jpg+faces'
                    else:
                        result['likely_screenshot'] = True
                        result['_classification_source'] = f'weak_natural_default_screenshot:{ext}'
                else:
                    # 完全不确定，默认为截图（保守策略）
                    ext = Path(path).suffix.lower()
                    result['likely_screenshot'] = True
                    result['_classification_source'] = f'fallback_default_screenshot:{ext}'

        # if vl_json contains scores, combine them moderately
        if vl_json:
            # copy returned fields if exist
            for k in ['blur_score', 'brightness_score', 'composition_score']:
                if k in vl_json:
                    try:
                        result['quality_scores'][k] = float(vl_json[k])
                    except Exception:
                        result['quality_scores'][k] = result['quality_scores'].get(k, None)

        # 5. generate rich textual description for embedding (more detailed for better similarity)
        description_parts = []
        if isinstance(vl_json, dict):
            # 主要描述
            main_desc = vl_json.get('short_description') or vl_json.get('scene')
            if main_desc:
                description_parts.append(main_desc)
            
            # 类型信息（重要：photo vs screenshot）
            img_type = vl_json.get('type')
            if img_type:
                description_parts.append(f"Image type: {img_type}")
            
            # 人物信息
            faces = vl_json.get('faces')
            if faces and faces > 0:
                description_parts.append(f"{faces} people")
                eyes_open = vl_json.get('eyes_open')
                if eyes_open is not None:
                    description_parts.append("eyes open" if eyes_open else "eyes closed")
            
            # APP信息（用于截图）
            app_name = vl_json.get('app_name')
            if app_name:
                description_parts.append(f"app: {app_name}")
        
        # 合并描述
        if description_parts:
            short_desc = ". ".join(description_parts)
        else:
            # fallback
            short_desc = Path(path).stem
        
        result['short_description'] = short_desc

        # 6. embedding (based on detailed description)
        emb = self.embed_text(short_desc)
        result['embedding'] = emb

        # 7. simple bestshot score combine: combine blur_norm, brightness, composition
        comp = None
        if isinstance(vl_json, dict):
            comp = vl_json.get('composition_score')
        comp = float(comp) if comp is not None else 0.5
        # weights: clarity 0.45, exposure 0.15, composition 0.3, face_quality 0.1
        face_factor = 1.0
        try:
            faces = int(vl_json.get('faces')) if isinstance(vl_json, dict) and vl_json.get('faces') is not None else 0
            eyes_open = vl_json.get('eyes_open') if isinstance(vl_json, dict) else None
            if faces > 0 and eyes_open is False:
                face_factor = 0.2
        except Exception:
            face_factor = 1.0

        clarity = result['quality_scores'].get('blur_norm', 0.0)
        exposure = result['quality_scores'].get('exposure', 0.5)
        bestshot_score = (0.45 * clarity + 0.15 * exposure + 0.3 * comp) * face_factor
        result['bestshot_score'] = float(bestshot_score)

        # 8. quick suggestion
        suggestion = {'delete': False, 'reason': None}
        if result['bestshot_score'] < 0.25:
            suggestion['delete'] = True
            suggestion['reason'] = 'very low quality (blur/dark/overexposed)'
        # repeated images detection should be done at batch level
        result['suggestion'] = suggestion

        return result

    # ---------------- batch helpers -------------------------------------------------
    def batch_embed_and_cluster(self, items: List[Dict[str, Any]], n_clusters: Optional[int] = None) -> Dict[str, Any]:
        """对 items (每项需包含 'embedding') 做聚类：
        - 如果 n_clusters None，则用简单的阈值 cosine 聚合
        返回一个 dict 包含 'clusters' (list of lists of idxs) 和 'centroids'
        """
        embeddings = [i.get('embedding') for i in items]
        idxs = [i for i,e in enumerate(embeddings) if e is not None]
        if not idxs:
            return {'clusters': [], 'centroids': []}
        X = np.array([embeddings[i] for i in idxs])
        # 如果 sklearn 可用并指定 n_clusters，使用 KMeans
        if KMeans is not None and n_clusters is not None and n_clusters > 0:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
            labels = kmeans.labels_
            clusters = {}
            for i,lab in zip(idxs, labels):
                clusters.setdefault(int(lab), []).append(i)
            centroids = kmeans.cluster_centers_.tolist()
            return {'clusters': list(clusters.values()), 'centroids': centroids}
        # otherwise 使用阈值合并（cosine）
        sims = cosine_similarity(X, X)
        used = set()
        clusters = []
        threshold = 0.78
        for i_local, i in enumerate(idxs):
            if i in used:
                continue
            group = [i]
            used.add(i)
            for j_local, j in enumerate(idxs):
                if j in used:
                    continue
                if sims[i_local, j_local] >= threshold:
                    group.append(j)
                    used.add(j)
            clusters.append(group)
        # centroids as mean
        centroids = [np.mean([embeddings[i] for i in g], axis=0).tolist() for g in clusters]
        return {'clusters': clusters, 'centroids': centroids}

    def screenshot_classify_with_llm(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """使用 LLM 对非照片类图片进行四分类（按删除可能性）并生成建议。
        
        注意：此函数只处理非照片类图片（截图、扫描件等）
        真实照片不需要调用此函数，直接使用 BestShot 逻辑
        
        四大类（按删除可能性从高到低）：
        1. temporary（临时类）- 删除可能性最高：会议截图、二维码、聊天图片、商品比价、临时凭证
        2. reference（参考类）- 删除可能性中高：流程图草稿、笔记截图、教程说明、网页保存内容
        3. saved（收藏类）- 删除可能性中低：表情包、海报、插画、社交媒体保存的内容
        4. memory（记忆类）- 删除可能性最低：扫描的重要文件、证件照等非真实拍摄但重要的图片
        
        期望 item 包含: path, vl(json), short_description
        返回: {category, app_name, suggestion}
        """
        if self.llm_client is None:
            return {}
        
        prompt = (
            f"Image path: {item.get('path')}\n"
            f"Description: {item.get('short_description')}\n"
            f"Vision tags: {json.dumps(item.get('vl', {}), ensure_ascii=False)}\n\n"
            "This is a NON-PHOTO image (screenshot/scan/etc). Classify into ONE of these 4 categories:\n\n"
            "1. 'temporary' - HIGHEST deletion likelihood (⭐⭐⭐⭐⭐)\n"
            "   - Meeting/classroom screenshots, web fragments\n"
            "   - Product comparison, tutorial screenshots\n"
            "   - QR codes, itineraries, vouchers\n"
            "   - Chat temporarily saved images\n"
            "   - Task reminders, meeting links\n\n"
            "2. 'reference' - MEDIUM-HIGH deletion likelihood (⭐⭐⭐⭐)\n"
            "   - Work reference screenshots (not final deliverable)\n"
            "   - UI prototypes, flowchart drafts\n"
            "   - Note screenshots, tutorial images\n"
            "   - Saved web content (news, articles)\n"
            "   - 'Might use later' materials\n\n"
            "3. 'saved' - MEDIUM-LOW deletion likelihood (⭐⭐⭐)\n"
            "   - Social media saved content\n"
            "   - Memes, posters, illustrations\n"
            "   - Entertainment/inspiration content\n\n"
            "4. 'memory' - LOWEST deletion likelihood (⭐⭐)\n"
            "   - Scanned important documents (ID, certificates)\n"
            "   - Historical screenshots with sentimental value\n"
            "   - Irreplaceable digital content\n\n"
            "Guidelines:\n"
            "- Temporary purpose → 'temporary'\n"
            "- Work/study reference → 'reference'\n"
            "- Saved for aesthetic/fun → 'saved'\n"
            "- Important/sentimental → 'memory'\n\n"
            "Identify source app if visible.\n"
            "Suggestion: 'delete' for temporary (unless important), 'keep' for others.\n\n"
            "Respond in ENGLISH JSON only:\n"
            f"{json.dumps({'category':'temporary|reference|saved|memory','app_name':'','suggestion':'keep'}, ensure_ascii=False)}\n"
            "No additional text."
        )
        try:
            # 默认使用 deepseek-chat，可以根据实际情况调整
            model_name = 'deepseek-chat'
            resp = self.llm_client.chat.completions.create(
                model=model_name,
                messages=[{"role":"user","content":prompt}],
                temperature=0.0
            )
            raw = resp.choices[0].message.content.strip()
            try:
                result = json.loads(raw)
                # 验证并标准化 category
                category = result.get('category', '').lower()
                if category not in ['temporary', 'reference', 'saved', 'memory']:
                    # 如果不在新分类中，尝试映射旧分类
                    old_to_new = {
                        'screenshot': 'temporary',
                        'software': 'temporary',
                        'flowchart': 'reference',
                        'document': 'reference',
                        'photo': 'memory',
                        'other': 'saved'
                    }
                    result['category'] = old_to_new.get(category, 'saved')
                return result
            except Exception:
                import re
                m = re.search(r"(\{[\s\S]*\})", raw)
                if m:
                    try:
                        result = json.loads(m.group(1))
                        # 验证并标准化 category
                        category = result.get('category', '').lower()
                        if category not in ['temporary', 'reference', 'saved', 'memory']:
                            old_to_new = {
                                'screenshot': 'temporary',
                                'software': 'temporary',
                                'flowchart': 'reference',
                                'document': 'reference',
                                'photo': 'memory',
                                'other': 'saved'
                            }
                            result['category'] = old_to_new.get(category, 'saved')
                        return result
                    except Exception:
                        return {"_raw": raw, "category": "saved"}
                return {"_raw": raw, "category": "saved"}
        except Exception as e:
            if self.debug:
                print('⚠️ screenshot_classify_with_llm failed', e)
            return {"category": "other"}

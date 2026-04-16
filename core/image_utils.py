"""图片工具函数。"""

import base64
import io
import os
import shutil
from PIL import Image


def pil_to_b64(img: Image.Image, fmt: str = "JPEG") -> str:
    """PIL Image 转 data URI base64。"""
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def compress_for_vlm(img: Image.Image, max_long_edge: int = 2048, jpeg_quality: int = 85) -> str:
    """压缩图片用于 VLM 调用：缩放 + JPEG 压缩 → data URI base64。

    将大图（如 23MB PNG）压缩到 ~2-4MB JPEG，base64 后 ~3-6M 字符，
    安全通过 20M 字符限制，同时大幅降低网络传输时间。
    """
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_long_edge:
        ratio = max_long_edge / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def file_to_b64(path: str) -> str:
    """本地图片文件转 data URI base64。"""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def copy_image(src: str, dst_dir: str, new_name: str | None = None) -> str:
    """复制图片到目标目录，返回目标路径。"""
    os.makedirs(dst_dir, exist_ok=True)
    fname = new_name or os.path.basename(src)
    dst = os.path.join(dst_dir, fname)
    shutil.copy2(src, dst)
    return dst

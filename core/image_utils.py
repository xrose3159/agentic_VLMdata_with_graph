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

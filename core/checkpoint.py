"""检查点管理：按步骤+图片ID粒度，支持断点续跑。"""

import json
import os
from typing import Optional

from .config import CHECKPOINT_DIR


def _ckpt_path(step: int, img_id: str) -> str:
    d = os.path.join(CHECKPOINT_DIR, f"step{step}")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{img_id}.json")


def is_done(step: int, img_id: str) -> bool:
    return os.path.exists(_ckpt_path(step, img_id))


def save_checkpoint(step: int, img_id: str, data: dict) -> None:
    path = _ckpt_path(step, img_id)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_checkpoint(step: int, img_id: str) -> Optional[dict]:
    path = _ckpt_path(step, img_id)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)

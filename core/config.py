"""全局配置，集中管理 API、路径、阈值等常量。"""

import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# VLM API
# ============================================================
API_KEY    = os.environ.get("API_KEY", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen-vl-max")
BASE_URL   = os.environ.get("BASE_URL", "")

# ============================================================
# 外部工具 API
# ============================================================
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
SERPER_KEY     = os.environ.get("SERPER_KEY", "")  # web_search / image_search(text) / Google Lens
E2B_API_KEY    = os.environ.get("E2B_API_KEY", "")
JINA_API_KEY   = os.environ.get("JINA_API_KEY", "")
JINA_READER_URL = "https://r.jina.ai/"
JINA_TIMEOUT    = 15  # 秒

# ============================================================
# 路径
# ============================================================
SOURCE_IMAGE_DIR   = "images"
OUTPUT_ROOT        = "output"
FILTERED_IMAGE_DIR = os.path.join(OUTPUT_ROOT, "images")
ENTITY_DIR         = os.path.join(OUTPUT_ROOT, "entities")
QUESTION_DIR       = os.path.join(OUTPUT_ROOT, "questions")
FINAL_DIR          = os.path.join(OUTPUT_ROOT, "final")
STATS_DIR          = os.path.join(OUTPUT_ROOT, "stats")
CHECKPOINT_DIR     = os.path.join(OUTPUT_ROOT, ".checkpoints")

# ============================================================
# 第一步筛选阈值
# ============================================================
FILTER_MIN_TOTAL   = 18
FILTER_MIN_PER_DIM = 3
FILTER_DIMENSIONS  = [
    "entity_richness", "detail_depth", "external_linkage",
    "entity_relations", "naturalness",
]

# ============================================================
# 工具定义
# ============================================================
VALID_TOOLS = {"web_search", "image_search", "visit", "code_interpreter"}

# ============================================================
# 并发与重试
# ============================================================
MAX_WORKERS      = 4
VLM_MAX_RETRIES  = 3
VLM_TIMEOUT      = 120
RATE_LIMIT_DELAY = 0.5   # 秒，每次 VLM 调用后的最小间隔

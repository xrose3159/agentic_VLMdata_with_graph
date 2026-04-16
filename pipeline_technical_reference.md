# Pipeline 技术文档

## 架构概览

```
images/
  ↓
batch_generate.py  高并发批量入口
pipeline.py        编排器（Step1 → Step2，可并发）
  ↓
step1_graph.py     Step1: 构建知识图谱
  ↓ 输出 output/entities/{img_id}.json
step2_generate.py → step2_question.py  Step2: 基于图谱出题
  ↓ 输出 output/questions/{img_id}.json
```

核心文件 6 个 + core/ 工具库。

---

## 文件清单

| 文件 | 职责 |
|------|------|
| `batch_generate.py` | 高并发批量生成入口（指定图片目录 + 并发数） |
| `pipeline.py` | 编排器：遍历图片，每张跑 Step1→Step2 |
| `step1_graph.py` | Step1: VLM 识别实体 → 搜索验证 → 三元组提取 → 多轮扩展 → 输出图谱 |
| `step2_generate.py` | Step2 入口：调 step2_question + checkpoint 管理 + 聚合输出 |
| `step2_question.py` | Step2 核心：格式化图谱 → 一次 LLM 调用出题 → 分类 |
| `step2_graphenv_runtime.py` | 工具函数库：relation 评分、自然语言化、兼容性判断 |
| `core/config.py` | 全局配置：API key、路径、阈值 |
| `core/vlm.py` | VLM/LLM 调用封装（支持图文混合和纯文本两种模式） |
| `core/image_utils.py` | 图片压缩、base64 编码 |
| `core/lens.py` | Google Lens 反向图搜（Serper） |
| `core/checkpoint.py` | 断点续跑（按 step + img_id 粒度） |

---

## batch_generate.py（高并发入口）

```bash
python batch_generate.py images/                   # 默认 10 并发
python batch_generate.py /path/to/photos/ -w 20    # 20 并发
python batch_generate.py images/ --no-cache         # 禁用 API 缓存
python batch_generate.py images/ --limit 50         # 只跑前 50 张
```

对每张图独立执行 Step1 → Step2，互不依赖，ThreadPoolExecutor 并发。完成后自动聚合为 JSONL。

---

## Step1：构建知识图谱（step1_graph.py）

入口函数：`enrich_image(img_path) -> dict | None`

整体流程：VLM 看图识别实体 → 搜索验证身份 + 收集 URL → 深读 URL 提取三元组 → 跨实体搜索 → 多轮扩展 → 别名归一

### 1.1 VLM 实体识别

**一次 VLM 调用**，输入图片 base64，输出 `image_description` + `entities[]`。

每个实体：`name`（实体名）、`type`（brand/person/landmark/product/text/object）、`bbox`（0-1000 归一化坐标）。

Prompt 明确要求识别人物（运动员、名人等），用真名或描述性名称（如"白色27号球衣球员"）。

**后处理**：
- 名称去重（忽略大小写）
- 归一化坐标 → 像素坐标
- 幻觉拦截：bbox 面积 > 80% 全图 → 移除
- 位置描述：`_bbox_to_location()` 从 bbox 中心比例计算中文方位词（如"画面上方偏左"）
- 5 档裁剪保存：tight / pad20（Lens 用）/ pad40 / context(4x) / full_scene

**跳过条件**：实体数 < 3

### 1.2 实体 Resolution（验证 + 收集 URL）

`_resolve_entities(entities, img_id)` — 4 个 Phase 并行/串行组合执行：

**Phase 1a — image_search(text)**：对 person/landmark/product 类实体
- 用实体名搜 Google Images（Serper /images API，max_results=5）
- 目的：验证 VLM 给的名字是否正确（看搜索标题是否匹配）
- 副产品：收集 source_url 进 visit 队列

**Phase 1b — web_search**：对 brand/text/object 类实体
- 用实体名搜 Google（Serper /search API，max_results=5）
- 目的：直接获取知识信息（snippets）
- 副产品：收集 URL 进 visit 队列

**Phase 2 — Google Lens 反向图搜**：对 person/landmark/product 类实体
- pad20 裁剪图 → 上传 litterbox（1h 临时图床，无需 API key）→ 调 Serper Lens API
- 产出：候选标题（经 LLM 清洗提取干净实体名）+ 来源页 URL
- 无 SERPER_KEY 时 fallback 到 VLM describe workaround

**Phase 3 — Visit 来源页**
- Jina Reader 并发深读所有收集到的 URL（image_search + web_search + Lens 来源页）
- 上限 30 个 URL，每个最多读 2000 字符
- Jina 付费 key 余额耗尽(402) → 自动 fast-fail 切免费模式（本次运行内不再尝试付费）

**Phase 4 — 合并 resolution 信息**
- 每个实体的 resolution 字段汇总：image_search_titles、web_search_snippets、lens 候选、visited_pages

### 1.3 实体池分层

- **core_anchors**：confidence >= 0.8 的实体
- **tier1**（full budget）：原始 VLM 检测的高置信实体 → 参与跨实体搜索和扩展
- **demoted**：其余实体，不参与后续搜索
- **Brand-dense 模式**：tier1 > 12 → 按面积 × 类型加权排序只保留 top-12

### 1.4 跨实体搜索

发现图中不同实体之间的真实世界关联：

1. tier1 实体两两配对（最多 5 对，按 confidence 降序选）
2. LLM 为每对生成 1-2 条搜索词（1 次 LLM 调用）
3. 并行 Serper web_search
4. snippet 同时提到两个实体名 → 才 visit 深读（否则跳过，省 API）
5. LLM 从搜索结果提取桥接三元组（1 次 LLM 调用）

### 1.5 三元组提取

把 Resolution 阶段收集到的所有信息（image_search 标题、web_search snippets、Lens 候选、visited_pages 正文）汇总，**一次 LLM 调用**提取结构化三元组。

每条三元组：
```json
{
  "head": "KitchenAid",
  "relation": "founded_in",
  "tail": "1919",
  "tail_type": "TIME",
  "normalized_value": 1919,
  "unit": "year",
  "fact": "KitchenAid 由 Hobart 公司于 1919 年创立",
  "source_snippet": "The company was started in 1919 by The Hobart Manufacturing Company..."
}
```

- `tail_type`：TIME / QUANTITY / LOCATION / PERSON / ORG / OTHER
- `normalized_value` + `unit`：TIME/QUANTITY 类三元组的标准化值，方便后续做比较/排序

### 1.6 多轮扩展搜索

目的：把图谱加厚，让三元组数量达到出题所需的密度。

每轮：
1. 选三元组里被引用最多的 tail 实体（最多 6 个）
2. LLM 为每个 tail 生成搜索词（1 次 LLM 调用，优先数值/日期/位置/人名类）
3. 对每个 tail 搜两条：exact_name（实体原名）+ rewrite（LLM 改写的带方向性搜索词）
4. 并行 Serper web_search + Jina visit 深读
5. LLM 从结果提取新三元组

**停止条件**：三元组 >= 30 条 或 最多 2 轮

### 1.7 后处理

1. **启发式实体名归一**：substring 匹配，把三元组里漂移的实体名统一回图中实体的标准名
2. **LLM 别名归一**：1 次 LLM 调用，把同义变体分组合并（如 "Nuggets" / "Denver Nuggets" → "Denver Nuggets"）
3. **空间关系兜底**：对在三元组图中没有直接连接的实体对，补 located_left_of / located_above 等空间关系

### Step1 输出

`output/entities/{img_id}.json`：
```json
{
  "img_id": "img_0010",
  "image_description": "纽约时代广场的照片...",
  "domain": "geography",
  "entities": [
    {"id": "E1", "name": "McDonald's", "type": "brand", "bbox": [50,1200,314,1514], "location_in_image": "画面下方最左侧", ...}
  ],
  "high_conf": [...],
  "local_artifacts": {"numeric_labels": [], "bbox_areas": [], "layout_relations": []},
  "search_results": [...],
  "triples": [
    {"head": "McDonald's", "relation": "founded_in", "tail": "1940", "tail_type": "TIME", "normalized_value": 1940, "unit": "year", ...}
  ]
}
```

---

## Step2：基于图谱出题（step2_question.py）

入口函数：`generate_questions(entity_json_path, image_path, seed, n_questions)`

### 2.1 格式化图谱

直接把 Step1 输出的 entities + triples 格式化为 LLM 能理解的文本：

```
【图中实体】
  - McDonald's  类型=brand  位置=画面下方最左侧  bbox=[50,1200,314,1514]
  - Francis P. Duffy Statue  类型=landmark  位置=画面底部  bbox=[1541,1461,1839,1943]
  ...

【知识三元组】（共 45 条）
  McDonald's —[founded_in]→ 1940 [TIME] (1940 year)
  McDonald's —[headquartered_in]→ Chicago [LOCATION]
  One Times Square —[height]→ 363 feet [QUANTITY] (363 feet)
  One Times Square —[designed_by]→ Cyrus L. W. Eidlitz [PERSON]
  ...

【空间细节】
  McDonald's: 中心=(182,1357) 尺寸=264×314 面积=82896px²
  Francis P. Duffy Statue: 中心=(1690,1702) 尺寸=298×482 面积=143636px²
  ...

图片描述: 纽约时代广场的照片，可以看到多个品牌广告牌...
```

内部用 `HeteroSolveGraph` 构建异构图（full_image → region → entity → fact 四层），用于过滤低价值 relation（askability <= 0 的边不进入格式化文本）。

### 2.2 一次 LLM 调用出题

把格式化文本 + 出题 prompt 喂给 LLM，一次返回 9 道题（JSON 数组）。

**出题结构**：3 个方向（retrieval / code / hybrid）× 3 个难度（easy / medium / hard）= 9 道题

### 智能体工具集

4 种工具，其中 `code_interpreter` 内置丰富的 OpenCV 图像处理能力：

| 工具 | 功能 |
|------|------|
| web_search | 搜索网络获取文本信息 |
| image_search | 文字搜图或反向图搜识别实体 |
| visit | 访问网页深度阅读 |
| code_interpreter | 执行 Python 代码（PIL/OpenCV/easyocr） |

code_interpreter 内置的视觉操作：

| 类别 | 操作 |
|------|------|
| 几何变换 | crop, rotate, flip, resize, zoom_in |
| 特征检测 | canny_edge, hough_line, hough_circle, contour（面积/周长） |
| 分割与过滤 | grabcut 前景分割, inrange_color 颜色过滤（HSV 阈值） |
| 标注绘制 | draw_line, draw_circle, draw_bbox |
| 颜色分析 | 主色调/HSV 统计, hist_eq 直方图均衡化 |
| 文字识别 | OCR（easyocr） |
| 测量计算 | 面积、距离、像素统计、坐标定位 |

### 视觉类型分类（按答案来源，互斥）

| 视觉类型 | 答案从哪来 | 示例 |
|---------|-----------|------|
| **pixel_reading** | 从像素中读出内容（文字、数量、颜色） | "小字写的是什么？"→"SALE 50%" |
| **pixel_computing** | 从像素坐标算出（面积、距离、排序） | "哪个面积更大？"→"A，差1200px²" |
| **pixel_operating** | 图像变换后才能确定（边缘检测、分割、拼接、路径） | "做边缘检测后轮廓周长是多少？"→"1234px" |

**分类规则**：看产出最终答案的那一步——读像素→reading，算坐标→computing，验变换→operating，推模式→reasoning。

每道题额外标记 `needs_knowledge: true/false`（是否需要外部知识检索），与视觉类型正交。

### 难度定义（按工具步数，客观）

| 难度 | 工具步数 |
|------|---------|
| easy | 1-2 步 |
| medium | 3-4 步 |
| hard | 5+ 步 |

### Prompt 中的答案质量约束

- **答案必须唯一确定**：只能有一个正确答案，禁止"有哪些""列举"类
- **答案必须可验证**：通过工具调用得到确定性结果，不允许主观判断
- **答案必须基于图谱**：不引入图谱外的随机知识
- **short_answer 是精确值**：数字、人名、地名，不是列表或段落

**LLM 参数**：temperature=0.6，max_tokens=4096

### 2.3 题目分类与输出字段

每道题的完整字段：

```json
{
  "question_id": "pixel_computing_hard_01",
  "question": "画面中 Mamma Mia! 和 Toy Story 3 广告牌，哪个面积更大？",
  "answer": "Mamma Mia! 面积为 82362px²，Toy Story 3 为 81162px²，前者更大。",
  "short_answer": "Mamma Mia!",
  "tools": ["code_interpreter"],
  "visual_type": "pixel_computing",
  "needs_knowledge": false,
  "difficulty": "easy",
  "n_steps": 1,
  "evidence": ["Mamma Mia!: 面积=82362px²", "Toy Story 3: 面积=81162px²"],
  "reasoning": "从 bbox 计算两个广告牌面积并比较"
}
```

### Step2 输出

`output/questions/{img_id}.json`：
```json
{
  "retrieval": [...],
  "code": [...],
  "hybrid": [...],
  "metadata": {
    "graph_stats": {"nodes": {...}, "edges": {...}},
    "category_counts": {"retrieval": 3, "code": 3, "hybrid": 3},
    "total_questions": 9
  }
}
```

---

## 外部 API 依赖

| API | 服务商 | 用途 | 调用时机 | 环境变量 |
|-----|--------|------|---------|---------|
| VLM | Qwen-VL / Gemini 等 | 实体识别、三元组提取、别名归一、出题 | Step1 + Step2 | `API_KEY` / `MODEL_NAME` / `BASE_URL` |
| Text LLM（可选） | 同上 | 纯文本调用（搜索计划、三元组提取等不传图片时） | Step1 | `TEXT_LLM_*` |
| Serper web_search | serper.dev | Google 搜索 | Step1 Resolution + 跨实体 + 扩展 | `SERPER_KEY` |
| Serper image_search | serper.dev | Google Images 搜索 | Step1 Resolution | `SERPER_KEY` |
| Serper Google Lens | serper.dev | 反向图搜 | Step1 Resolution | `SERPER_KEY` |
| Jina Reader | jina.ai | 网页深读 | Step1 visit | `JINA_API_KEY`（可选） |
| Litterbox | catbox.moe | 临时图床（上传 crop 给 Lens） | Step1 Lens | 无需 key |

**API 缓存**：所有 Serper / Jina 调用走 disk cache（`output/.cache/{serper_web,serper_img,serper_lens,jina}/`），缓存 key = sha256(请求参数)[:16]。相同请求不重复调 API。环境变量 `DISABLE_API_CACHE=1` 可禁用。

---

## 单张图片的完整 API 调用链（以 5 个实体的图为例）

| 阶段 | VLM/LLM | Serper web | Serper img | Serper Lens | Jina visit |
|------|---------|------------|------------|-------------|------------|
| 1.1 实体识别 | 1 | | | | |
| 1.2 Resolution Phase 1a (img_search) | | | ~2 | | |
| 1.2 Resolution Phase 1b (web_search) | | ~3 | | | |
| 1.2 Resolution Phase 2 (Lens) | | | | ~2 | |
| 1.2 Resolution Phase 3 (visit) | | | | | ~15 |
| 1.2 Lens 候选清洗 | 1 | | | | |
| 1.4 跨实体搜索计划 | 1 | | | | |
| 1.4 跨实体搜索执行 | | ~7 | | | |
| 1.4 跨实体三元组提取 | 1 | | | | |
| 1.5 主三元组提取 | 1 | | | | |
| 1.6 扩展搜索词生成 (每轮) | 1 | | | | |
| 1.6 扩展搜索执行 (每轮) | | ~10 | | | ~6 |
| 1.6 扩展三元组提取 (每轮) | 1 | | | | |
| 1.7 别名归一 | 1 | | | | |
| **Step1 小计** | **~9** | **~20** | **~2** | **~2** | **~21** |
| 2.2 出题 | 1 | | | | |
| **总计** | **~10** | **~20** | **~2** | **~2** | **~21** |

总 API 调用约 55 次/图（VLM 最贵，约 10 次；Serper 和 Jina 较便宜）。

---

## 超参数汇总

| 参数 | 位置 | 默认值 | 说明 |
|------|------|--------|------|
| 最少实体数 | step1 | 3 | 少于此数跳过该图 |
| image_search/web_search max_results | step1 | 5 | 每次搜索返回的最大结果数 |
| Lens source_pages | step1 | top-15 | 从 Lens 结果取的来源页数 |
| visit max_urls | step1 | 30 | Resolution 阶段最多 visit 的 URL 数 |
| visit max_chars | step1 | 2000 | 每个 URL 最多读取的字符数 |
| 跨实体对数上限 | step1 | 5 | tier1 实体两两配对的上限 |
| 图谱达标阈值 | step1 | 30 条三元组 | 停止扩展搜索的条件 |
| 最大扩展轮数 | step1 | 2 | 每轮搜 6 个尾实体 |
| brand-dense 阈值 | step1 | tier1>12 | 触发 top-12 截断 |
| 出题总数 | step2 | 9 | 3方向 × (1简单+2难) |
| LLM temperature（出题） | step2 | 0.6 | |
| LLM max_tokens（出题） | step2 | 4096 | |
| VLM 重试次数 | core | 3 | 单次调用最多重试 |
| VLM 超时 | core | 120s | 单次调用超时 |
| VLM rate limit | core | 0.5s | 每次调用后的最小间隔 |
| Jina 超时 | core | 15s | 单次 visit 超时 |
| 并发数 | batch | 10 | batch_generate.py 的 -w 参数 |

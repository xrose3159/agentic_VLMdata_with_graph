- ## 概述

  - 本 Pipeline 从候选图片出发，自动生成 **（图片, 问题, 答案, 工具序列）** 四元组的多模态 agentic 训练数据。数据分为三个难度等级（L1/L2/L3），难度逐级递增，模糊化程度和所需工具种类也逐级增加。
  - 八步流程（6 主步 + 2 post-step）：
    - **Step1 图片筛选**（可选，已支持跳过） — VLM 打分过滤，或直接用 `--source images/` 跳过
    - **Step2 异构证据图构建** — VLM 实体识别 → 场景自适应 person proposal 第二遍 → 5 档 crop → 分层预算搜索（promotion gate + web-first visit-later） → 跨实体桥接 → 三元组提取 → deficit-driven 多轮扩展 → 真 Serper Lens reverse → LLM 别名归一 → disk cache
    - **Step3 弱约束随机游走 + 后验归纳题型** — 异构求解图（含 synthetic 桥接实体）→ 自由扩展子图 → 后验枚举可闭合意图（含 multi_hop 2/3-hop）→ 不可约性检查（含 3-hop shortcut 检测）→ QuestionFrame 语言化（chain_trace 防翻译漂移）
    - **Step3b 轻量链可读性过滤** — LLM 对 L3 题打 flow/motivation/naturalness 三维分，拦截凑题感强的不自然链
    - **Step4 模糊化验证与修正** — 结构性校验 + VLM 模糊化审查 + 自动修正
    - **Step5 攻击式过滤** — A/B/C 三层 attacker + bucket-aware two_phase 调度（A0 全量 no_tool + A1 bucket-specific cascade）
    - **Step6 breach taxonomy 统计** — by_bucket × attack × family 聚合 + 阈值 verdict（keep_B / inspect / implement_A_prime）
    - **落盘 hard split** — `aggregate_hard_split.py` 聚合通过 Tier A+B 过滤的 L3 题为 JSONL 数据集

  ### 方法论：规则最小化 + 数据驱动验证

  - **不手写领域词典**：Step2/Step3 的 relation 处理都走通用启发式（tail_type + token 数），换新领域（医药/电影/产品）不改代码
  - **不写 shortcut 预防规则**：生成阶段只做结构性不可约性检查（5+3 道 shortcut 门），通过/降级；shortcut 的实际存在性交给 Step5 攻击器验证
  - **阈值驱动结构性修复**：Step6 统计 breach rate，只有 >15% 才考虑 targeted fix，>30% 才实施 A'（coarse-anchor substitution），避免凭 1-2 个 edge case 就"对旧分布过拟合"
  
  ## 工具定义

  - 本 Pipeline 生成的数据围绕以下四种工具，所有生成的问题必须严格使用这些工具定义：
  - TOOLS = {
  - ​    "web_search": {
  - ​        "description": "搜索网络获取文本信息。用于查找事实、规格参数、新闻、价格等。",
  - ​        "parameters": {
  - ​            "query": "搜索查询字符串（必填）",
  - ​            "max_results": "最大返回结果数（默认10）"
  - ​        },
  - ​        "output": "搜索结果列表，每条包含title, url, snippet"
  - ​    },
  - ​    "image_search": {
  - ​        "description": "通过文本描述搜索图片，或通过图片进行反向搜索（Serper Google Lens）。",
  - ​        "parameters": {
  - ​            "search_type": "'text'（文字搜图）或 'reverse'（以图搜图 via Serper Lens），默认'text'",
  - ​            "query": "搜索查询字符串（text 模式必填）",
  - ​            "image_url": "图片 URL（reverse 模式必填，必须是公网可访问 URL）",
  - ​            "max_results": "最大返回结果数（默认10）"
  - ​        },
  - ​        "output": "结果列表，每条包含 title / source_page_url / source_domain / thumbnail_url / description。reverse 模式额外可能返回 knowledge_graph。"
  - ​    },
  - ​    "visit": {
  - ​        "description": "访问指定网页并提取主要内容。用于从搜索结果中获取详细信息。",
  - ​        "parameters": {
  - ​            "url": "完整网页URL（必须以http://或https://开头）",
  - ​            "goal": "你希望从该页面获取什么信息（辅助提取）"
  - ​        },
  - ​        "output": "网页的主要文本内容"
  - ​    },
  - ​    "code_interpreter": {
  - ​        "description": "执行 Python 代码。用于：图像裁剪/几何（PIL/OpenCV）、OCR（pytesseract/easyocr）、数值计算与标准化、多源证据投票/去重/交集。原图可通过 IMAGE_PATH 访问。",
  - ​        "parameters": {
  - ​            "code": "要执行的 Python 代码（必填）"
  - ​        },
  - ​        "output": "代码执行结果（stdout、生成的文件路径等）",
  - ​        "preloaded_vars": "IMAGE_PATH（原图绝对路径）、WORK_DIR（临时工作目录）",
  - ​        "packages": "PIL, NumPy, OpenCV, Matplotlib, SciPy, Pandas, SymPy, pytesseract, easyocr, re, datetime, json",
  - ​        "skill_tags": [
  - ​            "cv_preprocess (crop/resize/threshold/contour)",
  - ​            "ocr_parse (OCR + regex/字段提取)",
  - ​            "layout_geometry (bbox 面积/距离/排序)",
  - ​            "tabular_aggregate (groupby/merge/pivot)",
  - ​            "normalize_compare (日期/单位/货币标准化后比较)",
  - ​            "evidence_vote_merge (多源去重/交集/投票)"
  - ​        ]
  - ​    }
  - }
  
  ## 第一步：筛选高信息密度图片（step1_filter.py）
  
  ### 目标
  
  - 从 `images/` 目录的候选图片中筛选出信息密度足够高、适合生成 agentic 训练数据的图片。
  
  ### 流程
  
  1. 遍历 `images/` 目录下所有图片（jpg/jpeg/png/webp）
  2. 对每张图片，将其 base64 编码后发送给 VLM，按 5 个维度打分（1-5 分）：
     1. **实体丰富度（entity_richness）**：图中有多少个可独立识别的实体
     2. **信息层次性（detail_depth）**：是否有需要放大/裁剪才能看清的细节
     3. **外部知识关联（external_linkage）**：图中实体是否能关联到可搜索的外部知识
     4. **多实体关系（entity_relations）**：实体之间是否存在可推理关系
     5. **自然真实性（naturalness）**：图片是否来自真实场景
  
  - 这里到底算什么样的图片算好图片？目前选择信息密度高的图片感觉不太自然
  
  - high-information -> high-agentic-potential
  - 信息丰富度打分（step1-1）+ agent 潜力打分（step1-2）
  - step1-2：让模型根据图片生成 agent task 草案，再反推潜力：
  -  给 VLM 看图，让它输出 2–3 个最小任务草案，每个草案只要这几个字段：
    - `user_goal`
    - `required_tools`
    - `why_tools_needed`
    - `expected_answer_type`
  -  再对任务草案进行打分。
    - 筛选条件：**总分 ≥ 18** 且 **每项 ≥ 3**
    - VLM 返回的 `pass` 字段由代码重新校正（不信任 VLM 的布尔判断）
    - 通过的图片复制到 `output/images/`，同时记录评分统计和类别分布

  ### 输出

  - `output/images/` — 筛选通过的图片
  - `output/stats/filter_scores.json` — 所有图片的评分、类别分布
  
  ## 第二步：异构证据图构建（step2_enrich.py）

  ### 目标

    从每张筛选后的图片构建一张**足够厚的异构证据图**，使 Step3 的随机游走能自由探索、自然长出多种 hard motif。

    停止条件是 **closure richness 达标**，不是 triples 数量。目标从"造很多三元组"改成"造很多可闭合的 hard motif"。
    具体看 6 个指标（`image_resolved_anchors / page_only_facts / compare_ready_pairs / rank_ready_triplets / cross_anchor_shared_nodes / total_triples`），满足任意 3 项即停。

  **方法选择：**

  - 实体提取：纯 VLM，直接输出 Bounding Box
  - 外部检索：Serper web_search + image_search(text)
  - 网页深读：Jina Reader（visit）
  - 搜索词：代码确定性生成（exact-name-first），不让 LLM 写搜索计划
  - 实体池：分两层（core_anchors 给 Step3 出题 / expansion_seeds 给 Step2 扩图，不限数量）
  - 不使用 YOLO / OCR / SAM 等额外模型

  **四种工具在 Step2 中的实际使用：**

  | 工具 | 用途 | 阶段 | 实际价值 |
  | ---- | ---- | ---- | ------- |
  | web_search | 搜实体名获取硬事实 | 跨实体搜索 + 主搜索 + 扩展搜索 | **核心**，三元组主要来源 |
  | visit (Jina Reader) | 深读搜索结果页 | 主搜索深读稳定 / 跨实体不稳定 | 主搜索可靠，其他阶段经常失败 |
  | image_search(text) | 文字搜图 + visit 来源页 | 实体 resolution | **有限**，和 web_search 信息重叠 |
  | image_search(reverse) — **真 Serper Lens** | 上传 crop 到 litterbox → Serper Lens → 拿 visual_matches 候选 + 来源页 | 实体 resolution（person/landmark/product） | ✅ **真改图**，候选进 expansion_seeds，来源页进 visit queue |
  | code_interpreter | 多视图裁剪（tight / pad20 / context） | 实体提取后处理 | search_views 已生成 |

  **Reverse search 真实方案（替代旧 VLM workaround）：** 使用 `core/lens.py` 模块。流程：

  1. crop（pad20）通过 `litterbox.catbox.moe`（1 小时临时图床，无 API key）上传拿公网 URL
  2. 调 `POST https://google.serper.dev/lens`（用现有 SERPER_KEY，**不需要额外 SerpApi/Bing/TinEye 账号**）
  3. 从 `organic` 列表里抽 top-20 candidate titles + top-15 source pages
  4. **候选标题真的进 `discovered_entities` → 经 LLM canonicalize → 加入 `expansion_seeds`**
  5. **来源页真的进 `visit_tasks` → 转化为 page_only fact**
  6. lens visual_matches > 0 → 强制升级 `resolve_mode = image_search_needed`（真证据，比规则可信）

  `SERPER_KEY` 不存在时自动 fallback 到旧的 VLM describe workaround。详见 `core/lens.py`。

  ### 流程

  #### 2a. 纯 VLM 实体提取

    将图片 base64 编码后发送给 VLM，输出每个实体的 `name`、`type`（brand/landmark/text/person/product/object）、`bbox`（0-1000 归一化）。VLM 同时输出图片整体描述。

  **后处理：**

  1. **实体去重**：名称相同（忽略大小写）只保留第一个
  2. **坐标转换**：归一化 bbox → 像素坐标
  3. **幻觉拦截**：bbox 面积占全图 >80% → 移除
  4. **位置描述**：`_bbox_to_location()` 从像素坐标计算（不依赖 VLM）
  5. **5 档搜索视图裁剪（search_views）**：

     | 档位 | padding | 用途 |
     | --- | --- | --- |
     | `tight` | 0（紧贴 bbox） | OCR + 精细 face/logo 识别 |
     | `pad20` | bbox 四周扩 20% | **Lens 反查默认用这档**（经 canary 对比 pad40 命中率更高） |
     | `pad40` | bbox 四周扩 40% | 更大周边上下文，给 Step3 未来做视觉参考用 |
     | `context` | 以实体为中心 4× 面积 | 场景级周边（球场/货架/招牌墙） |
     | `full_scene` | 整张图 | 场景匹配后备（单图共享一份，减少磁盘 I/O） |

     **Lens 反查选 pad20 的原因**：canary 对比发现 pad40 带进太多无关背景稀释 Lens 视觉信号（poster_01 从 ev_marked=1 降到 0），回退到 pad20。

    每个实体输出：`name`、`type`、`value`、`bbox`、`location_in_image`、`confidence`、`confidence_level`、`search_views: {tight, pad20, pad40, context, full_scene}`

    实体数少于 3 个则跳过该图片。

  #### 2a-1b. Person proposal 第二遍（场景自适应，条件触发）

  **问题**：VLM 主 prompt 导向识别"品牌/文字/标志"，在 sports、演唱会、颁奖典礼、会议等场景里完全不检测人物 bbox。这不是 type 分类错误，是**检测层面的缺失**（实测 8 张 NBA 图 84 个实体仅 1 个 person）。

  **触发条件**（三者都满足）：
  1. 第一遍实体里 `person_count == 0`
  2. 图片描述含"图里可能有人"的关键词（覆盖 6 大场景域，不限 sports）：
     - **体育**：NBA/arena/court/stadium/jersey/球场/球员 等
     - **演艺**：concert/stage/performer/演唱会/舞台 等
     - **典礼/会议**：ceremony/award/conference/颁奖/会议 等
     - **公共场景**：crowd/pedestrian/tourist/人群/行人 等
     - **职业场景**：factory/worker/restaurant/chef/工厂/工人 等
     - **通用人物**：player/coach/actor/politician/运动员/演员/导演 等
  3. 第一遍实体数 ≥ 3（排除空图）

  **方案**：tile 切割 + person-only prompt，**不改主 prompt**
  1. 把整图切成 **2×2 tiles**（解决人太小 + 注意力被 logo/记分牌抢走的问题）
  2. 每个 tile 单独问 VLM **"只标出人物，不要标 logo/文字"**，最多 3 人/tile
  3. tile 坐标映射回全图坐标
  4. IoU > 0.5 去重（相邻 tile 重叠区的同一人）
  5. 最多加 6 个 person 实体，做 5 档 crop + search_views
  6. `source = "sports_person_proposal"`，`confidence = 0.7`（低于主 prompt 的 0.9）

  **为什么不全局改主 prompt**：
  - 改了会让零售/街景图多出低价值 person 节点（路人、海报背景人影等）
  - 会抬高 `expansion_seeds` 和 pairwise 搜索的噪声，拖慢 Step2
  - 违反"局部问题用局部 targeted fix"的方法论
  - 第二遍只在**第一遍漏掉人物时**才触发，不影响已经正常检测到 person 的图（如 poster、mixed）

  **实测效果**：sports_03（Heat vs Knicks）从 `person=0` → `person=6`（Mitchell Robinson, Duncan Robinson, Tom Thibodeau, Erik Spoelstra, Kyle Lowry, 1 个描述性 player）。这些 person 实体后续会走 Lens → ev_marked → 有机会触发 image_heavy。

  #### 2a-2. 实体 resolution（把 VLM 名字"落地"成真实体 + 拿来源页）

    这一步要解决的是一个具体问题：Step2a VLM 只给我们一个 `name` 字符串（比如 "Jamal Murray"），
    但这个字符串可能是**猜的**（尤其 person / landmark / product 这种视觉实体）。
    Step2a-2 的任务是**用外部 API 验证 name 是否对得上图里的像素**，并顺手抓一批来源页供后续深读。

    按实体类型决定走什么路径：

    ```
    entity.type ∈ {text, brand}        → ocr_likely，只做验证性文字搜图，不做 reverse
    entity.type ∈ {person, landmark,   → image_search_needed，做真 reverse image search
                   product}
    ```

    然后并行跑两条独立的查询，最后再集中 visit 来源页。

  **查询 A：`image_search(text=entity.name)`（所有实体都做）**

    目的：用 VLM 给的名字去 Google Images 搜一遍，看返回的图像标题和来源域名是否和预期对得上。

    调用：`POST https://google.serper.dev/images` with `{"q": entity.name}`
    返回：一组 `{title, image_url, source_url, source_domain}`，写入 `entity.resolution.image_search_titles` 和 `image_search_sources`。

    作用：
  - 验证名字：如果 top 标题完全不提到 `entity.name`，说明 VLM 猜错了，把 `resolve_mode` 从 `ocr_likely` 升级到 `image_search_needed`
  - 收集来源页：`source_url` 列表进入后面的 visit 队列

  **查询 B：`image_search(reverse via Serper Lens)`（仅 person / landmark / product 做）**

    ### 它要解决什么

    查询 A（文字搜图）是用 VLM 给的名字去搜，所以**前提是名字本身是对的**。
    查询 B 不信任 VLM 的名字，直接让 Google Lens 看 crop 图像本身，问"这张小图里的东西是谁/是什么"。

    典型场景：VLM 说这是 "Julian Strawther"，但可能其实是 "Peyton Watson"——名字错了，
    查询 A 就会带着错名字继续搜，错上加错。Lens 看的是像素，不受名字影响。

    ### 为什么要先上传到 litterbox

    `POST https://google.serper.dev/lens` 的请求体是 `{"url": "<public image URL>"}`，
    **它不接受上传本地图片字节**，必须给它一个公网可访问的 HTTP URL。

    但我们手里只有 `entity.search_views.pad20` 这个本地文件路径（如 `output/entities/crops/1/E5_pad20.jpg`），
    Google Lens 的服务器访问不到本地路径。所以需要一个中转步骤：**把本地 crop 推到一个免费公网图床，拿临时 URL**。

    用 `litterbox.catbox.moe` 的原因：
  - 匿名，无需 API key
  - 原生支持 1 小时 / 12 小时 / 24 小时 / 72 小时过期档位（我们用 1 小时，够 Lens 拉一次就够了）
  - 单一 HTTP POST 就能上传，返回一个公网 URL 字符串（如 `https://litter.catbox.moe/abc123.jpg`）
  - 过期自动删除，不需要清理，也不会长期暴露 crop 隐私

    ### 完整流程（以 `E5` = Jamal Murray crop 为例）

    ```
    Step 1: 拿 crop 路径
      entity.search_views.pad20 = "output/entities/crops/1/E5_pad20.jpg"

    Step 2: 上传 litterbox
      POST https://litterbox.catbox.moe/resources/internals/api.php
        form: reqtype=fileupload, time=1h, fileToUpload=<binary>
      → "https://litter.catbox.moe/daqxwy.jpg"

    Step 3: 调 Serper Lens
      POST https://google.serper.dev/lens
        headers: X-API-KEY: <SERPER_KEY>
        body: {"url": "https://litter.catbox.moe/daqxwy.jpg"}

    Step 4: 解析响应
      Serper Lens 的响应结构是：
      {
        "organic": [                ← 主结果列表（类似 web search 的 organic）
          {
            "title":  "Jamal Murray Denver Nuggets Authentic Jersey...",
            "link":   "https://www.ebay.com/...",
            "source": "eBay",
            "thumbnail": "https://..."
          },
          ... (通常 20-60 条)
        ],
        "knowledgeGraph": {         ← 有时会返回，包含 Google 已知的实体信息
          "title": "Jamal Murray",
          "type":  "Basketball player",
          "description": "...",
          "attributes": {...}
        }
      }
    ```

    ### 从响应里具体抽什么

    | 从 Lens 响应的哪里 | 提取出来放进哪里 | 后续怎么用 |
    | --- | --- | --- |
    | `organic[i].title`（top-20） | `entity.resolution.lens_candidate_titles` | 原样留存，供下游 LLM / 诊断溯源 |
    | `organic[i].title`（top-5，经清洗） | 全局 `discovered_entities` 列表 | 见下方时间线 T2 / T4 / T5：清洗成干净实体名 → 并入 `expansion_seeds` → 下一轮 web_search |
    | `organic[i].link`（top-15） | 全局 `visit_tasks` 列表 | 见下方时间线 T2 / T3：被 visit_url 真的读取 → 进 entity.resolution.visited_pages + image_pages |
    | `knowledgeGraph.title` | `entity.resolution.lens_knowledge_graph` | Lens 如果直接认出来，这里是最强的 canonical 名字 |
    | `len(organic)` | `entity.resolution.lens_n_visual` | 用于 resolve_mode 升级判定 |

    > 注：Serper Lens 的响应没有独立的 `visual_matches` 和 `exact_matches` 字段，只有一个 `organic` 列表。
    > 我们在记录时还是用 `lens_n_visual` / `lens_n_exact` 两个字段名是为了兼容 `core/lens.py` 里面向 SerpApi 设计的接口。
    > 当前实现下 `lens_n_exact` 恒为 0，`lens_n_visual = len(organic)`。

    ### Lens 的返回在 Step2 后续代码里被谁消费

    之前这段写"副作用"听起来抽象。换个说法：Lens 返回后，`enrich_image()` 这个函数继续往下跑，
    下面几个步骤会**从 Lens 的返回值里读东西再去做事情**。不是"Lens 写完备注就散会"。

    下面用一个具体的时间线跟着走一遍。场景：Step2 处理 `images/1.jpg`（NBA 球场图），
    实体 E5 是 VLM 提取的 "Jamal Murray"（球员照片）。

    **T0: 实体提取刚结束（2a 末尾）**

    这些变量都是 `enrich_image()` 函数内部的 python 局部变量，不是数据库：

    ```python
    entities = [
        {"id": "E5", "name": "Jamal Murray", "type": "person",
         "search_views": {"pad20": "output/entities/crops/1/E5_pad20.jpg"},
         "resolution": {}},
        ...
    ]
    expansion_seeds = [e for e in entities if e.confidence >= 0.8]
    discovered_entities = []  # 空
    visit_tasks = []           # 空
    triples = []                # 空
    ```

    **T1: 对 E5 调 `core.lens.reverse_search_entity()`**

    ```python
    lens_result = {
        "lens_url": "https://litter.catbox.moe/daqxwy.jpg",
        "candidate_titles": [
            "NWT Jamal Murray Denver Nuggets Authentic Association Jersey Sz 48...",
            "Amazon.com: Jamal Murray Denver Nuggets Autographed Nike 2021 Jersey",
            "Jamal Murray Stats - Basketball-Reference.com",
            ...  # 20 条
        ],
        "source_pages": [
            {"url": "https://www.ebay.com/itm/...", "title": "NWT Jamal Murray...", "domain": "ebay.com"},
            {"url": "https://www.nba.com/player/1627750/jamal_murray", ...},
            ...  # 15 条
        ],
        "n_visual_matches": 60,
    }
    ```

    **T2: `_resolve_entities()` 把 Lens 返回的东西 append 进两个列表**

    这是最关键的一步——Lens 返回的 titles 和 links 被写进 **两个 python 列表变量**，
    后面的代码会直接循环这两个列表去做事情：

    ```python
    # 把 top-5 candidate_titles 加进 discovered_entities
    for title in lens_result["candidate_titles"][:5]:
        discovered_entities.append({
            "name": title,  # 还是脏的，比如 "NWT Jamal Murray Denver Nuggets Jersey Sz 48..."
            "source_entity_id": "E5",
            "provenance": "lens",
        })

    # 把 top-8 source_pages 加进 visit_tasks
    for sp in lens_result["source_pages"][:8]:
        visit_tasks.append(("E5", sp["url"], sp["title"]))

    # 顺手把 lens 原始数据写进 entity.resolution（这一步是给 Step6 溯源用，不驱动后续）
    entities[4]["resolution"]["lens_candidate_titles"] = lens_result["candidate_titles"]
    entities[4]["resolution"]["lens_n_visual"] = 60
    entities[4]["resolution"]["lens_url"] = "https://litter.catbox.moe/daqxwy.jpg"
    ```

    此时状态：

    ```python
    discovered_entities = [
      {"name": "NWT Jamal Murray Denver Nuggets Jersey...", ...},
      {"name": "Amazon.com: Jamal Murray Denver Nuggets...", ...},
      {"name": "Jamal Murray Stats - Basketball-Reference.com", ...},
      ...  # 5 条（脏数据）
    ]
    visit_tasks = [
      ("E5", "https://www.ebay.com/itm/...", "..."),
      ("E5", "https://www.nba.com/player/1627750/jamal_murray", "..."),
      ...  # 8 条
    ]
    ```

    **T3: `_resolve_entities()` 继续往下，用 `visit_tasks` 跑 visit**

    ```python
    # 这个 for 循环读的就是 T2 里 append 进去的列表
    for eid, url, title in visit_tasks:
        resp = visit_url(url, source_stage="lens_source")
        # resp["content"] = "Jamal Murray (born September 23, 1997) is a Canadian
        #                    professional basketball player for the Denver Nuggets..."
        entities[4]["resolution"]["visited_pages"].append({
            "url": url, "title": title, "content": resp["content"][:2000],
        })
        image_pages.append({"url": url, "provenance": "lens_source", ...})
    ```

    Lens 给的 URL **真的被 `visit_url` 调用**，拿到 page 原文。

    **T4: `_resolve_entities()` 末尾做 canonicalize，清洗 `discovered_entities` 并入 `expansion_seeds`**

    ```python
    # 输入 LLM 的是 T2 收集的脏 titles
    raw_titles = [d["name"] for d in discovered_entities]
    # = ["NWT Jamal Murray Denver Nuggets Jersey Sz 48...",
    #    "Amazon.com: Jamal Murray Denver Nuggets Autographed Nike...",
    #    "Jamal Murray Stats - Basketball-Reference.com", ...]

    # LLM canonicalize 输出干净的实体名
    canonical_names = ["Jamal Murray", "Denver Nuggets", "Basketball-Reference.com"]

    # 这 3 个干净的实体名被 append 进 expansion_seeds
    for name in canonical_names:
        if name not in existing_names:
            expansion_seeds.append({"name": name, "source": "lens_reverse", ...})
    ```

    `expansion_seeds` 变长了 3 条。注意这个列表是 `enrich_image()` 的局部变量，后面的代码马上会用。

    **T5: 主流程进 2b-2（确定性搜索），for 循环 `expansion_seeds`**

    ```python
    # expansion_seeds 现在比 2a 结束时长（多了 T4 加的 "Jamal Murray" / "Denver Nuggets" ...）
    for e in expansion_seeds:
        search_result = web_search(e["name"])
        all_search_results.append(search_result)
    ```

    因此 "Jamal Murray" 这个实体会**真的被调 web_search**，结果里可能有身价、队伍、生日这些。
    这些新的 search_result 会被塞进后面的 triple 提取 LLM 里。

    **T6: triple 提取 LLM 的输入比没 Lens 时多**

    给三元组提取 LLM 的 prompt 里包含：
  - 所有实体的 `resolution.visited_pages`（含 T3 visit 到的 Lens source pages 原文）
  - 所有 `expansion_seeds` 的 `all_search_results`（含 T5 对 "Jamal Murray" 等新实体的 web 搜索结果）

    LLM 从这两堆材料里抽出新的三元组，比如：

    ```python
    {"head": "Jamal Murray", "relation": "born_in", "tail": "Kitchener, Canada",
     "source_snippet": "...born September 23, 1997 in Kitchener, Canada..."}
    ```

    **T7: `_mark_retrieval_mode()` 对 triples 打 retrieval_mode 标记**

    对每条新抽出的 triple，检查它的 `source_snippet` 是否在 **T3 读到的 page 文本**里出现、
    且不在 snippet_corpus 里出现。如果是，就打 `retrieval_mode = "page_only_evidenced"`。

    这是下游 Step3 `visit_heavy` bucket 唯一接受的档位（`page_only_semantic` 不算）。

    **T8: Step2 结束，写出 `output/entities/1.json`**

    和没 Lens 跑时的对比：

    | 字段 | 没 Lens 时 | 有 Lens 时 |
    | --- | --- | --- |
    | `entities[4].resolution.lens_*` | 不存在 | 20 条候选 titles + URLs |
    | `expansion_seeds` | 只有 VLM 看到的 in-image 实体 | + "Jamal Murray" / "Denver Nuggets" / ... |
    | `triples` | 只有对 in-image 实体 web_search 抽的 | 多出对新实体搜到的所有 triples |
    | `page_only_evidenced` triples | 接近 0 | 含 Lens source page 里 snippet 没有的 fact |
    | `image_pages[]` | 空 | 8 条真 lens 来源页 |

    ### 一句话总结

    `discovered_entities` 和 `visit_tasks` 是 `enrich_image()` 函数内部的两个 python list。
    Lens 返回后，T2 步往这两个 list 里各 append 了 5-8 条内容。
    **后面的代码会 for 循环这两个 list**（T3 循环 visit_tasks 去 visit_url，T4 清洗 discovered_entities 后 append 进 expansion_seeds，T5 循环 expansion_seeds 去 web_search）。
    所以 Lens 的返回不是"备注"——它变成了后续代码的 for 循环输入，最终变成落地的三元组。

    "改动 Step2 数据状态" 说的就是：因为 Lens 往这些 list 里加了东西，Step2 后续的循环比没 Lens 时**多跑了几圈**，
    多跑的那几圈的结果就是多出来的 triples 和多出来的 `page_only_evidenced` 标签。

    ### resolve_mode 强制升级规则

    `entity.resolve_mode` 默认按**类型规则**确定：
  - `text` / `brand` → `ocr_likely`
  - `person` / `landmark` / `product` → `image_search_needed`

    但类型规则会犯错：比如 VLM 把一个写着文字的 logo 标成 `product`（实际上是 brand），
    那它按规则会被标 `image_search_needed`，浪费工具调用。

    我们加一道**真证据优先**的覆盖规则：

    ```
    if lens_n_visual > 0 or lens_n_exact > 0:
        entity.resolve_mode = "image_search_needed"
        entity.resolve_mode_evidence = "lens_visual_matches"
    ```

    也就是只要 Lens 真的为这个 crop 返回了任何视觉匹配，就认为它"不是靠 OCR 就能认的"，强制升级。
    反之如果 Lens 返回 0 个匹配，就退回去信类型规则。
    这里的逻辑是：**真跑过的 API 结果比静态类型规则更可信**，如果现实里 Google Lens 都认不出来，那这个 crop 基本不可能靠纯文字 OCR 识别。

  **查询 C：visit 来源页（Phase A + Phase B 的 URL 汇总后并行深读）**

    `visit_tasks` 收齐 A 和 B 拿到的所有 source page URL（去重后最多 8 条），用 `visit_url` 并发读取：

  - Jina Reader 付费模式优先（带 `JINA_API_KEY`）
  - **402 余额耗尽时自动降级到免费模式**（canary 验证 80% URL 能读成功）
  - 每次调用打审计日志：`source_stage / status_code / elapsed_ms / via=paid|free`

    成功读到的页面内容写入 `entity.resolution.visited_pages`，
    同时按来源分别进 `image_pages` / `web_pages` 一等字段（供 Step6 按 provenance 归因）。

  **最后产出 `entity.resolution` 的字段清单：**

  | 字段 | 来源 | 用途 |
  | --- | --- | --- |
  | `image_search_titles` | 查询 A | VLM 名字的对齐信号 |
  | `image_search_sources` | 查询 A | 候选来源页 URL 列表 |
  | `lens_provenance` | 查询 B | `"lens_reverse"` 或 `"vlm_describe_workaround"`（fallback） |
  | `lens_url` | 查询 B | litterbox 上的临时 URL（1h 后失效，仅用于调试溯源） |
  | `lens_candidate_titles` | 查询 B | Lens organic 抽的 top-20 候选 |
  | `lens_n_visual` / `lens_n_exact` | 查询 B | 匹配计数（>0 → 升级 resolve_mode） |
  | `lens_knowledge_graph` | 查询 B | Lens 返回的 KG 实体（如有） |
  | `reverse_candidates` | 查询 B | 复用字段，存 top-5 候选供下游 LLM 消费 |
  | `visited_pages` | 查询 C | 实际读到的页面内容（title / content_preview） |
  | `resolve_mode_evidence` | 综合 | `"lens_visual_matches"` 表示 lens 真命中过 |

  **依赖**：只需 `SERPER_KEY`（一个 KEY 同时驱动 web_search / image_search(text) / image_search(reverse via Lens)）+ `JINA_API_KEY`（可选，免费模式 fallback 可用）。不需要 SerpApi / 独立图床账号 / Bing Visual Search。

  **fallback 策略**：
  - `SERPER_KEY` 不存在或余额耗尽 → 查询 A + 查询 B 全部失败；Phase B 降级到旧的"VLM 描述 crop → 文字搜图"workaround
  - `JINA_API_KEY` 余额耗尽 → Jina Reader 降级到免费模式（有速率限制但能读）
  - 两个都挂 → resolution 能产出 `lens_candidate_titles` 只要 Serper Lens 还在，最差情况下 `visited_pages=0` / `image_pages=0`，Step3 降级为纯 snippet 路径

  **resolve_mode 最终标记规则**：

  | 条件 | resolve_mode |
  | --- | --- |
  | type ∈ {text, brand} 且 `image_search_titles` 里包含 `entity.name` | `ocr_likely` |
  | type ∈ {text, brand} 但 `image_search_titles` 里完全找不到 `entity.name` | 升级为 `image_search_needed` |
  | type ∈ {person, landmark, product} | 默认 `image_search_needed` |
  | `lens_n_visual > 0` 或 `lens_n_exact > 0`（Lens 真命中）| 强制 `image_search_needed` + `resolve_mode_evidence=lens_visual_matches` |

    `ocr_likely` → Step3 不加任何工具步骤（VLM 直接视读）
    `image_search_needed` → Step3 工具编译在题目开头加一步 `image_search`

  #### 2b-0. 实体池分层 + 跨实体关联发现

  **实体池分两层：**

  - **core_anchors**：confidence >= 0.8，给 Step3 出题优先级用
  - **expansion_seeds**：所有 searchable 实体（排除太泛的 object），不限数量，给 Step2 扩图用

  **跨实体关联发现：**

  1. 对 expansion_seeds 两两配对（上限 20 对，按 entity type + confidence 优先排序）
  2. 搜索 `"{A} {B}"`（实体名拼接）
  3. 并行 web_search + 每对 visit 深读 top-2 页面
  4. LLM 提取桥接三元组，标记 `provenance="cross_entity"`

  #### 2b-1. 确定性搜索

    不让 LLM 写搜索词。由 `_deterministic_search_queries` 代码生成：

    ```
    主策略：web_search(entity_name)
    fallback：web_search(entity_name + type_alias)
      brand → "brand company"
      product → "product"
      landmark → "landmark building"
      person → "person biography"
    ```

    省掉一次 LLM 调用。

  #### 2b-2. 执行搜索（web_search + visit）

    对每个 expansion_seed 并行执行：

  - **web_search(name)**：返回摘要、Knowledge Graph、搜索结果
  - **web_search(name + type_alias)**：类型限定搜索
  - **visit（Jina Reader）**：深读 top 搜索结果页面（最多 5 篇/实体，排除社交媒体）

    注意：Jina Reader 使用独立 HTTP client（`trust_env=True`），不能用搜索用的 `trust_env=False` client，否则 HTTPS 握手失败。

  #### 2b-0.5. Fan-out 分层预算（Promotion Gate）

  **问题**：person proposal + Lens 发现的新实体一进图就拿到和主实体同级搜索预算（跨实体配对 + 多轮扩展 + Jina visit），导致实体 15+ 时 Step2 膨胀到 10-60 分钟。

  **方案**：三层预算池 + web-first visit-later + deficit-driven 扩展

  **三层实体池**：

  | 层级 | 来源 | 预算 |
  | --- | --- | --- |
  | **tier1** (full) | 原始 VLM 检测 + high confidence | 跨实体配对 + 主搜索 + 扩展 |
  | **tier2** (limited) | Lens 证实的 proposal / discovered（最多 3 个 proposal 升格） | 主搜索 + Lens，不参与 pairwise |
  | **demoted** | 其余 discovered / 低信度 proposal | 0 搜索预算，但保留在图中供 Step3 |

  **跨实体配对分组预算**：总上限 10 对，只用 tier1 实体

  **web-first, visit-later**：跨实体搜索先 `web_search("{A} {B}")`，只有 snippet 同时提到 A、B 时才开 Jina visit（节省 ~60% 跨实体 Jina 调用）

  **扩展轮数**：默认 3 轮 + 1 轮 deficit-driven reopen（closure richness 差 1 项达标时触发）

  **实测效果**：

  | 图片 | 实体 | 改前 | 一刀切 | 分层预算 | L3 变化 |
  | --- | --- | --- | --- | --- | --- |
  | poster_03 | 6 | 356s | 314s | **314s** | 4→6 ✓ |
  | mixed_06 | 15→12 | 418s | 326s (L3=0!) | **378s** | 2→2 ✓ |
  | retail_06 ★ | 15 | 821s | — | **1796s** | 5→4 ⚠️ |

  **教训**：retail_06（15 个 brand，全是 vlm_only → 全 promote 进 tier1）说明 promotion gate 对**全 brand 图无效**——VLM 直检的实体默认就是 tier1，没有东西被 demote。这类图需要额外的 entity-count-based budget cap（如 tier1 > 12 时只取 top-12 by confidence），或者限制主搜索的 per-entity visit 数量。**deficit reopen 已加 ≤10 实体限制**防止大图触发额外扩展轮。

  #### 2b-2.5. API 调用缓存层（disk-based memoization）

  **问题**：39 张 stress suite + 4 张 sku 的一次完整 Step2 消耗 ~2300 Serper credits（新注册免费额度 2500），加上 Step5 攻击器调用，一个 session 就吃干额度。

  **方案**：给 4 个外部 API 加 disk cache，同参数请求直接返回缓存，0 网络开销：

  | API | 缓存目录 | cache key | 实测加速 |
  | --- | --- | --- | --- |
  | `web_search` | `output/.cache/serper_web/` | sha256(q + num) | **4000x** |
  | `image_text_search` | `output/.cache/serper_img/` | sha256(q + num) | 同上 |
  | `serper_google_lens` | `output/.cache/serper_lens/` | sha256(image_url) | 同上 |
  | `visit_url` (Jina) | `output/.cache/jina/` | sha256(url + max_chars) | 同上 |

  - 只缓存 **成功响应**（不缓存 error / 400 / 402）
  - 环境变量 `DISABLE_API_CACHE=1` 可跳过缓存（强制真调）
  - cache key 不含 API key → 换 key 后缓存仍有效
  - Lens 的 litterbox URL 是 1h 临时链接，同 session 内缓存有效，跨 session 自动 miss

  **Serper 消耗大头分解**（39+4 图一轮 Step2）：

  | 来源 | credits | 占比 |
  | --- | --- | --- |
  | web_search（主搜 + 扩展搜索 6 轮） | 1534 | 66% ★ |
  | image_text_search（每实体 1 次） | 482 | 21% |
  | Step5 attacker agent 工具调用 | ~186 | 8% |
  | Lens reverse | 70 | 3% |

  **缓存对 rerun 的影响**：同一张图清 checkpoint 后重跑 Step2，已缓存的 web_search/image_search 请求直接命中 → **Step2 rerun 耗时从 5 min/图降到 ~1 min/图**（主要节省 Serper 网络延迟）。

  #### 2b-3. 三元组提取

    将搜索结果 + resolution 信息交给 LLM 一次性提取事实三元组：

    ```json
    {
      "head": "头实体名",
      "relation": "关系类型",
      "tail": "尾实体名",
      "tail_type": "TIME / QUANTITY / LOCATION / PERSON / ORG / OTHER",
      "normalized_value": "标准化值",
      "unit": "单位",
      "fact": "一句话描述",
      "source_snippet": "佐证原文",
      "provenance": "text_exact / text_rewrite / cross_entity"
    }
    ```

    entities_summary 中包含 resolution 信息（image_search 图片标题、来源页内容摘要），帮助 LLM 更准确地提取事实。

  **关键规则：**

  - 每个三元组必须有搜索结果佐证
  - 图中实体名使用原始名称，禁止加后缀变体
  - 不仅提取「图中实体→外部」，也提取「外部→外部」（多跳链路）

  #### 2b-3.5. 合并桥接三元组 + 实体名规范化 + LLM 别名归一

  - 跨实体桥接三元组按 `(head, relation, tail)` 去重合并
  - 实体名规范化：漂移变体统一到 canonical name
  - 自环丢弃
  - **2b-5b LLM 别名归一**（`_llm_canonicalize_aliases`）：启发式 substring match 后，再用一次 LLM call 处理 substring 管不住的同义变体（如 "Nuggets" / "Denver Nuggets" / "the Denver Nuggets"）。LLM 返回别名分组，脚本按分组重写 triples head/tail + 删除 entities 里的别名项 + 去重。在空间兜底之前跑，因为空间兜底会用干净的实体名建连接。失败 graceful fallback。

  #### 2b-4. 多轮扩展搜索

  **停止条件（closure richness）：**

    每一轮结束后检查 6 个 hard motif 指标，满足任意 3 项即停，最多 `MAX_EXTEND_ROUNDS=6` 轮：

  - `image_resolved_anchors >= 3`：被 image_search 锁定身份的 in-image 锚点数
  - `page_only_facts >= 6`：只能从深读页面拿的 fact（relation 硬 override + chunk/tail 证据混合判定）
  - `compare_ready_pairs >= 4`：两个不同 entity 有同类 TIME/QUANTITY fact 且 relation 兼容
  - `rank_ready_triplets >= 2`：三个及以上 entity 有同类可排序 fact
  - `cross_anchor_shared_nodes >= 3`：不同锚点共享的 fact 尾节点（set_merge motif 基础）
  - `total_triples >= 40`：最低 triples 总数兜底

    这套指标替代旧的 `graph richness`（triples≥30 / entities≥15 类堆量指标），目标从"造很多三元组"改成"造很多可闭合的 hard motif"。

  **扩展 frontier：**
  - 未搜过的 tail entities（按引用次数排序）
  - round ≥2 **放开了 in-image tail 过滤**，Denver Nuggets 这类 in-image 桥接实体也会作为 seed 做二跳扩展
  - exact-name-first 搜索，不调 LLM

  **空间关系兜底：**
  - 扩展完成后，为无知识连接的实体对补充空间三元组（`located_above / left_of / right_of` 等）
  - 空间兜底后**再跑一次 `_mark_retrieval_mode`**，把新加的空间 triples 标为 `"spatial"`（不参与 page_only 比例分母）

  ### 输出

  - `output/entities/{img_id}.json`：
    - `entities`：完整实体列表（含 search_views、resolution 信息）
    - `core_anchors`：Step3 出题用的高置信锚点
    - `expansion_seeds`：Step2 扩图用的全量实体
    - `triples`：fact 三元组
    - `search_results`：web_search 结果
    - `image_description`、`domain`
  - `output/entities/crops/{img_id}/`：5 档裁剪图（E1.jpg / E1_pad20.jpg / E1_pad40.jpg / E1_context.jpg / full_scene.jpg）


  ## 第三步：分层问题生成（弱约束随机游走 + 后验归纳题型）

    实现文件：
  - `step3_generate.py`（薄壳入口，437 行）：兼容老 API，保留 `_sanitize_triples` / `_build_nx_graph` 等工具
  - `step3_trajectory.py`（核心生题引擎，2000+ 行）：HeteroSolveGraph + walker + closure 枚举 + frame 编译
  - `step3_graphenv_runtime.py`（工具库）：relation natural text、askability profile 等

    > 注：原 `experimental/random_walk_step3/` 文件夹已删除，所有逻辑迁到根目录。

    这是整个 Pipeline 最核心的步骤。**题型是游走的输出，不是输入。**

    核心流程：
    1. 从 step2 输出构建**异构求解图**（4 类节点 × 3 类边，带 resolve_mode / retrieval_mode）
       - **桥接提升**：把 triple 中非 in-image 的 head 也提升为 entity 节点（标记 `synthetic=True`），但不创建 region，不让 walker spawn 选中。这让 multi_hop 能跨多跳走出 in-image 实体之外
    2. **弱约束自由游走**：从视觉锚点出发，逐步扩展证据子图（不是线性链）
       - HARD 难度下，walker 偏好 image_search_needed 锚点 + page_only 边 + bridge affordance（这条边的 tail 恰好是另一个 entity 的名字）
    3. **后验归纳题型**：枚举子图中所有可闭合问题意图，选最佳
    4. **5+1 个不可约性检查**（含 multi_hop 3-hop shortcut 检查）
    5. **6 个 hard bucket** 分类 + 全局配额选题
    6. 编译成 **QuestionFrame**，单次 LLM 调用语言化（按题型给好/坏例子引导）
    7. **确定性工具序列**编译（根据 resolve_mode / retrieval_mode 决定工具）

  ### 3a. 异构求解图（HeteroSolveGraph）

    从 step2 entity JSON 构建。
    
    **节点 4 类：**

  | 类型 | 含义 | 来源 |
  | ---- | ---- | ---- |
  | `full_image` | 整图（1 个） | — |
  | `region` | 图中实体裁剪区 | step2 entities（bbox / location / type / search_views） |
  | `entity` | 实体 canonical name（含 in-image + 桥接合成） | step2 entities + triples 的 head |
  | `fact` | 知识事实值 | step2 triples 的 tail |

    **桥接实体（synthetic）：** triples 里出现的 head 如果不在 in-image 实体列表里（如 round 2+ 搜出来的 "Memphis Grizzlies"、"Thriving Brands LLC"），不再丢弃；先尝试 canonical 折叠到已有实体（避免 "Nuggets" / "Denver Nuggets" 别名重复），折叠不上则创建标记 `synthetic=True, in_image=False` 的 entity 节点。这些节点：
  - **不会**被 walker 当作 spawn 锚点（没有 region）
  - **可以**作为 multi_hop 的中间桥接跳
  - 让 3-hop 链能跨过 out-of-image 实体（image 1 增加 9 个 synthetic，sku_8845 增加 22 个）

    **边 3 类：**

  | 类型 | 含义 | 关键属性 |
  | ---- | ---- | ------- |
  | `observe` | full_image → region | — |
  | `resolve` | region → entity | **resolve_mode**：`ocr_likely` 或 `image_search_needed` |
  | `retrieve` | entity → fact | **retrieval_mode**：`snippet_only` / `page_only` / `spatial`（空间关系兜底，不算知识 fact）+ askability / lexicalizability |

    `resolve_mode` 决定工具编译时是否需要 image_search（ocr_likely → 直接视读，无工具步骤；image_search_needed → +1 image_search 步骤）。
    `retrieval_mode` 决定是否需要 visit 深读（page_only 的 fact 必须 visit 才能获取）。
    
    **预缓存查表 `_name_to_entity_canon_map`**：构图时建立 `canonical_name → entity_key` 映射，供 walker scoring（bridge affordance）和 multi_hop 枚举（找桥接点）复用。

  ### 3b. 弱约束自由游走（SubgraphWalker）

    游走结果不是链，而是逐步长大的**证据子图（DAG）**。
    
    **状态**：`subgraph` / `frontier` / `used_anchors` / `steps_taken`
    
    **三种动作**（统一 softmax 采样）：`expand` / `spawn_anchor` / `stop`
    
    **单一打分函数**（不同难度只是权重不同）：
    
    ```
    score(move) = w_visual_novelty × 引入新锚点
               + w_fact_gain × 引入新可问事实
               + w_compute_affordance × 形成可比较/排序的值
               + w_branch_novelty × 形成新分支
               + w_closure_gain × 离"可问"更近
               - w_shortcut_penalty × 答案太容易直达
               - w_redundancy × 凑数步骤
               - w_generic_penalty × 答案太泛
    ```
    
    **HARD 难度下的额外加分项（让 walker 偏好长链 closure）：**

  | 信号 | 条件 | 加分 |
  | ---- | ---- | ---- |
  | bridge affordance | 这条 retrieve 边的 tail value 是图里另一个 entity 的名字（multi_hop 桥接点） | +1.5 |
  | page_only 边奖励 | edge 的 retrieval_mode == "page_only"（visit 必需） | +1.0 |
  | image_search_needed 锚点 | spawn 的 region 对应实体需要 image_search resolve | +1.0 |
  | 合成源折扣 | move.src 是 synthetic 实体（避免被合成实体的大扇出淹没） | × 0.5 |

    **STOP 的硬预算检查（hard 难度下）：**
    
    每个 budget 缺口 -3 分。缺口越多，walker 越不愿意停：

  | 预算项 | 条件 | 缺口惩罚 |
  | ------ | ---- | ------- |
  | image_search | 子图中无 `resolve_mode=image_search_needed` 的实体 | -3 |
  | visit | 子图中无 `retrieval_mode=page_only` 的事实 | -3 |
  | 多锚点 | `used_anchors < 2` | -3 |
  | compute | 无 compare/rank/set_merge closure | -3 |

    这样 hard walker 必须凑齐足够多的工具原材料才会停下。

  ### 3c. 后验归纳题型（ClosureCompiler）

    游走过程中和停止时，枚举当前子图中所有可闭合的问题意图：

  | 闭合类型 | 条件 | Level |
  | -------- | ---- | ----- |
  | `read` | region 实体可直接识别（仅 ocr_likely） | L1 |
  | `lookup` | region → entity → fact 链路完整 | L2 |
  | `compare` | 两个 entity 有同类 TIME/QUANTITY fact | L3 |
  | `compare_then_follow` | compare + 赢家还有别的 fact | L3 |
  | `rank` | 3+ 个 entity 有同类可排序 fact | L3 |
  | `set_merge` | 两个 entity 共享同一 fact target value（用 canonical 比较，非 fact_key 比较） | L3 |
  | `multi_hop` (2-hop) | A → rel1 → bridge → rel2 → answer，其中 bridge 在图里是另一个 entity | L3 |
  | `multi_hop` (3-hop) | A → rel1 → B → rel2 → C → rel3 → answer，B 和 C 都是图里的 entity（含 synthetic） | L3 |

    **multi_hop 枚举关键约束：**
  - 锚点（hop 0）必须是 in-image entity（不能是 synthetic）
  - 中间桥实体（hop 1, 2）可以是 synthetic 或 in-image
  - 中间跳的 tail_type 必须是 entity-like（PERSON/ORG/LOCATION/OTHER），**不能是 TIME/QUANTITY**（避免 player→team→founded_year→数值这种语义垃圾链）
  - 任意两个 hop 实体的 canonical 名字不能相等（防 alias 坍缩）
  - 最终答案不能回指到链上任何 entity（防循环）
  - 3-hop 每锚点 cap 8 条（防组合爆炸）

    **L1 read 仅在 resolve_mode=ocr_likely 时生成**：image_search_needed 的 region 不再产 L1 read（避免 read 题里塞 image_search 工具）。

  ### 3d. 不可约性检查

  **基础 5 项**（适用所有 closure）：

  1. **answer_uniqueness** — 非 yes/no/unknown/太泛
  2. **realizable_question** — 锚点有视觉描述
  3. **no_python_shortcut** — L3 去掉 compute 后答案必须变
  4. **answer_not_visible** — 答案不是图中直接可见的实体名
  5. **no_branch_shortcut** — 删掉任一分支后答案必须变

  **multi_hop 严格 shortcut 检查（3 道独立门）**：

  6. **multi_hop_direct_shortcut** — 锚点 eA 不能有直达 final_val 的 retrieve 边（否则一跳就够）
  7. **multi_hop_skip_bridge** (3-hop only) — eA 不能有直达 eC（最后桥）的 retrieve 边（否则可以跳过 eB）
  8. **multi_hop_hop2_direct_shortcut** (3-hop only) — eB 不能有直达 final_val 的 retrieve 边（否则 hop2+hop3 可合并为 1 跳）

    这三道门保证 3-hop 链是真的需要 3 跳才能解，不是被 walker 凑出来的伪链。

  ### 3e. 6 个 Hard Bucket + 全局配额

    每个 L3 closure 编译工具序列后，打 **hard bucket** 标签：

  | Bucket | 条件 | 优先级 |
  | ------ | ---- | ------ |
  | `all_tools` | 4 种工具全部出现 + 非平凡 code skill ≥ 2 | 1（最高） |
  | `image_heavy` | 有 image_search + (≥2 视觉锚点 **或** ≥1 锚点 + follow≥2) | 2 |
  | `visit_heavy` | 包含 visit + 至少一个 retrieval_mode=page_only 的 fact | 3 |
  | `code_heavy` | ≥2 类不同非平凡 code skill | 4 |
  | `ultra_long` | tool_plan ≥ 7 步 + 工具种类 ≥ 2（**但仅当以上工具类 bucket 都不命中时**） | 5 |
  | `chain_heavy` | multi_hop 裸链（无 visit / image_search） | 6 |
  | `standard` | 以上都不满足 | 7 |

  **关键变更（数据驱动 targeted fix）**：
  - **ultra_long 从"最高优先级"降为"次级长度标签"**——诊断数据证明旧 ultra_long 优先级抢走了 8 个含 image_search 的闭合（sports 8 图诊断）。现在 `is_ultra_long` 只作为 flag 记录，不再作为主 bucket
  - **image_heavy 拆两个 subtype**（对外仍是同一个 bucket，内部记录 `hard_bucket_subtype`）：

  | Subtype | 条件 | 典型场景 |
  | --- | --- | --- |
  | `image_compare` | ≥2 visual anchors + image_search（保留旧定义） | 两个锚点分别 resolve → 比较属性 |
  | `image_resolve_follow` | **≥1 visual anchor + image_search + follow chain ≥2** | person → team → arena → capacity |

  `image_resolve_follow` 不是 sports 特判——它对 person/poster/landmark/product 都成立。诊断实测：sports 8 图 `image_heavy: 0 → 9`（5 compare + 4 resolve_follow）。

  **Step5 攻击调度同步更新**：A1 不再纯查 bucket 表，改用 `_compute_a1_attacks(q)` 函数按 tool signature + is_ultra_long flag 动态决定。image_heavy + is_ultra_long 的题会先跑 `no_image_search`，再补跑 `no_visit`/`no_code`（如果 tool_sequence 里有这些工具）。
    
    **Bucket 是硬门槛，不只是标签。** 入选某个 bucket 的 closure 必须通过 `check_tool_irreducibility`：

  - `no_image_search_shortcut`：所有锚点都是 ocr_likely → image_search 可被 OCR 替代 → **降级到 standard**
  - `no_visit_shortcut`：所有 fact 都是 snippet_only → visit 多余 → **降级到 standard**
  - `visit_heavy_prior_answerable`：最终 hop 的 relation 是强先验类型（headquartered_in / founded_by / born_in / owner / capital 等 25 种），且 head entity 是命名实体 → **模型靠先验就能答，降级到 standard**（Tier B 实测 visit_heavy × no_tool = 44% breach 驱动的 targeted fix）
  - `no_code_shortcut`：code 只有 crop/OCR，无 compute/set_op → **降级到 standard**
  - `ultra_long_too_short` / `ultra_long_not_diverse`：ultra_long 必须真的 ≥7 步且工具种类 ≥3（避免 7 个 web_search 凑数）

    不通过的 closure 不会被 reject，而是降级到 standard bucket，仍可作为普通 L3 题。

    **全局选题按 bucket 配额**（优先 L3，再 L2，最后 L1）：

  | Level | 配额 |
  | ----- | ---- |
  | L3 image_heavy | 2 题 |
  | L3 visit_heavy | 1 题 |
  | L3 code_heavy | 1 题 |
  | L3 all_tools | 1 题 |
  | L3 chain_heavy | 2 题 |
  | L3 ultra_long | 2 题 |
  | L3 standard | 1 题 |
  | L2 | 3 题 |
  | L1 | 4 题 |

  L3 总额上限 8 题（`max_per_level[3] = 8`）。

  ### 3f. QuestionFrame + 语言化

    不把原始子图丢给 LLM。先编译成结构化 QuestionFrame，再单次 LLM 调用语言化。
    
    **按题型给引导和好/坏例子：**

  - **L1 read**：问"XX上写的是什么文字/品牌"，不要只说"XX是什么"
  - **L2 lookup**：先用视觉特征描述，再用日常口语问需要查资料的问题。不堆砌形容词。
  - **L3 compare**：用外观差异区分两个目标（球衣号码/颜色），问法日常化（"年龄更大"而非"出生日期更早"）
  - **L3 rank**：简洁列出几个目标，问谁最XX
  - **L3 set_merge**：问两个目标的共同点
  - **L3 multi_hop (2-hop)**：用 "这个XX的YY的ZZ" 结构串联两跳，中间节点不写名字。"画面里穿白色27号球衣那位球员，他效力的那支球队的主场所在城市人口是多少？"
  - **L3 multi_hop (3-hop)**：代词叠加："那位的...的...的..."。"画面里穿白色8号球衣那位球员，他所在的那支球队的母公司，曾经收购过的那家公司总部位于哪座城市？"

    **QuestionFrame 隐藏列表（multi_hop）**：
  - `hidden_entities`：所有 hop 的 entity 名（锚点 + 所有桥）
  - `hidden_values`：所有中间 hop 的 value（除最终答案外的桥接字面值）
  - `criterion`：所有中间跳的 relation 用 "的" 连接（"所在球队的母公司"）
  - `follow_relation`：最后一跳的 relation（最终问的属性）

    **Postcheck `postcheck_name_leak`**：
  - 检查所有 hop entity 名字没出现在问题里
  - 检查所有中间 bridge value（不含最终答案）没出现在问题里
  - 检查任何 in-image entity 名都没泄露

  ### 3g. 工具序列（确定性编译）

    从 resolve_mode / retrieval_mode **确定性**生成工具序列：

  | 条件 | 工具 |
  | ---- | ---- |
  | resolve_mode=ocr_likely | （无步骤，VLM 直接视读） |
  | resolve_mode=image_search_needed | **image_search** 1 步（不再加 crop code 步骤） |
  | retrieve, retrieval_mode=snippet_only | web_search |
  | retrieve, retrieval_mode=page_only | web_search + **visit** |
  | compute (compare/rank) | code_interpreter（normalize + compute 共 2 步） |
  | set_merge merge | code_interpreter（vote merge + intersection 共 2 步） |

    **重要简化：**
  - L1 read 的 tool_plan 是空数组 `[]`（VLM 直接看图）
  - 不再为每个 region 加 "裁剪 bbox" 的 code 步骤（视为隐式）
  - resolve 步骤只在 image_search_needed 时加 1 步 image_search

    **multi_hop 的 tool_plan 编译（泛化 N-hop 循环）：**

    ```
    [可选] image_search（锚点 resolve）
    for hop in hop_chain:
        web_search（每跳一次）
        [可选] visit（如果该 hop 的 retrieval_mode=page_only）
    ```

    各 N-hop 长度示例：
  - 2-hop snippet_only 锚点 ocr_likely：`web + web` = **2 步**
  - 2-hop snippet_only 锚点 image_search：`image_search + web + web` = **3 步**
  - 3-hop snippet_only 锚点 ocr_likely：`web + web + web` = **3 步**
  - 3-hop snippet_only 锚点 image_search：`image_search + web + web + web` = **4 步**
  - 3-hop **all page_only** 锚点 image_search：`image_search + web + visit + web + visit + web + visit` = **7 步** ✓ ultra_long

    工具序列的多样性由 Step2 的 retrieval_mode / resolve_mode 标记决定，不是硬编码。

  ### 输出

  - `output/questions/{img_id}.json`：L1/L2/L3 题目，每题含 hard_bucket 标签
  - 每题包含：question / answer / tool_sequence / level / family / hard_bucket / reasoning_path

  ### 性能 Profile：为什么 Pipeline 慢？Step3 在哪里花时间？

  **实测数据（5 张图 benchmark，分层预算 + disk cache，Gemini 3 Flash 后端）：**

  | 图片 | 实体 | Step2 | Step3 | 总耗时 | L3 |
  | --- | --- | --- | --- | --- | --- |
  | poster_03 (电影海报) | 6 | 272s | 41s | **314s (5.2min)** | 6 |
  | mixed_06 (NBA 球馆) | 12 | 361s | 17s | **378s (6.3min)** | 2 |
  | sports_01 (NBA 赛场) | 14 | 355s | 15s | **370s (6.2min)** | 1 |
  | street_07 (罗马广场) | 6 | 615s | 20s | **635s (10.6min)** | 2 |
  | poster_05 (电影海报) | 10 | 328s | 17s | **344s (5.7min)** | 4 |
  | retail_06 ★异常 | 15 | 1796s | 24s | **1820s (30min)** | 4 |

  **典型耗时：5-10 min/张，产出 9 题/张（L1=3-4, L2=3, L3=1-6）**
  **异常 case**：实体 15+ 的 retail 图 Step2 可达 30 min（Jina visit 累积 + 扩展轮数）

  **结论：Step3 不是瓶颈（17-41s/张），Step2 的网络 I/O（Serper + Jina + LLM）才是。**

  #### Step2 为什么慢

  Step2 的时间几乎全花在**网络 I/O**上，不是 CPU：

  1. **Serper API 调用**：只对 person/landmark/product 做 image_text_search + 全部走 Lens。promoted 实体 × 2-3 次搜索 = ~20 次网络请求
  2. **Jina Reader 深读**：visit_tasks 队列混合 3 路来源。**跨实体阶段已改为 web-first visit-later**（snippet 没同时提到两个实体 → 跳过 visit），但主搜索和扩展阶段的 visit 仍然多
  3. **LLM 调用**：三元组提取 + 搜索词生成 + 别名归一 + person proposal（4 tile × 1 call）
  4. **扩展轮数**：默认 3 轮 + 1 轮 deficit reopen（仅 ≤10 实体的图触发）
  5. **极端 case**：15+ 实体的 retail/mixed 图，即使有 promotion gate，tier1 全是 brand（vlm_only 来源全 promote），导致搜索量仍然大

  **Step2 的时间构成（估算）：**
  ```
  Serper API           ~35%   ← promoted 实体搜索 + 扩展
  Jina Reader          ~30%   ← 主搜 visit + 跨实体 visit (web-first 后已减)
  LLM calls            ~25%   ← 三元组提取 + 别名 + person proposal tile
  Python/图操作        ~10%   ← crop 生成、_mark_retrieval_mode、dedup
  ```

  **已实施的优化**（vs 初始版本，对中小图省 10-20%）：
  - Promotion gate：discovered/proposal 实体不自动拿 full-budget
  - 跨实体 web-first visit-later：snippet 不提到双方 → 跳过 Jina（省 ~60% 跨实体 visit）
  - 跨实体配对上限 20 → 10
  - image_text_search 只对 person/landmark/product（省 ~70% 图搜）
  - 扩展轮数 6 → 3 + deficit reopen（仅 ≤10 实体）
  - Disk cache（rerun 时 4000x 加速）

  **待做的优化**（对大图收益更大）：
  - 主搜索 per-entity visit 上限从 5 降到 2（当前最大 Jina 消耗源）
  - Jina 并发超时 + 失败快速跳过（当前 402 fallback 是双倍延迟）
  - 全链路 memoization（含 Step5 agent tool result）

  #### Step3 为什么快（23s/张）

  Step3 大部分工作是**纯 CPU 计算**（图遍历 + softmax 采样 + 枚举），唯一的网络 I/O 是 LLM 问题语言化。

  **Step3 内部 7 个阶段的耗时分解（单张图 ~23s）：**

  | 阶段 | 耗时估算 | 计算类型 | 说明 |
  | --- | --- | --- | --- |
  | 1. 构建 HeteroSolveGraph | ~100ms | 纯 CPU | 遍历 entities + triples，建 4 类节点 + 3 类边 |
  | 2. 24 次随机游走 | ~3-5s | 纯 CPU | 每步要对所有 frontier 候选算 score（8 维向量）→ softmax → 采样。n_walks=24 × ~10 步/walk × ~10 候选/步 = ~2400 次 score 计算 |
  | 3. 闭合枚举 | ~2-4s | 纯 CPU | 每个 walk 停止后枚举 compare / rank / set_merge / multi_hop。**multi_hop 3-hop 是最贵的**：对每个 in-image anchor 做 3 层嵌套循环扫桥接点。有 cap（每锚点 8 条 3-hop）防止组合爆炸 |
  | 4. 候选去重 + 排序 | ~100ms | 纯 CPU | 按 score 降序 |
  | 5. 不可约性检查 | ~1-2s | 纯 CPU | 每个 L3 候选做 5+3 道检查（answer_uniqueness / shortcut / branch_shortcut / tool_irreducibility），check 次数 ∝ L3 候选数 |
  | 6. **LLM 语言化** | **~12-18s** | **网络 I/O** | 8-10 道选中题 × 1 次 call_vlm_json (1.5-3s/次)。**这是 Step3 唯一的慢步**。用 ThreadPoolExecutor 并发跑，4-6 worker |
  | 7. Postcheck + 输出 | ~100ms | 纯 CPU | name_leak 检查 + JSON 写盘 |

  **Step3 真正的时间在哪：**
  ```
  LLM 语言化           ~60-70%  ← 唯一网络步，被 ThreadPool 并发加速
  24 walks + 闭合枚举   ~20-25%  ← 纯 CPU，O(n_walks × n_steps × n_candidates)
  不可约性检查           ~5-10%  ← 纯 CPU
  其他（构图/写盘）       ~5%
  ```

  **如果要加速 Step3：**
  - 降 `n_walks`（目前 24，降到 12 约减半时间，但题目多样性下降）
  - 增 LLM 语言化并发数（目前 4-6 worker，受限于 VLM endpoint 容量）
  - 不建议减不可约性检查（它是 hard 质量的核心保障）

  #### Step5 为什么慢

  - **Tier A**（Qwen3-VL-30B）：71 min / 136 L3。Phase A0 每题 1 步 × 4 worker = 20 min；Phase A1 每题 4-6 步 × 2 worker + Serper/Jina 工具调用 = 50 min
  - **Tier B**（Gemini 3 Flash）：更慢，因为 Gemini endpoint 尾延迟比 Qwen 高 2-3x。full_run 超时卡死（killed after 2.5h），targeted 版本 18 题 22 min
  - **根因**：每次 attack 是一个完整的 agent loop（多轮 LLM + 工具），每轮都有 HTTP 往返。并发受限于 endpoint 容量和 Serper/Jina 限速

  #### 真正的 Pipeline 瓶颈图

  ```
  Step1 (1 min) → Step2 (48 min ★) → Step3 (3 min) → Step3b (5 min) → Step5 A (71 min ★) → Step5 B (75 min ★) → Step6 (<1s)
                      ↑                                                       ↑
                 网络 I/O bound                                          模型 latency bound
                 (Serper + Jina + LLM)                                   (agent loop × HTTP)
  ```

  **优化建议（未实施）：**
  - Step2：预缓存 Serper 结果（同一 entity 重跑时不重复调）；Jina batch API（目前是串行请求）
  - Step5：用 vLLM 自建推理服务替换第三方 endpoint（消除 endpoint 尾延迟）；A0 和 A1 可以 pipeline 化（A0 单题完成后立刻送 A1，不等 batch）

  ## Step 3b：轻量链可读性过滤（step3b_readability_filter.py）

  独立 post-step 脚本，不改 step3_trajectory。

  **拦截三类不自然题目：**

  1. **flow ≤ 2**：最后一跳语义硬拐（比如前两跳讲球员生涯，突然问家乡城市的人口）
  2. **motivation ≤ 2**：无提问动机（"这家公司的竞争对手的止汗剂品牌的母公司成立日期"——概念堆砌、纯为出题）
  3. **naturalness ≤ 2**：翻译腔/定语堆砌（"画面上方偏右那个蓝色背景且带有白色手写体品牌广告牌所代表公司的..."）

  **方法**：对每道 L3 题 1 次 LLM call（question + reasoning_path 摘要 → 3 维 1-5 分），任一项 ≤ 2 → `readability_flag="rejected"`

  **实测（stress+sku 156 L3 → 19 成功打分 / 137 endpoint error）：**
  - 6/19 被拦截（32%）
  - 拦截理由全部合理（"凑多跳动机极弱"、"通过母公司关联两个子产品不自然"、"强行对比两个无关领域"）
  - `aggregate_hard_split.py --apply-readability-filter` 可选启用

  **幂等设计**：已有 `readability_flag` 的题默认跳过，`--force` 可强制重打分。

  ## 第四步：模糊化验证与修正（step4_verify.py）

  ### 目标

    对第三步生成的问题进行质量校验，确保模糊化正确执行，修正不合格的题目。
      
    跑一遍模型

  ### 流程

  #### 结构性检查（不需要 VLM）

  - 所有工具必须在有效工具集内（`web_search/image_search/visit/code_interpreter`）
  - 必须有 `question` 和 `answer` 字段

  #### VLM 模糊化验证（只检查 L2 和 L3）

  - L2：被识别的实体名应已从问题文本中移除，但问题仍然清晰可理解
    - 正确示例：*"图中标记为U3的芯片"*（明确指向位置，但不透露型号）
    - 错误示例：*"图中的某个芯片"*（太含糊）
  - L3：问题文本中不应存在任何可直接搜索获得有用结果的实体名

  #### 自动修正

  - 如果 VLM 发现模糊化不正确，生成修正后的问题文本
  - 将修正回写到问题文件（原文保留到 `question_original` 字段）
  - 标记所有问题为 `verified: true`

  #### 重新聚合

    验证后重新调用 `step3_generate.aggregate_final()`，更新 `output/final/` 下的 JSONL 文件。

  ### 输出

  - `output/stats/verification_report.json` — 验证报告（含结构问题数、模糊化修正数）
  - 更新后的 `output/questions/{img_id}.json` 和 `output/final/*.jsonl`

  ## 第五步：攻击式过滤（step5_attack.py）

  ### 目标

    灵感来自 MTA-Agent / VSearcher / WebSTAR 的"生成很多 → 攻击 → 只留通不过攻击的"策略：
    Step3 产出的"看上去 hard"题不一定真 hard，必须经过弱模型攻击验证才能进训练集。
    
    **核心理念**：不在生成阶段手写规则预防 shortcut，而是靠攻击式过滤兜底。
    规则写得再多都会漏（领域词典爆炸），但攻击器是数据驱动的，通过率就是不可约性的直接度量。

  ### 4 个攻击器（工具白名单）

  | 攻击 | 允许工具 | 检查的 shortcut |
  | --- | --- | --- |
  | `no_tool` | (无) | 模型先验 + 视觉 shortcut（图里直接看出答案） |
  | `no_image_search` | web_search / visit / code | image_search 可被文字 + OCR 替代 |
  | `no_visit` | web_search / image_search / code | page_only fact 可被 snippet 替代 |
  | `no_code` | web_search / image_search / visit | code 可被纯检索替代 |

    任一攻击成功（模型答案命中 ground truth）→ 题被攻破，标 `hard_attacker_passed=False`，从 hard split 排除。

  ### 三层 attacker

  | Tier | 模型 | 用途 | 范围 |
  | ---- | ---- | ---- | ---- |
  | A | Qwen3-VL-30B-A3B | 便宜广筛 | 全部 L3，two_phase 调度 |
  | B | Gemini 3 Flash | Hard 复判 | 只跑 A 通过的 hard bucket（visit_heavy / code_heavy / chain_heavy / ultra_long），two_phase 调度 |
  | C | Claude Sonnet 4.5 | 校准集 | ≤500 题版本校准，不进主流程 |

  **关键约束**：在真 lens reverse 大批量生效前，B/C 层不对 `image_heavy` / `all_tools` bucket 做最终 reject——因为 reverse 真改图之后，这些 bucket 的供料会大变，提前严判反而是在"对旧分布过拟合"。

  ### Bucket-aware 两阶段调度（默认 `--mode two_phase`）

    原始实现对每道题跑全部 4 个攻击，运行时间被 endpoint 尾延迟拉爆。
    新调度和 hard bucket 的语义对齐：**只验证题目声称依赖的工具是否真不可替代**。

  **Phase A0**（全量广筛）
  - 所有 L3 题跑 `no_tool`（最便宜、所有 bucket 都该过的一关）
  - `max_steps=1`（固定 1 次 LLM call）
  - `workers_a0=4`（高并发，因为单步快）

  **Phase A1**（按 bucket 跑专属攻击，仅对 A0 通过的题）
  - `image_heavy` → 只跑 `no_image_search`
  - `visit_heavy` → 只跑 `no_visit`
  - `code_heavy` → 只跑 `no_code`
  - `all_tools` / `ultra_long` → 串行 cascade（no_image_search → no_visit → no_code），任一攻破立即停
  - `chain_heavy` / `standard` → 止步，不再加攻击
  - `workers_a1=2`（agent loop 多步，endpoint 尾延迟大时不适合高并发）

  **max_steps 分层**：
  ```
  no_tool                       : 1 call   固定
  单一 ablation (no_X)          : max_steps=4
  all_tools/ultra_long cascade  : max_steps=6
  ```

    这样大多数题只会经历 `no_tool + 1 个 bucket 专属攻击`，只有 all_tools/ultra_long 才走 3-ablation cascade。
    预估 wall time 从"每题 4 攻击各自多步"的 50–150 分钟压到 20–40 分钟。

  ### 用法

  ```bash
  # A 层两阶段（推荐，默认模式）
  python step5_attack.py output/questions/sku_*.json --tier A --mode two_phase \
      --workers-a0 4 --workers-a1 2

  # B 层只跑 hard bucket
  python step5_attack.py output/questions/sku_*.json --tier B --mode two_phase

  # Legacy 模式：每题跑全部 4 攻击（慢，只用于校准）
  python step5_attack.py output/questions/sku_8846.json --tier A --mode legacy --attacks no_tool
  ```

  ### 输出

    每个 question 字段加：
  - `attack_qwen3-vl-30b-a3b` / `attack_gemini-3-flash`：每层的攻击结果（含 attack_results 字典）
  - `hard_attacker_passed`：bool，所有生效的攻击都通过 → True
  - `breached_for_filter`：实际是否参与最终 reject（image_heavy 在 lens 未跑全前 = False）

  ### 下游：breach taxonomy 统计（step6_attack_stats.py）

    攻击跑完后，`step6_attack_stats.py` 聚合 `output/questions/*.json` 的 attack 结果成 taxonomy：

  ```bash
  python step6_attack_stats.py output/questions/sku_*.json -o output/stats/attack_breach_stats.json
  ```

    输出包含：
  - `by_bucket`：每个 hard bucket 的 total / breached / rate + 各攻击细分
  - `by_family`：每个 closure family 的 breach rate
  - `by_attack`：每种攻击的 tried / breached
  - `breached_examples`：被攻破题目的完整 context（含 `shortcut_path_type: null` 待手工标注）
  - `threshold_verdict`：`keep_B_path` / `inspect_motif_concentration` / `implement_A_prime_coarse_anchor_substitution`

  **`shortcut_path_type` 手工分类（4 档）**：
  - `scene_shortcut`：整图场景就能推出答案所属实体（如记分牌+球场 → 直接推 Denver Nuggets）
  - `visible_text_shortcut`：OCR/显性文字就够（如广告牌品牌名）
  - `known_entity_shortcut`：不用锚点识别，靠已知大实体就能搜到
  - `compute_not_needed`：code 只是装饰，实际单次搜索就够

  **阈值决策策略**（事先定好，避免事后 rationalize）：

  | image_heavy × no_image_search rate | 处理 |
  | --- | --- |
  | ≤ 15% | **keep_B_path**：继续走攻击过滤，不加生成期规则 |
  | 15-30% | **inspect_motif_concentration**：看 shortcut_path_type 是否集中，决定 targeted 修复 |
  | > 30% | **implement_A_prime**：实施 coarse-anchor substitution test（图结构判定，非领域词典）|

  **A' 设计（只有阈值命中才实施）**：对 image_heavy closure，暂时移除锚点的 resolve 边，看是否存在 `full_image → coarse_entity → answer` 的替代路径。存在则降级为 standard。这是图级别的替代路径检测，不是手写"场景无关"脆弱规则。

  ## Pipeline 编排（pipeline.py）

  ### 运行方式

    python pipeline.py                    # 运行全部4步
      
    python pipeline.py --start-from 2     # 从第2步开始
      
    python pipeline.py --only 1           # 只运行第1步
      
    python pipeline.py --workers 8        # 设置并发数
      
    python pipeline.py --limit 20         # 第一步只处理前20张图片

  ### Step2+Step3 流水线

    Step2 和 Step3 采用流水线执行：Step2 完成一张图片后立即提交 Step3，而非等 Step2 全部完成。
      
    ┌─── step2 线程池（workers 数）──────────────────┐
      
    │  img_01: 实体提取 → 搜索 → 三元组 → 多轮扩展      │
      
    │  img_02: 实体提取 → 搜索 → 三元组 → 多轮扩展      │──→ 完成一张立即提交 step3
      
    │  img_03: 实体提取 → ...                            │
      
    └─────────────────────────────────────────────────────┘

  ​                          ↓ (逐张提交)

    ┌─── step3 线程池（workers 数，无 API 限制）──────────┐
      
    │  img_01: 找链 → 骨架 → LLM润色 → 输出问题          │
      
    │  img_02: 找链 → 骨架 → LLM润色 → 输出问题          │
      
    └─────────────────────────────────────────────────────┘

  ### 检查点机制

    每步每张图片完成后保存检查点到 `output/.checkpoints/step{N}/{img_id}.json`，重跑时自动跳过已完成的。需要重新生成时，手动删除对应检查点。

  ## 数据质量保障机制总结

  | 问题 | 解决方案 | 实现位置 |
  | ---- | ------- | ------- |
  | 图谱太小 | expansion_seeds 不限数量 + closure richness 停止条件 | step2_enrich |
  | 搜索 crop 太小 | 三档 search_views（tight/pad20/context） | step2_enrich |
  | 搜索词不稳定 | exact-name-first 确定性搜索，删 LLM 搜索计划 | step2_enrich |
  | round ≥2 不扩展 in-image 实体 | 放开 in-image 过滤，让 Denver Nuggets 这类桥接实体也能 expand | _extend_search |
  | page_only 检测过于保守 | 50+ relation 硬 override + 不依赖 deep_read_corpus + 空间关系单独标 spatial | _mark_retrieval_mode |
  | 实体名不匹配 | 类型感知的模糊关键词匹配 | _build_entity_index / _match_entity |
  | 实体名漂移 | 后处理启发式 canonical name 替换 | _normalize_triple_entities |
  | out-of-image entity 在 Step3 被丢 | 桥接提升：synthetic entity 节点，可作 multi_hop 中间跳 | HeteroSolveGraph._build |
  | LLM 泄露实体名 | postcheck_name_leak 检查所有 hop 实体 + bridge value | step3_trajectory |
  | 工具序列造假 | 从轨迹确定性编译，不让 LLM 规划 | compile_tool_plan |
  | 答案太泛 | answer_uniqueness 不可约性检查 | check_irreducibility |
  | 答案看图直接可得 | answer_not_visible 检查 | check_irreducibility |
  | L3 题去掉比较也能答 | no_python_shortcut 检查 | check_irreducibility |
  | 多跳题可一跳到达 | 3 道 multi_hop shortcut 检查（direct / skip_bridge / hop2_direct） | check_irreducibility |
  | 多分支题删分支还能答 | no_branch_shortcut 检查 | check_irreducibility |
  | 锚点无法视觉描述 | realizable_question 检查 | check_irreducibility |
  | 题型死板 | 后验归纳而非预设 family，含 multi_hop 2/3-hop | enumerate_closures |
  | 难度靠 hop 数 | 偏好向量 + budget deficit + bridge affordance + page_only 加分 | DifficultyProfile + _score_expand |
  | walker 卡在短链 | HARD 难度对 image_search_needed 锚点 + page_only 边加分 | _score_spawn / _score_expand |
  | 实体间缺乏跨域关联 | 跨实体穷举 C(n,2) 搜索 + 桥接三元组 | _find_cross_entity_relations |
  | 位置描述不准确 | 由代码从 bbox 计算，不依赖 VLM | _bbox_to_location |
  | 视觉指代太模糊 | 方位 + 视觉类别（entity_type + relation 推断） | visual_descriptor |
  | tool_plan 步数不够长 | 桥接提升 + 3-hop multi_hop + ultra_long bucket 配额 | enumerate_closures + _classify_hard_bucket |
  | 7+ 步链没有专属 bucket | ultra_long bucket（≥7 步且工具种类 ≥2，长度优先于其他 bucket）| _classify_hard_bucket |

  ## 每张图片的预期产出

  | 级别 | 题型（后验归纳） | 工具签名 | 目标数量 | 模糊化 |
  | ---- | ---------- | -------- | -------- | ------ |
  | L1 | read（纯视觉识别，仅 ocr_likely） | （无工具，VLM 直接看图） | ≤4 | 无 |
  | L2 | lookup（识别+查知识） | (image_search?) + web_search + (visit?) | ≤3 | 图中实体用视觉描述 |
  | L3 | compare / compare_then_follow / rank / set_merge / **multi_hop (2-hop)** / **multi_hop (3-hop)** | 2-7+ 步组合（含 ultra_long 配额可达 8 步）| ≤8 | 视觉描述 + 隐藏所有中间桥 |
  | 总计 | | | 约 10-15 题 | |

    题型由**弱约束随机游走后验归纳**产出，不预设。工具序列由轨迹**确定性编译**，不由 LLM 规划。实际数量取决于异构证据图的 richness 和 closure 质量，宁缺勿滥。

  ## 文件结构

    项目根目录（精简后，无 experimental/）：
    
    agenticdata_only_vlm/
    ├── pipeline.py                # 4 步流程编排器
    ├── step1_filter.py            # Step1：图片筛选
    ├── step2_enrich.py            # Step2：实体提取 + 知识图谱构建
    ├── step3_generate.py          # Step3 入口（437 行薄壳，提供旧 API）
    ├── step3_trajectory.py        # Step3 核心：HeteroSolveGraph + walker + closure 枚举（2000 行）
    ├── step3_graphenv_runtime.py  # Step3 工具库：relation natural text、askability profile
    ├── step4_verify.py            # Step4：模糊化验证
    ├── step5_attack.py            # Step5：攻击式过滤（A/B/C 三层 attacker，two_phase 调度）
    ├── step6_attack_stats.py      # Step6：breach taxonomy 聚合 + 阈值 verdict
    ├── run_step3_batch.py         # Step3 并发 batch runner
    ├── eval_agent.py              # 真实工具调用 eval（接 API）
    ├── core/                      # 配置、日志、checkpoint、VLM client
    │   ├── config.py
    │   ├── lens.py                # ✨ Serper Google Lens reverse + litterbox upload
    │   ├── vlm.py
    │   └── ...
    └── output/                    # 所有产出（结构如下）
    
    output/
    
    ├── images/                     # 筛选后的图片
      
    │   ├── img_0010.jpg
      
    │   └── ...
      
    ├── entities/                   # 每张图的实体 + 三元组
      
    │   ├── img_0010.json           # 含 entities, core_anchors, expansion_seeds, triples, search_views, search_results
      
    │   └── ...
      
    ├── questions/                  # 每张图的分层问题
      
    │   ├── img_0010.json           # 含 level_1[], level_2[], level_3[]
      
    │   └── ...
      
    ├── final/                      # 聚合后的最终数据
      
    │   ├── level_1_all.jsonl
      
    │   ├── level_2_all.jsonl
      
    │   ├── level_3_all.jsonl
      
    │   └── all_questions.jsonl
      
    ├── stats/                      # 统计信息
      
    │   ├── filter_scores.json
      
    │   ├── question_stats.json
      
    │   └── verification_report.json
      
    └── .checkpoints/               # 断点续传检查点

  ​      ├── step1/

  ​      ├── step2/

  ​      ├── step3/

  ​      └── step4/

  ## 单条最终数据格式（JSONL 每行）

    **注意**：工具序列由 Step3 确定性编译，不含"裁剪 crop"这种隐式步骤。L1 read 的 tool_sequence 是空数组（VLM 直接视读）。L2 lookup 的 tool_sequence 只有 web_search（± image_search / visit，取决于 resolve_mode / retrieval_mode）。
    
    L2 lookup 示例（anchor resolve_mode=ocr_likely, retrieval_mode=snippet_only）：
    
    ```json
    {
      "id": "sku_8846_L2_02",
      "image_id": "sku_8846",
      "image_path": "output/images/sku_8846.png",
      "level": 2,
      "family": "lookup",
      "hard_bucket": "standard",
      "question": "货架上方偏右位置那个蓝绿色背景、印着白色字母的品牌广告牌，它所属的母公司是哪家企业？",
      "answer": "Haleon",
      "tool_sequence": [
        {"step": 1, "tool": "web_search",
         "action": "搜索 Sensodyne 的所属母公司",
         "expected_output": "获取到 Haleon"}
      ],
      "reasoning_path": {
        "family": "lookup",
        "anchors": ["region:sensodyne"],
        "entity_key": "entity:sensodyne",
        "follow_relation": "parent_company"
      },
      "obfuscation_applied": true,
      "obfuscated_entities": ["Sensodyne"],
      "code_skill_tags": [],
      "verified": true,
      "domain": "retail"
    }
    ```
    
    L3 multi_hop 3-hop 示例（含 image_search resolve + 2 visit）：
    
    ```json
    {
      "id": "sku_8845_L3_01",
      "level": 3,
      "family": "multi_hop",
      "hard_bucket": "ultra_long",
      "n_hops": 3,
      "question": "画面右侧印有女性剪影的橙黄色品牌广告牌，其所属品牌的母公司最近一次剥离出来的独立公司总部位于哪座城市？",
      "answer": "New Brunswick, NJ",
      "tool_sequence": [
        {"step": 1, "tool": "image_search", "action": "对画面右侧区域做图像检索以识别实体"},
        {"step": 2, "tool": "web_search", "action": "搜索 [anchor] 的母公司"},
        {"step": 3, "tool": "visit", "action": "深读搜索结果页确认母公司"},
        {"step": 4, "tool": "web_search", "action": "在上一步结果基础上，搜索其剥离历史"},
        {"step": 5, "tool": "visit", "action": "深读搜索结果页确认剥离公司名"},
        {"step": 6, "tool": "web_search", "action": "在前 2 步结果基础上，继续搜索其总部所在城市"}
      ],
      "reasoning_path": {
        "family": "multi_hop",
        "anchors": ["region:tone"],
        "hop_chain": [
          {"entity": "entity:tone", "relation": "owned_by", "value": "Henkel"},
          {"entity": "entity:henkel", "relation": "spun_off_brand", "value": "Dial Corporation"},
          {"entity": "entity:dial corporation", "relation": "headquartered_in", "value": "New Brunswick, NJ"}
        ],
        "n_hops": 3
      },
      "code_skill_tags": [],
      "obfuscated_entities": ["Tone", "Henkel", "Dial Corporation"],
      "verified": true,
      "domain": "retail"
    }
    ```


  ## 实测产出（4 张 sku_88xx 图）

  Step2 + Step3 端到端跑过的最新统计（基于改进后的 _mark_retrieval_mode + 桥接提升 + 3-hop multi_hop + ultra_long bucket）：

  | 图 | total triples | 空间(skip) | 知识 triples | page_only | page_only % |
  | --- | --- | --- | --- | --- | --- |
  | sku_8845 | 147 | 103 | 44 | 35 | **80%** |
  | sku_8846 | 260 | 189 | 71 | 32 | **45%** |
  | sku_8847 | 117 | 79 | 38 | 15 | **39%** |
  | sku_8848 | 89 | 45 | 44 | 18 | **41%** |

  **Step3 L3 题目分布（含 multi_hop 2/3-hop + ultra_long）：**

  | 图 | L3 题数 | 3-hop multi_hop | ultra_long | 最长 chain |
  | --- | --- | --- | --- | --- |
  | sku_8845 | 2 | — | — | 5 |
  | sku_8846 | 6 | 1 | **2** | **7** |
  | sku_8847 | 6 | — | — | 5 |
  | sku_8848 | 7 | 2 | **2** | **7** |

  **典型 7 步 closure**（compare_then_follow，含双 page_only visit）：
  ```
  step 1  web_search   搜索分支 A 的事实
  step 2  visit        深读 A 的页面（page_only）
  step 3  web_search   搜索分支 B 的事实
  step 4  visit        深读 B 的页面（page_only）
  step 5  code         normalize 标准化数值
  step 6  code         compare 选出 winner
  step 7  web_search   追问 winner 的额外属性
  ```

  每一步都通过 `check_tool_irreducibility` 严格检查（不允许"visit 验证"等冗余填充）。

  ## 已知局限与下一步改进计划

  ### 已解决的问题

  - ~~walker 遇到第一个 closure 就停~~ → hard 难度下有硬预算检查（image/visit/多锚点/compute 四项 budget，每项缺口 -3 分）
  - ~~bucket 只是标签~~ → 现在是硬门槛，不通过 `check_tool_irreducibility` 的降级到 standard
  - ~~retrieval_mode 未标记~~ → Step2 现在给每条 triple 标记 `snippet_only` / `page_only` / `spatial`，多信号融合 + relation 硬 override（owner/founder/ceo/manufacturer/headquartered_in 等命中即标 page_only，**不依赖 deep_read_corpus 是否非空**——visit 实际成不成功不影响标记，因为 page_only 是"只能从深读拿"的语义判断）
  - ~~resolve_mode 未标记~~ → Step2 现在给每个实体标记 `ocr_likely` / `image_search_needed`
  - ~~visit 全部失败~~ → 修复了 Jina Reader 的 `trust_env` 问题，用独立 http client
  - ~~reverse search 完全未实现~~ → 实现了 **VLM describe crop → search workaround**（Serper 不支持图片上传，改用 VLM 描述 crop 内容再拿描述做文字搜索）
  - ~~discovered_entities 在 log 里打印后丢掉~~ → 用 LLM 把候选标题清洗成 canonical name，加进 `expansion_seeds` 触发下一轮搜索
  - ~~Step2 停止条件是 graph richness 不是 closure richness~~ → 改成 6 个 closure richness 指标（image_resolved_anchors / page_only_facts / compare_ready_pairs / rank_ready_triplets / cross_anchor_shared_nodes / total_triples），满足 ≥3 项即停
  - ~~Step2 round ≥2 不会扩展 in-image 实体（如 Denver Nuggets）~~ → 放开了 `_extend_search` 的 in-image 过滤，round ≥2 把 in-image 实体也作为 seed 展开二跳事实
  - ~~Step3 没有 multi_hop 题型~~ → 加了 `multi_hop` closure family（2-hop + 3-hop）
  - ~~3-hop 跨不过 out-of-image 实体~~ → `HeteroSolveGraph._build` 做桥接提升，对 triple 中非 in-image 的 head 创建 synthetic entity 节点
  - ~~set_merge 用 fact_key 比较永远不命中~~ → 改成用 canonical tail value 比较，并加 relation 语义兼容性检查
  - ~~L1 read tool_plan 塞满 crop+OCR~~ → L1 read 返回空 tool_plan（VLM 直接视读）
  - ~~`experimental/random_walk_step3/` 文件夹和主 pipeline 分离~~ → 整合到根目录，删 experimental 文件夹，新结构：`step3_generate.py`（薄壳）+ `step3_trajectory.py`（核心）+ `step3_graphenv_runtime.py`（工具库）
  - ~~tool_sequence 最多 5 步~~ → 桥接提升 + 3-hop multi_hop + ultra_long bucket 后可达 7 步以上（实测 sku_8846/sku_8848 各产出 2 条 7 步 ultra_long）
  - ~~大量空间关系污染 page_only 比例~~ → 空间关系（located_above/left_of/right_of 等）单独标 `"spatial"`，不参与 page_only/snippet_only 二分类
  - ~~image_search(reverse) 完全是 VLM workaround，结果不进图~~ → 接通真 Serper Google Lens（`core/lens.py`），crop 上传 litterbox 拿 URL → Serper Lens → top-20 visual_matches + top-15 source pages 真的进 `discovered_entities` / `visit_tasks`，且 `lens_n_visual > 0` 强制升级 `resolve_mode=image_search_needed`。**不需要 SerpApi 账号**，复用现有 SERPER_KEY 同一个 endpoint
  - ~~Step2 page_only 几乎为 0~~ → 三处修复：(1) 扩展 `_mark_retrieval_mode` 的 page_leaning_relations 列表（owned_by / launched_in / sold_brand_to / formerly_division_of 等共 50+ relation）；(2) 去掉 `and deep_read_corpus` 的硬依赖，让 relation override 即使本次 visit 全失败也能生效；(3) 空间关系（located_above 等）单独标 `"spatial"`，不参与 page_only 比例分母。修复后实测 sku_8845 / sku_8846 / sku_8847 / sku_8848 的知识三元组 page_only 比例分别为 80% / 45% / 39% / 41%
  - ~~rank closure 答案是 `rank_winner` 占位符~~ → `enumerate_closures` rank 分支按 `rank_type`（earliest/latest/largest/smallest）真实计算 argmin/argmax，`answer=winner_entity_name`；tie 时直接 skip。`compile_frame` rank 分支用 rank_type 构造 criterion（"成立时间最早" / "数值最大"）
  - ~~multi_hop 把 `covered_the_pre_draft_workout_of` 这种句子碎片当 relation 用~~ → `_relation_profile` 按 token 数分 3 档 lexicalizability：hit_known_map=0.85 / crisp_raw ≤3 tokens=0.65 / sentence_fragment >4 tokens=0.20；`enumerate_closures` multi_hop 要求每跳 lex ≥0.5，自动过滤句子碎片型 relation
  - ~~realize LLM 把 `won_championship_in` 翻译成"加入 NBA 年份"等自由发挥~~ → 两处修复：(1) 删掉 `_relation_natural_text` 里的领域特定映射（championship / plays_for / draft_year 等），未知 relation 直接返回 raw slug 去下划线形式，不再手写领域词典；(2) `QuestionFrame` 加 `chain_trace` 字段，multi_hop prompt 显式展示完整 hop 关系链 `hop1[plays_for → plays for] → hop2[won_championship_in → won championship in]`，LLM 看到 raw slug 无法自由翻译
  - ~~`_mark_retrieval_mode` 在 `_add_spatial_fallback` 之前跑，空间兜底三元组拿不到 "spatial" 标记~~ → 空间兜底后再调一次 `_mark_retrieval_mode`，补标新加的空间关系
  - ~~`_mark_retrieval_mode` 依赖手写 relation 词典，换新领域就爆炸（产品类 has_volume / brand_belongs_to 等）~~ → 加 **tail_type-based 通用 fallback**：tail_type 是具体类型（PERSON / LOCATION / ORG / TIME / QUANTITY）且 tail 值没在 snippet 完整出现 → 标 page_only。这是领域无关的启发式，新领域（产品/药品/电影）不改代码也能工作。实测 sku 四张图 page_only 比例恢复到 50-69%
  - ~~step5_attack 每题跑全部 4 攻击，wall time 失控~~ → 加 **bucket-aware 两阶段调度**（`--mode two_phase`）：Phase A0 全量只跑 no_tool（便宜广筛），Phase A1 只对 A0 通过的题按 bucket 跑专属攻击（image_heavy → no_image_search / visit_heavy → no_visit / code_heavy → no_code / all_tools+ultra_long → 3-ablation cascade / chain_heavy+standard → 止步）。max_steps 分层（1 / 4 / 6）。预估总时长从 50-150 分钟压到 20-40 分钟
  - ~~没有 breach taxonomy 统计~~ → `step6_attack_stats.py` 聚合 `by_bucket × attack × family × anchor_type`，输出 `attack_breach_stats.json`，含 `shortcut_path_type` 手工分类槽位和 `threshold_verdict` 自动决策（keep_B_path / inspect_motif_concentration / implement_A_prime）
  - ~~`page_only` 是"语义上需要 visit"，但可能"实际没证据"~~ → 分层为 `page_only_evidenced`（Signal 1/2 有真证据）/ `page_only_semantic`（Signal 0/3 是 relation override 或 tail_type fallback 推的）/ `snippet_only` / `spatial`。`check_tool_irreducibility` 中 `visit_heavy` bucket 严格要求 `page_only_evidenced`（不接受 semantic），`ultra_long` 接受两者。**分层立即暴露了 visit 通道的结构性失败**：实测 4 张 sku 图 `page_only_evidenced=0`，全部是 semantic（见下方 visit 通道问题）
  - ~~Step2 输出只有 triples，没有 provenance 可溯~~ → 加一等证据层输出：`resolution_edges` / `image_pages` / `web_pages` / `evidence_stats`。`resolution_edges` 记录 `region → candidate_entity → canonical_entity` 的证据链（provenance: lens_reverse / lens_kg / image_search_text / vlm_describe_workaround）。Step5/6 诊断时可以按 provenance 归因。

  ### Reverse search 现状（已升级到真 Serper Lens）

    对 `person/landmark/product` 类型实体，新流程（`core/lens.py`）：
    
    1. 读取 pad20 crop
    2. 上传到 `litterbox.catbox.moe`（1 小时临时图床，无 API key）拿公网 URL
    3. 调 `POST https://google.serper.dev/lens`（用现有 SERPER_KEY，不需要 SerpApi）
    4. 从 `organic` 抽 top-20 candidate titles + top-15 source pages
    5. 候选标题进 `discovered_entities` → LLM canonicalize → 加入 `expansion_seeds`（触发下一轮搜索）
    6. 来源页进 `visit_tasks` → 实际深读 → 转化为 page_only fact
    7. lens visual_matches > 0 → 强制升级 `resolve_mode=image_search_needed`（真证据）

  **实测效果（NBA 球员 crop）：**
  - E10_context（球场场景） → Lens 返回 60 个 visual_matches，包括 Reddit/Facebook 上关于球队的帖子
  - 候选标题包含真实的篮球文章 → LLM 清洗后得到球队名 / 球员名 / 教练名等 canonical entities
  - 来源页（reddit / facebook 等）进入 visit queue，真的读出 page_only 内容

  **仍可改进的地方：**

  - 候选标题质量取决于 crop 质量；面对"局部物件 crop"时 Lens 可能返回不相关结果（如把球鞋 crop 误识别为 Amazon 商品页）
  - LLM canonicalize 步骤目前规则较保守，可能漏掉一些有效候选
  - 没有 SERPER_KEY 时自动 fallback 到旧的 VLM describe workaround（保留为兜底）

  ### 核心遗留问题

  **0. visit 通道结构性失败（P0 — 最高优先级）**

  - **现象**：sku_8846 实测 52 次 searches 对应 **0 deep_reads**；4 张 sku 图的 `page_only_evidenced` 全部为 0，所有声称的 page_only 全部是 `page_only_semantic`（relation override 或 tail_type fallback 推的）
  - **影响**：旧 `visit_heavy` 和 `ultra_long` bucket 都是"语义虚胖"——它们不是因为真的需要 visit 才闭合的，而是因为我们标签系统乐观推断的
  - **诊断结论**：Jina Reader 要么在主流程里全部失败，要么 deep_reads 没落地到 `entity.resolution.visited_pages`。需要查 `visit_url` 函数的实际调用/返回路径
  - **临时缓解**：`check_tool_irreducibility` 的 `visit_heavy` bucket 只吃 `page_only_evidenced`（严格门槛），所以 evidenced=0 的情况下 visit_heavy 会自动归零（被降级成 standard），不再产出虚假的 visit_heavy 题
  - **正确修复**：排查 `step2_enrich.py::visit_url` + `tool_visit`，定位 Jina Reader 失败原因；把真正的 visit 结果落地到 `image_pages` / `web_pages` 一等字段

  **1. 搜索词仍偏通用**

  - "Jamal Murray" 只拿到维基百科级别概览，拿不到合同金额、赛季数据等细节
  - 没有结合图片场景细化搜索词
  - **改进方向**：在 round 2+ 的搜索 prompt 里给 LLM 加更具体的"模板查询词"（draft_year / arena_capacity / acquisition history 等）

  **2. visit 在跨实体和 reverse 阶段仍然不稳定**

  - 主搜索阶段 visit 正常（Jina Reader 能读）
  - 跨实体阶段经常 0 篇深读（可能是并发太多 + Jina 限流）
  - reverse search 来源页 visit 也全失败
  - **不再卡瓶颈**：page_only 标记现在不依赖 visit 实际成功，relation override 是语义判断（"这个 fact 只能从深读拿"），所以即使 visit 全挂了，page_only fact 仍然会被标出，walker 仍然会调度 visit 步骤，agent 在 eval 阶段才真正去 visit

  **3. 3-hop 链有时语义不自然** ✅ 已缓解

  - 桥接关系组合可能产出 "player→team→arena→capacity" 这种拼凑感强的链
  - 当前用"中间跳 tail_type 不能是 TIME/QUANTITY"做基础过滤
  - **已实施**：`step3b_readability_filter.py` 用 LLM 对 L3 题打 flow/motivation/naturalness 三维分（1-5），任一项 ≤ 2 拦截。实测 6/19 scored 被拦，拦截理由全部合理（"概念堆砌"、"强行对比无关领域"、"凑多跳动机极弱"）

  **4. ultra_long bucket 依赖 page_only fact 数量**

  - ultra_long 必需 ≥7 步，主要靠 page_only → visit 步数堆叠
  - 如果一张图的 Step2 只产出 1-2 条 page_only fact，难以形成 ultra_long
  - **现状已大幅改善**：50+ relation override + 不依赖 visit 成功 + 空间关系隔离，实测 4 张 sku 图的知识三元组 page_only 比例都 ≥39%（最高 80%），sku_8846/sku_8848 各产出 2 条 ultra_long 的 7 步 closure

  **5. 合成桥接实体的 fact 数据质量** ✅ 已缓解

  - synthetic entity 的 fact 来自 round ≥2 的扩展搜索，数据噪声比 in-image entity 多
  - 桥接折叠用 canonical 严格匹配，"Nuggets" 和 "Denver Nuggets" 能合并，但 "the Denver Nuggets" 会单独成桥
  - **已实施**：`_llm_canonicalize_aliases` 在 Step2 2b-5b 阶段用 LLM 做别名分组归一，把同义变体合并后重写 triples + entities + dedup

  **6. VLM 对 sports 类图的实体分类盲区** ✅ 已解决

  - **现象**：NBA 赛场图 84 个实体中仅 1 个 person，球员 bbox 根本没进 entities[]（不是分错 type，是检测缺失）
  - **根因**：VLM 主 prompt 导向"识别品牌/文字/标志"，球员在动作场景中尺寸小且被记分牌/广告压制
  - **已实施**：`sports person proposal 第二遍`（2a-1b 段）——条件触发（person=0 + sports 上下文），整图 2×2 tile 切割 + person-only prompt，不改主 prompt
  - **实测**：sports_03（Heat vs Knicks）从 person=0 → person=6（Mitchell Robinson, Duncan Robinson, Tom Thibodeau, Erik Spoelstra, Kyle Lowry, 1 描述性 player）

  **7. visit_heavy bucket 可能过度标注**（新发现）

  - **现象**：Tier B no_tool 在 visit_heavy 上 44.4% breach，但 no_visit 0%
  - **含义**：Step3 把"Jina 读过这个页面"等同于"必须读这个页面"，但很多 page_only fact 是模型先验已知的百科事实（如"State Farm 总部在 Bloomington"）
  - **改进方向**：`check_tool_irreducibility` 加轻量 prior-knowledge filter（"这题靠先验能答吗？"），是则降级 bucket

  ### 改进方向 1：重定义"难"

    不用 hop 数，用 **effective_difficulty**：
    
    ```
    effective_difficulty
    = 必需的 image resolve 次数
    + 必需的 page visit 次数
    + 必需的非平凡 code 操作次数
    + 独立分支数
    + 合流后的 follow 次数
    ```
    
    hard 分成 4 个 bucket：

  | Bucket | 核心约束 |
  | ------ | ------- |
  | image-heavy | ≥1 reverse image search + ≥2 visual anchors + ≥1 follow |
  | visit-heavy | 关键 fact 必须来自 page-only 内容，snippet 不够 |
  | code-heavy | ≥2 类 code 操作（不允许只有 crop + argmin） |
  | all-tools | code + web_search + image_search + visit 全部必须出现 |

    全局选题器按 bucket 配额选，不只按 L1/L2/L3。

  ### 改进方向 2：image_search 不能只在 resolution 里用完就蒸发

  **当前问题：** image_search 结果只留在 `entity.resolution` 里当备注，Step3 的图里没有对应边。

  **改法：** Step2 输出两层证据图：

  - **resolution graph**：`region → candidate_entity → canonical_entity`，保留 `image_reverse / text_to_image` 等 resolve 证据
  - **fact graph**：`canonical_entity → fact`，保留 `retrieval_mode: snippet_only / page_only / image_page`

    每条 `resolve` 边带 `resolve_mode`（ocr_only / image_reverse_only / text_to_image_only / hybrid），Step3 据此编译工具序列。

  **新增 3 类 image-heavy closure：**

  1. `reverse_then_lookup`：reverse 识别实体 → 查外部事实
  2. `reverse_compare_then_follow`：两个锚点分别 reverse → 查值 → compare → follow
  3. `text_to_image_disambiguate`：用已知名搜图匹配另一个 region → 确认后查事实

  **新增不可约性检查：**

  - `no_image_search_shortcut`：去掉 image_search 后还能靠 OCR + web_search 直接答 → reject
  - `no_visit_shortcut`：去掉 visit 后靠 snippet 也能答 → reject

  ### 改进方向 3：visit 不能只被压成三元组

  **当前问题：** visit 读的网页内容被 LLM 压成三元组后，"只有深读才能拿到"的信息丢失了。

  **改法：** 每条 fact 标记 `retrieval_mode`：

  - `snippet_only`：搜索摘要就能拿到
  - `page_only`：必须 visit 才能拿到
  - `image_page`：必须 image_search + visit 才能拿到

    visit-heavy bucket 只允许从 `page_only` fact 闭合。

  ### 改进方向 4：code 操作拆成 6 个 skill tag

  | Skill Tag | 典型操作 | 包 |
  | --------- | ------- | -- |
  | cv_preprocess | 裁剪、缩放、二值化、轮廓 | PIL / OpenCV |
  | ocr_parse | OCR 结果清洗、regex、字段提取 | re / pandas |
  | layout_geometry | bbox 面积、距离、排序、最近/最远 | NumPy |
  | tabular_aggregate | 多值整理成表、groupby、merge | Pandas |
  | normalize_compare | 日期/单位/货币标准化后比较 | datetime / pandas |
  | evidence_vote_merge | 多搜索结果投票、去重、交集 | collections / set |

    每道题打 skill tag，全局配额保证覆盖。code-heavy bucket 要求 ≥2 个不同 skill tag。

  ### 改进方向 5：Step2 持久化一等证据层（P1，当前正在做）

    Lens + visit 已经在运行时改图，但输出结构还是"只序列化 triples"。**下一步让 Step2 输出结构显式持久化**：

  - `resolution_edges`：`region → candidate_entity → canonical_entity` 的证据链（含 provenance: `lens_reverse / lens_kg / vlm_describe_workaround / text_exact`）
  - `image_pages`：lens visual_matches 返回的 source pages（url / domain / title / provenance / visited_ok）
  - `web_pages`：主搜索 + 跨实体 + 扩展搜索的 visited pages（和 image_pages 同 schema）
  - `local_artifacts`（扩展）：`ocr_text_blocks / numeric_labels / bbox_area / relative_size / dominant_color / layout_relations`，供 Step3 code-heavy 使用

    **为什么现在做**：Step5 攻击器 + Step6 breach taxonomy 在诊断时需要按 provenance 归因（"这题的 shortcut 是因为 scene 太明显 / page 实际没读到 / lens 没命中"）。没有这三类一等输出，Step6 只能看到 triples，看不到溯源。

  ### 改进方向 6：`page_only` 分层为 semantic / evidenced（P0，当前正在做）

    当前 `_mark_retrieval_mode` 的 relation override + tail_type fallback 会把"语义上应当 visit"的 triple 直接标 `page_only`，即使本次样本并没有真的从 deep_read 拿到证据。这让 `visit_heavy / ultra_long` bucket 可能被语义标签人为抬高。
    
    **分层方案**：

  - `page_only_evidenced`：chunk / tail 在 `deep_read_corpus` 里有真证据，且 snippet 找不到
  - `page_only_semantic`：relation override 或 tail_type fallback 推出来的"原则上需要 visit"，但本次 deep_read 没证据
  - `snippet_only` / `spatial`：原样不变

    **下游语义**：
  - `check_tool_irreducibility` 的 `visit_heavy` bucket **只吃 `page_only_evidenced`**
  - `ultra_long` 可以用 `semantic + evidenced` 混合，但 Step6 在产出 hard split 时优先保留 evidenced
  - 这样既保住 recall（未来 visit 稳定后 semantic 会自动升级），也不会"虚胖"

  ### 实现优先级

  | 优先级 | 改进 | 状态 |
  | ------ | ---- | ---- |
  | P0 | resolve_mode + retrieval_mode 标记 | ✅ 已完成 |
  | P0 | hard bucket + 全局配额 | ✅ 已完成（6 buckets：image/visit/code/all_tools/chain/ultra_long/standard）|
  | P0 | image_search reverse workaround | ✅ 已完成（VLM describe → text search）|
  | P0 | image_search reverse 真接 Serper Lens（litterbox 图床 → Lens API → 候选/来源页真改图） | ✅ 已完成（`core/lens.py`，复用 SERPER_KEY）|
  | P0 | fact 标记 page_only / snippet_only | ✅ 已完成（含 relation 硬 override）|
  | P0 | discovered_entities 进 expansion_seeds | ✅ 已完成 |
  | P0 | closure richness 停止条件 | ✅ 已完成 |
  | P0 | 桥接提升 + 3-hop multi_hop | ✅ 已完成 |
  | P0 | ultra_long bucket（≥7 步工具链） | ✅ 已完成 |
  | P0 | step3 整合到根目录，删 experimental/ | ✅ 已完成 |
  | P1 | code skill tag + 配额 | ✅ 已完成（6 类 skill tag）|
  | P1 | local_artifacts（bbox_areas / layout_relations） | ✅ 已完成 |
  | P2 | 5 档 search_views（tight/pad20/pad40/context/full_scene） | ✅ 已完成。pad40 作备选，Lens 仍用 pad20 |
  | P2 | 链可读性判别器（过滤拼凑感强的 3-hop） | ✅ 已完成（`step3b_readability_filter.py`，flow/motivation/naturalness 三维评分，≤2 拦截） |
  | P2 | LLM 别名归一（避免桥接重复） | ✅ 已完成（`_llm_canonicalize_aliases`，Step2 的 2b-5b 阶段） |
  | P1 | visit_heavy 先验可答过滤 | ✅ 已完成（`_is_prior_answerable_visit`，25 种强先验 relation 降级）|
  | P1 | sports person proposal 第二遍 | ✅ 已完成（2×2 tile + person-only prompt，sports_03 从 0→6 person）|
  | P1 | API 调用缓存（Serper + Jina disk memoization） | ✅ 已完成（4 个 API 点全加 cache，rerun 4000x 加速） |
  | P2 | Step3b 可靠化（规则预筛 + hash 缓存 + LLM 复核） | ✅ 已完成（三层过滤架构） |
  | P2 | v0 hard split 落盘 | ✅ 已完成（`aggregate_hard_split.py`，133 题 JSONL + 审计统计） |
  | P1 | Step2 fan-out 分层预算（promotion gate + web-first visit-later） | ✅ 已完成（中小图省 10-20%，大图需进一步优化） |
  | P1 | image_heavy 拆 subtype（image_compare + image_resolve_follow） | ✅ 已完成（sports 8 图 image_heavy: 0→9） |
  | P1 | ultra_long 降级为次级 flag + bucket 优先级重排 | ✅ 已完成（工具依赖类 > 长度类） |
  | P1 | Step5 A1 攻击调度按 tool signature 动态决定 | ✅ 已完成（`_compute_a1_attacks`） |
  | P1 | v1 hard split | ✅ 已完成（128 题，image_heavy=54） |
  | P2 | 搜索词更精细（领域专属模板） | 待做 |
  | P2 | Jina visit 并发超时 + 失败快速跳过（当前 402 fallback 双倍延迟） | 待做（对大图 Step2 收益最大） |
  | P2 | 主搜索 per-entity visit 上限从 5 降到 2 | 待做 |
  | P3 | step4 verify 升级到完整模糊化审查 | 待做 |
  | P3 | Step1 bucket-aware 选图（预测 image_heavy 产出率，按配额筛图） | 待做 |
  | P3 | 全链路 memoization（含 Step5 agent step 中间结果） | 待做 |

  ### 验证周期 2 实测结果（stress suite 39 图 + sku 4 图）

  **数据分布：**
  - 39 张异质图（sports 8 / street 8 / poster 7 / retail 9 / mixed 7）+ sku 4 张
  - Step2 产出 412 个实体 / 3345 条三元组 / 67 个 `lens_visual_matches` 证据戳
  - Step3 产出 136 道 L3 题（含 **10 道 image_heavy** ★ 首次触发）

  **attack 结果：**

  | | Tier A (Qwen3-VL-30B) | Tier B (Gemini 3 Flash) |
  | --- | --- | --- |
  | 总 breach rate | 0/136 (0%) | 18/136 (13.2%) |
  | image_heavy × no_image_search | 0/10 (0%) | **0/8 (0.0%)** |
  | visit_heavy × no_visit | 0/9 (0%) | 0/5 (0.0%) |
  | visit_heavy × no_tool | — | 4/9 (**44.4%**) ★ 新发现 |
  | 阈值 verdict | keep_B_path | **keep_B_path** |

  **关键发现：**
  1. **image_heavy verdict = keep_B_path**，不做 A'
  2. **visit_heavy × no_tool = 44.4%**：Step3 把"模型先验可答"的题错标成 visit_heavy（Jina 读过 ≠ 必须读）。是下一轮 targeted fix 的候选
  3. **QTDL 手工预测（60-80% breach）vs 实际（20% no_tool breach, 0% no_image_search breach）**：Gemini 在有工具时不靠 QTDL shortcut 答题，QTDL 不是 image_heavy 的主要风险
  4. **VLM type 盲区**：sports 类 84 个实体中仅 1 个 person（球员被分成 brand）→ image_heavy=0。非 bug，是 VLM 对运动场景的分类局限

  **v0 hard split：133 题**（156 total - 21 Tier B breached - 2 no_tool override on image_heavy）

  | category | total | image_heavy | ultra_long | chain_heavy | visit | code | standard |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | mixed | 23 | 1 | 9 | 6 | 0 | 0 | 7 |
  | sports | 23 | 0 | 7 | 7 | 2 | 0 | 7 |
  | poster | 22 | 2 | 7 | 4 | 2 | 0 | 7 |
  | street | 16 | 2 | 7 | 0 | 0 | 2 | 5 |
  | retail | 34 | 3 | 12 | 6 | 1 | 3 | 9 |
  | sku | 15 | 0 | 2 | 5 | 3 | 1 | 4 |
  | **TOTAL** | **133** | **8** | **44** | **28** | **8** | **6** | **39** |



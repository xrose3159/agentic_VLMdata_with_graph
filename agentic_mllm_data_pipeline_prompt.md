- ## 概述

  - 本 Pipeline 从候选图片出发，自动生成 **（图片, 问题, 答案, 工具序列）** 四元组的多模态 agentic 训练数据。数据分为三个难度等级（L1/L2/L3），难度逐级递增，模糊化程度和所需工具种类也逐级增加。
  - 四步流程：
    - **筛选高信息密度图片** — VLM 打分过滤
    - **异构证据图构建** — VLM 实体识别 → 多视图裁剪 → 确定性搜索 → 跨实体桥接 → 三元组提取 → graph richness 驱动的多轮扩展
    - **弱约束随机游走 + 后验归纳题型** — 异构求解图 → 自由扩展子图 → 后验枚举可闭合意图 → 不可约性检查 → QuestionFrame 语言化
    - **模糊化验证与修正** — 结构性校验 + VLM 模糊化审查 + 自动修正
  
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
  - ​        "description": "通过文本描述搜索图片，或通过图片进行反向搜索。",
  - ​        "parameters": {
  - ​            "search_type": "'text'（文字搜图）或 'reverse'（以图搜图），默认'text'",
  - ​            "query": "搜索查询字符串（text模式必填）",
  - ​            "image_url": "图片路径或URL（reverse模式必填）",
  - ​            "max_results": "最大返回结果数（默认10）"
  - ​        },
  - ​        "output": "图片结果列表，每条包含image_url和description"
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
  - ​        "description": "执行Python代码。支持图像处理（PIL, OpenCV）、数学计算、数据分析。输入图片已预加载为PIL Image对象。",
  - ​        "parameters": {
  - ​            "code": "要执行的Python代码（必填）"
  - ​        },
  - ​        "output": "代码执行结果（stdout输出、生成的图片等）",
  - ​        "preloaded": "original_image / original_image_N 已预加载为PIL Image对象",
  - ​        "packages": "PIL, NumPy, OpenCV, Matplotlib, SciPy, Pandas, SymPy"
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

    从每张筛选后的图片构建一张**足够厚的异构证据图**，使 Step3 的随机游走能自由探索、自然长出多种题型。

    不再以"找到一条 L3 链"为目标，而是以 **graph richness 达标**为停止条件。

  **方法选择：**

  - 实体提取：纯 VLM，直接输出 Bounding Box
  - 外部检索：Serper web_search + image_search(text)
  - 网页深读：Jina Reader（visit）
  - 搜索词：代码确定性生成（exact-name-first），不让 LLM 写搜索计划
  - 实体池：分两层（core_anchors 给 Step3 出题 / expansion_seeds 给 Step2 扩图，不限数量）
  - 不使用 YOLO / OCR / SAM 等额外模型

  **四种工具在 Step2 中的使用：**

  | 工具 | 用途 | 阶段 |
  | ---- | ---- | ---- |
  | web_search | 搜实体名获取硬事实（知识面板、搜索摘要） | 跨实体搜索 + 主搜索 + 扩展搜索 |
  | image_search(text) | 文字搜图验证实体身份 + 拿来源页 URL | 实体 resolution |
  | visit (Jina Reader) | 深读搜索结果页和 image_search 来源页 | 跨实体深读 + 主搜索深读 + resolution 来源页 |
  | code_interpreter | 多视图裁剪（tight / pad20 / context） | 实体提取后处理 |

  ### 流程

  #### 2a. 纯 VLM 实体提取

    将图片 base64 编码后发送给 VLM，输出每个实体的 `name`、`type`（brand/landmark/text/person/product/object）、`bbox`（0-1000 归一化）。VLM 同时输出图片整体描述。

  **后处理：**

  1. **实体去重**：名称相同（忽略大小写）只保留第一个
  2. **坐标转换**：归一化 bbox → 像素坐标
  3. **幻觉拦截**：bbox 面积占全图 >80% → 移除
  4. **位置描述**：`_bbox_to_location()` 从像素坐标计算（不依赖 VLM）
  5. **多视图裁剪（search_views）**：
     - `tight`：原 bbox 紧裁（Step3 位置引用）
     - `pad20`：四周扩 20%（保留上下文，用于未来 reverse search）
     - `context`：以实体为中心的大区域裁剪（场景级搜索）

    每个实体输出：`name`、`type`、`value`、`bbox`、`location_in_image`、`confidence`、`confidence_level`、`search_views: {tight, pad20, context}`

    实体数少于 3 个则跳过该图片。

  #### 2a-2. 实体 resolution（image_search + visit）

    对每个实体并行执行 `image_search(text, entity_name)`：

  - Serper Images API 返回图片标题 + 来源页 URL
  - 图片标题用于验证 VLM 识别是否靠谱（如搜 "Superior Light Beer" 返回 "Michelob ULTRA..."，确认了品牌身份）
  - 对 top-2 来源页调 `visit`（Jina Reader）深读，获取更多实体背景信息

    resolution 结果写入 `entity.resolution`：
  - `image_search_titles`：图片标题列表
  - `image_search_sources`：来源页 URL 和域名
  - `visited_pages`：深读的来源页内容摘要

    这些信息传给后续三元组提取 LLM 作为额外上下文。

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

  #### 2b-3.5. 合并桥接三元组 + 实体名规范化

  - 跨实体桥接三元组按 `(head, relation, tail)` 去重合并
  - 实体名规范化：漂移变体统一到 canonical name
  - 自环丢弃

  #### 2b-4. 多轮扩展搜索

  **停止条件（graph richness）：**

    满足以下任意两项即停止，或达到最多 4 轮：
  - fact triples 总数 >= 30
  - 可比较的 TIME/QUANTITY fact group >= 2
  - cross-entity bridge triples >= 3
  - canonical entities 总数 >= 15

  **扩展 frontier：**
  - 未搜过的 tail entities（按引用次数排序）
  - exact-name-first 搜索，不调 LLM

  **空间关系兜底：**
  - 扩展完成后，为无知识连接的实体对补充空间三元组（`located_near` 等）

  ### 输出

  - `output/entities/{img_id}.json`：
    - `entities`：完整实体列表（含 search_views、resolution 信息）
    - `core_anchors`：Step3 出题用的高置信锚点
    - `expansion_seeds`：Step2 扩图用的全量实体
    - `triples`：fact 三元组
    - `search_results`：web_search 结果
    - `image_description`、`domain`
  - `output/entities/crops/{img_id}/`：多视图裁剪图（E1.jpg / E1_pad20.jpg / E1_context.jpg）


  ## 第三步：分层问题生成（弱约束随机游走 + 后验归纳题型）

    实现文件：`experimental/random_walk_step3/trajectory_runtime.py`

    这是整个 Pipeline 最核心的步骤。**题型是游走的输出，不是输入。**

    核心流程：
    1. 从 step2 输出构建**异构求解图**（4 类节点 × 3 类边，带 resolve_mode / retrieval_mode）
    2. **弱约束自由游走**：从视觉锚点出发，逐步扩展证据子图（不是线性链）
    3. **后验归纳题型**：枚举子图中所有可闭合问题意图，选最佳
    4. **5 个不可约性检查**
    5. **4 个 hard bucket** 分类 + 全局配额选题
    6. 编译成 **QuestionFrame**，单次 LLM 调用语言化（按题型给好/坏例子引导）
    7. **确定性工具序列**编译（根据 resolve_mode / retrieval_mode 决定工具）

  ### 3a. 异构求解图（HeteroSolveGraph）

    从 step2 entity JSON 构建。

    **节点 4 类：**

  | 类型 | 含义 | 来源 |
  | ---- | ---- | ---- |
  | `full_image` | 整图（1 个） | — |
  | `region` | 图中实体裁剪区 | step2 entities（bbox / location / type / search_views） |
  | `entity` | 实体 canonical name | step2 entities |
  | `fact` | 知识事实值 | step2 triples 的 tail |

    **边 3 类：**

  | 类型 | 含义 | 关键属性 |
  | ---- | ---- | ------- |
  | `observe` | full_image → region | — |
  | `resolve` | region → entity | **resolve_mode**：`ocr_likely` 或 `image_search_needed` |
  | `retrieve` | entity → fact | **retrieval_mode**：`snippet_only` 或 `page_only` + askability / lexicalizability |

    `resolve_mode` 决定工具编译时用 code_interpreter(OCR) 还是 image_search(reverse)。
    `retrieval_mode` 决定是否需要 visit 深读（page_only 的 fact 必须 visit 才能获取）。

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

    **STOP 的 budget 检查（hard 难度下）：**

  - 如果子图中没有 `resolve_mode=image_search_needed` 的实体 → STOP 减分，继续探索
  - 如果子图中没有 `retrieval_mode=page_only` 的事实 → STOP 减分
  - 如果锚点数 < min_anchors 或没有 compute closure → STOP 减分

    这样 walker 在 hard 难度下不会在第一个简单 closure 上就停。

  ### 3c. 后验归纳题型（ClosureCompiler）

    游走过程中和停止时，枚举当前子图中所有可闭合的问题意图：

  | 闭合类型 | 条件 | Level |
  | -------- | ---- | ----- |
  | `read` | region 实体可直接识别 | L1 |
  | `lookup` | region → entity → fact 链路完整 | L2 |
  | `compare` | 两个 entity 有同类 TIME/QUANTITY fact | L3 |
  | `compare_then_follow` | compare + 赢家还有别的 fact | L3 |
  | `rank` | 3+ 个 entity 有同类可排序 fact | L3 |
  | `set_merge` | 两个 entity 共享同一 fact target | L3 |

  ### 3d. 不可约性检查

  1. **answer_uniqueness** — 非 yes/no/unknown/太泛
  2. **realizable_question** — 锚点有视觉描述
  3. **no_python_shortcut** — L3 去掉 compute 后答案必须变
  4. **answer_not_visible** — 答案不是图中直接可见的实体名
  5. **no_branch_shortcut** — 删掉任一分支后答案必须变

  ### 3e. 4 个 Hard Bucket + 全局配额

    每个 L3 closure 编译工具序列后，打 **hard bucket** 标签：

  | Bucket | 条件 |
  | ------ | ---- |
  | `all_tools` | 4 种工具（code + web_search + image_search + visit）全部出现 |
  | `image_heavy` | 有 image_search 且 ≥2 个视觉锚点 |
  | `visit_heavy` | 包含 visit 步骤（retrieval_mode=page_only 的 fact） |
  | `code_heavy` | ≥2 类 code 操作 |
  | `standard` | 以上都不满足 |

    **全局选题按 bucket 配额**（优先 L3，再 L2，最后 L1）：

  | Level | 配额 |
  | ----- | ---- |
  | L3 image_heavy | 1-2 题 |
  | L3 visit_heavy | 1 题 |
  | L3 code_heavy | 1 题 |
  | L3 all_tools | 1 题 |
  | L3 standard | 2 题 |
  | L2 | 3 题 |
  | L1 | 4 题 |

  ### 3f. QuestionFrame + 语言化

    不把原始子图丢给 LLM。先编译成结构化 QuestionFrame，再单次 LLM 调用语言化。

    **按题型给引导和好/坏例子：**

  - **L1 read**：问"XX上写的是什么文字/品牌"，不要只说"XX是什么"
  - **L2 lookup**：先用视觉特征描述，再用日常口语问需要查资料的问题。不堆砌形容词。
  - **L3 compare**：用外观差异区分两个目标（球衣号码/颜色），问法日常化（"年龄更大"而非"出生日期更早"）
  - **L3 rank**：简洁列出几个目标，问谁最XX
  - **L3 set_merge**：问两个目标的共同点

    **Postcheck**：只查名称泄露。

  ### 3g. 工具序列（确定性编译）

    从 resolve_mode / retrieval_mode **确定性**生成工具序列：

  | 条件 | 工具 |
  | ---- | ---- |
  | observe(region) | code_interpreter（裁剪 bbox） |
  | resolve_mode=ocr_likely | code_interpreter（OCR） |
  | resolve_mode=image_search_needed | **image_search**（反向搜索） |
  | retrieve, retrieval_mode=snippet_only | web_search |
  | retrieve, retrieval_mode=page_only | web_search + **visit** |
  | compute | code_interpreter（Python 比较/排序/集合运算） |

    这样工具序列的多样性由 Step2 的标记决定，不是硬编码。

  ### 输出

  - `output/questions/{img_id}.json`：L1/L2/L3 题目，每题含 hard_bucket 标签
  - 每题包含：question / answer / tool_sequence / level / family / hard_bucket / reasoning_path

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
  | 图谱太小 | expansion_seeds 不限数量 + graph richness 停止条件 | step2_enrich |
  | 搜索 crop 太小 | 三档 search_views（tight/pad20/context） | step2_enrich |
  | 搜索词不稳定 | exact-name-first 确定性搜索，删 LLM 搜索计划 | step2_enrich |
  | 实体名不匹配 | 类型感知的模糊关键词匹配 | _build_entity_index / _match_entity |
  | 实体名漂移 | 后处理启发式 canonical name 替换 | _normalize_triple_entities |
  | LLM 泄露实体名 | postcheck_name_leak 检查图中实体名 + 隐藏节点名 | trajectory_runtime |
  | 工具序列造假 | 从轨迹确定性编译，不让 LLM 规划 | compile_tool_plan |
  | 答案太泛 | answer_uniqueness 不可约性检查 | check_irreducibility |
  | 答案看图直接可得 | answer_not_visible 检查 | check_irreducibility |
  | L3 题去掉比较也能答 | no_python_shortcut 检查 | check_irreducibility |
  | 多分支题删分支还能答 | no_branch_shortcut 检查 | check_irreducibility |
  | 锚点无法视觉描述 | realizable_question 检查 | check_irreducibility |
  | 题型死板 | 后验归纳而非预设 family | enumerate_closures |
  | 难度靠 hop 数 | 偏好向量 + budget deficit 驱动 | DifficultyProfile |
  | 实体间缺乏跨域关联 | 跨实体穷举 C(n,2) 搜索 + 桥接三元组 | _find_cross_entity_relations |
  | 位置描述不准确 | 由代码从 bbox 计算，不依赖 VLM | _bbox_to_location |
  | 视觉指代太模糊 | 方位 + 视觉类别（entity_type + relation 推断） | visual_descriptor |
  
  ## 每张图片的预期产出
  
  | 级别 | 题型（后验归纳） | 工具签名 | 目标数量 | 模糊化 |
  | ---- | ---------- | -------- | -------- | ------ |
  | L1 | read（纯视觉识别） | code_interpreter 或 image_search | ≤4 | 无（实体名是答案） |
  | L2 | lookup（识别+查知识） | code_interpreter + web_search | ≤3 | 图中实体用视觉描述 |
  | L3 | compare / compare_then_follow / rank / set_merge | code_interpreter×2 + web_search×2 + code_interpreter(python) | ≤4 | 视觉描述 + 隐藏中间值 |
  | 总计 | | | 约 8-11 题 | |

    题型由**弱约束随机游走后验归纳**产出，不预设。工具序列由轨迹**确定性编译**，不由 LLM 规划。实际数量取决于异构证据图的 richness 和 closure 质量，宁缺勿滥。
  
  ## 文件结构
  
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
  
    {
  
  ​      "id": "img_0010_L2_01",
  
  ​      "image_id": "img_0010",
  
  ​      "image_path": "output/images/img_0010.jpg",
  
  ​      "level": 2,
  
  ​      "question": "图中左下角带有金色拱门标志的品牌，其总部位于哪个美国城市？",
  
  ​      "answer": "Chicago, Illinois",
  
  ​      "tool_sequence": [
  
  ​          {"step": 1, "tool": "code_interpreter", "action": "裁剪并识别左下角金色拱门标志", "input": "...", "expected_output": "McDonald's"},
  
  ​          {"step": 2, "tool": "web_search", "action": "搜索该品牌总部位置", "input": "McDonald's headquarters", "expected_output": "Chicago, Illinois"}
  
  ​      ],
  
  ​      "reasoning_path": {
  
  ​          "chain": "McDonald's →[headquartered_in] Chicago, Illinois",
  
  ​          "start": "McDonald's",
  
  ​          "end": "Chicago, Illinois",
  
  ​          "hops": ["headquartered_in→Chicago, Illinois"],
  
  ​          "depth": 1,
  
  ​          "in_image_count": 1
  
  ​      },
  
  ​      "obfuscation_applied": true,
  
  ​      "obfuscated_entities": ["McDonald's"],
  
  ​      "verified": true,
  
  ​      "domain": "retail"
  
    }


  ## 已知局限与下一步改进计划

  ### 核心问题：题不够难

    当前的"难"本质上还是靠 hop 数和 compare 来撑，真正的有效求解长度不够。表现为：

  - walker 遇到第一个干净的 compare_then_follow 就停了
  - image_search 在 Step2 被消耗掉了，没有保留为可游走结构
  - visit 的内容被压成三元组后，page-only 的信息丢失了
  - code 操作只有 crop/OCR/argmin 这几种单一模式

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

  ### 改进方向 5：Step2 停止条件从 graph richness 改成 closure richness

  **当前：** fact triples >= 30 / comparable groups >= 2 / bridge >= 3 / entities >= 15，满足 2 项就停。

  **改成：**

  - `image_resolved_anchors >= 3`
  - `page_only_facts >= 6`
  - `compare_ready_pairs >= 4`
  - `rank_ready_triplets >= 2`
  - `code_ready_artifacts >= 4`
  - `cross_anchor_shared_nodes >= 3`

    满足任意 3 项再停。目标从"造出很多三元组"变成"造出很多可闭合的 hard motif"。

  ### 改进方向 6：Step2 保留 local_artifacts

    不只关心外部 fact，还保留本地图像工件：

  - `ocr_text_blocks`、`numeric_labels`、`bbox_area`、`relative_size`、`dominant_color`、`layout_relations`

    这样 Step3 能生成真正 code-heavy 的题：
  - "三个价格牌里数值最低的那个品牌总部在哪？"
  - "画面中面积最大的那块广告牌对应品牌成立于哪一年？"

  ### 改进方向 7：工具定义补全

  **image_search 输出补字段：**

    ```json
    {
      "image_url": "...",
      "title": "...",
      "description": "...",
      "source_page_url": "...",
      "source_domain": "..."
    }
    ```

  **code_interpreter 能力说明补全：**

  - 明确支持 easyocr / pytesseract
  - 或者不把 OCR 伪装成 code，而是只把 code 用于预处理、裁剪、比较、解析

  ### 实现优先级

  | 优先级 | 改进 | 预期收益 |
  | ------ | ---- | ------- |
  | P0 | resolve_mode + retrieval_mode 标记 | 工具序列不可约性 |
  | P0 | 4 个 hard bucket + 全局配额 | 题目工具覆盖 |
  | P1 | image_search reverse 进 Step2 主环 | image-heavy 题 |
  | P1 | fact 标记 page_only / snippet_only | visit-heavy 题 |
  | P1 | code skill tag + 配额 | code-heavy 题 |
  | P2 | local_artifacts | 更丰富的 code 操作 |
  | P2 | closure richness 停止条件 | Step2 产出质量 |
  | P2 | 5 档 search_views | reverse search 精度 |

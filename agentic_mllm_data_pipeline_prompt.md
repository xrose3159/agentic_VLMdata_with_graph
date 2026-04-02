- ## 概述

  本 Pipeline 从候选图片出发，自动生成 **（图片, 问题, 答案, 工具序列）** 四元组的多模态 agentic 训练数据。数据分为三个难度等级（L1/L2/L3），难度逐级递增，模糊化程度和所需工具种类也逐级增加。

  四步流程：

  1. **筛选高信息密度图片** — VLM 打分过滤
  2. **实体提取 +** **知识图谱****扩展** — VLM 实体识别 → 跨实体关联发现 → 搜索计划 → 真实搜索 → 三元组提取 → 桥接合并 → 多轮扩展
  3. **分层问题生成** — 找链 → 骨架（预模糊化 + 预工具规划）→ LLM 润色
  4. **模糊化验证与修正** — 结构性校验 + VLM 模糊化审查 + 自动修正

  ## 工具定义

  本 Pipeline 生成的数据围绕以下四种工具，所有生成的问题必须严格使用这些工具定义：

  TOOLS = {

  ​    "web_search": {

  ​        "description": "搜索网络获取文本信息。用于查找事实、规格参数、新闻、价格等。",

  ​        "parameters": {

  ​            "query": "搜索查询字符串（必填）",

  ​            "max_results": "最大返回结果数（默认10）"

  ​        },

  ​        "output": "搜索结果列表，每条包含title, url, snippet"

  ​    },

  ​    "image_search": {

  ​        "description": "通过文本描述搜索图片，或通过图片进行反向搜索。",

  ​        "parameters": {

  ​            "search_type": "'text'（文字搜图）或 'reverse'（以图搜图），默认'text'",

  ​            "query": "搜索查询字符串（text模式必填）",

  ​            "image_url": "图片路径或URL（reverse模式必填）",

  ​            "max_results": "最大返回结果数（默认10）"

  ​        },

  ​        "output": "图片结果列表，每条包含image_url和description"

  ​    },

  ​    "visit": {

  ​        "description": "访问指定网页并提取主要内容。用于从搜索结果中获取详细信息。",

  ​        "parameters": {

  ​            "url": "完整网页URL（必须以http://或https://开头）",

  ​            "goal": "你希望从该页面获取什么信息（辅助提取）"

  ​        },

  ​        "output": "网页的主要文本内容"

  ​    },

  ​    "code_interpreter": {

  ​        "description": "执行Python代码。支持图像处理（PIL, OpenCV）、数学计算、数据分析。输入图片已预加载为PIL Image对象。",

  ​        "parameters": {

  ​            "code": "要执行的Python代码（必填）"

  ​        },

  ​        "output": "代码执行结果（stdout输出、生成的图片等）",

  ​        "preloaded": "original_image / original_image_N 已预加载为PIL Image对象",

  ​        "packages": "PIL, NumPy, OpenCV, Matplotlib, SciPy, Pandas, SymPy"

  ​    }

  }

  ## 第一步：筛选高信息密度图片（step1_filter.py）

  ### 目标

  从 `images/` 目录的候选图片中筛选出信息密度足够高、适合生成 agentic 训练数据的图片。

  ### 流程

  1. 遍历 `images/` 目录下所有图片（jpg/jpeg/png/webp）
  2. 对每张图片，将其 base64 编码后发送给 VLM，按 5 个维度打分（1-5 分）：
     1. **实体丰富度（entity_richness）**：图中有多少个可独立识别的实体
     2. **信息层次性（detail_depth）**：是否有需要放大/裁剪才能看清的细节
     3. **外部知识关联（external_linkage）**：图中实体是否能关联到可搜索的外部知识
     4. **多实体关系（entity_relations）**：实体之间是否存在可推理关系
     5. **自然真实性（naturalness）**：图片是否来自真实场景

  - 这里到底算什么样的图片算好图片？目前选择信息密度高的图片感觉不太自然

  high-information -> high-agentic-potential

  信息丰富度打分（step1-1）+ agent 潜力打分（step1-2）

  step1-2：让模型根据图片生成 agent task 草案，再反推潜力：

   给 VLM 看图，让它输出 2–3 个最小任务草案，每个草案只要这几个字段：

  - `user_goal`
  - `required_tools`
  - `why_tools_needed`
  - `expected_answer_type`

   再对任务草案进行打分。

  1. 筛选条件：**总分 ≥ 18** 且 **每项 ≥ 3**
  2. VLM 返回的 `pass` 字段由代码重新校正（不信任 VLM 的布尔判断）
  3. 通过的图片复制到 `output/images/`，同时记录评分统计和类别分布

  ### 输出

  - `output/images/` — 筛选通过的图片
  - `output/stats/filter_scores.json` — 所有图片的评分、类别分布

## 第二步：实体提取与知识图谱扩展（step2_enrich.py）

### 目标

对每张筛选后的图片提取可识别实体，并通过真实网络搜索构建以图中实体为起点的知识三元组图谱，确保图谱中存在多跳链路（L3 所需）。

**核心设计：LLM 只负责提取原子三元组 (head, relation, tail)，不负责构建多跳知识链。多跳链的发现由第三步的代码（DFS）完成。**

**Step 2 的方法选择（本版固定）：**

- 实体提取使用 **纯 VLM（视觉语言模型）**，直接输出 Bounding Box
- 外部检索使用 **Serper (Google Search)**
- 网页深度读取使用 **Jina Reader**
- **不使用 YOLO / OCR / SAM 等额外模型**
- **不使用 image_search（不做文搜图/图搜图）**

### 流程

#### 2a. 纯 VLM 实体提取

将图片 base64 编码后发送给 VLM，要求输出每个实体的：
- `name`：可搜索的实体名称（如 "McDonald's"、"Bank of America"）
- `type`：类型（brand/landmark/text/person/product/object）
- `bbox`：边界框坐标 `[x_min, y_min, x_max, y_max]`，范围 0-1000 归一化。

VLM 同时输出图片整体描述（100字以上）。

**后处理：**
1. **实体去重**：名称完全相同（忽略大小写）的实体只保留第一个
2. **坐标转换**：将 0-1000 归一化 bbox 转为实际像素坐标
3. **幻觉拦截**：bbox 面积占全图 >80% → 判定为幻觉，移除
4. **位置描述**：由 `_bbox_to_location()` 根据实际像素坐标计算，输出如"画面下方偏左"、"画面顶部最右侧"（不依赖 VLM 感知，保证准确性和格式一致性）
5. **图片裁剪**：按像素坐标裁剪实体小图，保存到 `crops/` 目录

每个实体最终输出：
- `name`：可搜索的实体名称
- `type`：类型
- `value`：具体值（价格、规格等；由规则提取，无则 null）
- `bbox`：像素坐标 `[x1, y1, x2, y2]`
- `location_in_image`：自然语言位置描述（由代码从 bbox 计算，用于 step3 模糊化）
- `confidence`：固定 0.9（VLM 输出，无 SAM3 置信度）
- `confidence_level`：high
- `crop_path`：裁剪后的实体图片路径

实体数少于 3 个则跳过该图片。

#### 2b-0. 高置信度实体选取 + 跨实体关联发现

实体提取完成后，先选取高置信度实体，再发现它们之间的真实世界关联，生成桥接三元组。

**高置信度实体选取：**
- 从所有识别实体中过滤 `confidence_level` 为 `high` 或 `medium` 的实体，最多取前 5 个
- 若过滤后不足 1 个，则直接取前 5 个实体
- **后续所有步骤（跨实体搜索、搜索计划、三元组提取、Motif 探测）均只针对这 5 个实体**，不处理其余实体

**跨实体关联发现流程（穷举 C(n,2) 对）：**

1. 对 high_conf 实体进行两两枚举，生成所有 C(n,2) 实体对（5 个实体最多 10 对）
2. 为每对自动生成搜索查询：`"{A} {B}"`（只搜实体名本身，让搜索引擎自然返回两者的共现内容；效果不好时可改为 LLM 生成查询词）
3. **并行搜索所有对**（不依赖 VLM 选对，确保两两之间都有搜索覆盖）
4. 每条搜索额外调用 Jina Reader 深读 top-1 网页（最多 2000 字）
5. 将所有对的搜索结果拼装成文本，交给 LLM 提取桥接三元组，重点寻找：
   - 两个图中实体之间的**直接关系** `(A, relation, B)`
   - 通过**桥节点**的间接关联 `(A, rel, X) + (B, rel, X)`
6. 所有桥接三元组标记 `"source": "cross_entity"`

**为什么穷举而非 VLM 选对：**
- VLM 倾向于只选 2-4 对"明显相关"的实体对，忽略不明显但可能存在关联的组合
- 穷举确保图中任意两个实体之间都有搜索结果，为 Step 3 的 Bridge Motif 提供更多素材
- C(5,2)=10 对，并行搜索代价可接受

**为什么需要这个阶段：**
- 各实体独立搜索时，搜索计划围绕单个实体展开，难以发现实体之间的跨域关联
- 例如 Times Square 的 Toshiba 和 Maxell 广告牌——各自搜索时不会刻意搜索二者的关系，但它们都是日本电子品牌这一关联在跨实体阶段可以被发现
- 桥接三元组可直接为 Step 3 提供跨实体的 L2/L3 链路

#### 2b-1. LLM 生成搜索计划

选取高置信度实体（最多 5 个），LLM 为每个实体生成 2-3 条搜索查询。要求：

- 查询覆盖不同方向（基本信息、关联公司/人物、历史/地理背景）
- 考虑实体间的交叉关联
- 通用词（如"红色窗帘"）标记 `skip=true`
- 查询词用实体相关的语言（英文实体用英文搜）

#### 2b-2. 执行真实搜索（Serper）

按计划逐条执行搜索，每条查询返回：

- 搜索摘要/答案框（如有）
- Google Knowledge Graph 信息（如有）
- 最多 5 条搜索结果（标题、URL、内容片段）
- 每次搜索间隔 0.3s 以控制速率

#### 2b-3. 三元组提取

将所有搜索结果拼成文本，交给 LLM 一次性提取事实三元组。每个三元组：

{

    "head": "头实体名",
    
    "relation": "关系类型（如 located_at, founded_by 等）",
    
    "tail": "尾实体名",
    
    "fact": "一句话描述这个事实",
    
    "source_snippet": "搜索结果中的佐证原文片段"

}

**关键 prompt 规则：**

- 每个三元组必须有搜索结果佐证（`source_snippet`），不要编造
- 实体名要具体、准确（如"Imperial Theatre"而非"某剧院"）
- **命名一致性**：图中实体的 head/tail 必须使用实体列表中的原始名称，禁止加地点后缀或任何变体（如图中是"McDonald's"，不得写为"McDonald's Times Square"）
- **不仅提取「图中实体→外部知识」的关系，也要提取「外部知识→外部知识」的关系**（这样才能形成多跳链）
- head 和 tail 不能是同一个实体

#### 2b-3.5. 合并跨实体桥接三元组

将 2b-0 阶段生成的跨实体桥接三元组与 2b-3 提取的常规三元组合并：
- 基于 `(head, relation, tail)` 小写去重，避免重复
- 桥接三元组保留 `"source": "cross_entity"` 标记，便于后续分析来源
- 合并后的三元组统一进入 2b-4 的多轮扩展搜索

#### 2b-4 后. 实体名规范化（启发式合并）

所有轮次完成后，对三元组中漂移的实体名进行后处理统一：
- 若某个 head/tail 包含图中实体名（或被其包含），且图中实体名长度 ≥ 3，则替换为图中实体的 canonical name
- 按实体名长度降序匹配，优先匹配更长的名称，避免误匹配
- 规范化后若 head == tail（产生自环），该三元组丢弃
- 规范化后再次去重

#### 2b-4. 多轮扩展搜索

检查当前三元组图中是否存在 length ≥ 3 的链（DFS 快速检测）。如果没有，进行扩展搜索：

**循环逻辑（最多 3 轮）：**

1. 检查当前三元组图是否已有 L3 链 → 有则停止
2. 从已有三元组的 tail 实体中选出未搜过的、非图中实体（按引用次数排序，最多 6 个）
3. 将这些 tail 实体及其来源三元组上下文交给 LLM，生成每个实体一条搜索查询
4. 执行 SerpAPI 搜索
5. 从搜索结果提取新三元组（**提示 LLM 复用已有实体名以保持链路连通**）
6. 基于 `(head, relation, tail)` 去重合并到已有三元组中
7. 如果本轮无可扩展实体或搜索无结果 → 停止

### 输出

- `output/entities/{img_id}.json` — 包含实体列表、搜索计划、搜索结果、三元组

## 第三步：分层问题生成（step3_generate.py）

  这是整个 Pipeline 最核心的步骤。**核心思路：预先定义训练目标推理能力，用三种「推理拓扑模板（Reasoning Motif）」替代盲目的链枚举。**

  流程：3a 建图 → 3b 三种探测器探测 Motif → 3c Motif 转结构化骨架 → 3d LLM 按 Motif 类型润色

  ### 3a. 建 NetworkX 有向图（_build_nx_graph）

  将三元组构建为 NetworkX `DiGraph`，每个节点携带 `in_image`、`location`、`name` 属性。

  **实体匹配**（判断节点是否为图中实体）：

  - **精确匹配**：三元组实体名（小写）与 VLM 提取的实体名完全一致
  - **品牌关键词前缀匹配**：图中品牌实体关键词只匹配以该词**开头**的节点名（防止 "fender" 误匹配 "leo fender"）
  - **人名/地名宽松匹配**：关键词作为独立单词或子串出现即匹配
  - **双向包含匹配**：图中实体全名包含在节点名中，或节点名包含在实体全名中（如 "Billy Elliot" 匹配 "Billy Elliot the Musical"）

  ### 3b. 三种推理拓扑探测器（find_motifs）

  #### 探测器一：交集桥接（Bridge, L2）

  **拓扑形态**：`[图中实体 A] →rel_1→ [答案 T] ←rel_2← [图中实体 B]`

  两个图中实体通过各自不同的关系共同指向同一个图外节点。

  **探测逻辑**：遍历所有图外节点，找其前驱中有 ≥2 个图中实体的节点。

  **对应题型**："请找出同时满足条件一（与A有rel_1关系）和条件二（与B有rel_2关系）的实体。"

  **工具序列**：`code_interpreter`（识别两个视觉实体）→ `web_search`（分别搜索，寻找共同交集）

  **实例**（Times Square 图）：Bank of America `→[has_billboard_presence_in]→` **Times Square** `←[located_in]←` Father Duffy Square

---

  #### 探测器二：深度跳跃（MultiHop, L3）

  **拓扑形态**：`[图中实体 A] →rel_1→ [中间节点 X，带限定词 Y] →rel_2→ [答案 T]`

  经典多跳路径，要求中间节点 X 至少有 2 条出边——一条通往答案，另一条作为**判别属性（discriminator）**，用于在问题中唯一确定 X 而不泄露其名称。

  **探测逻辑**：从图中节点出发，遍历长度为 2 的路径，检查中间节点出度 ≥ 2。

  **对应题型**："图中左下角品牌的创始兄弟（通过其另一属性唯一锁定）所引入的管理体系叫什么名字？"

  **工具序列**：`code_interpreter`（识别视觉实体）→ `web_search`（找中间节点）→ `visit`（深读查答案）

  **实例**（Times Square 图）：McDonald's `→[founded_by]→` Dick & Mac McDonald（discriminator：opened McDonald's）`→[introduced]→` **Speedee Service System**

---

  #### 探测器三：平行对比（Comparative, L2）

  **拓扑形态**：`[图中实体 A] →数值型rel→ [值1]` vs `[图中实体 B] →同类rel→ [值2]`

  两个图中实体均有同类数值/时间关系，问题要求 Agent 并行查询后比较得出结论。

  **探测逻辑**：按关系名分桶，筛选含明确数值关系的边，找两个图中节点都有同类边的组合。

  **可比关系白名单**（精确匹配，排除 founded_by 等非数值关系）：`founded_in_year`、`opened_in_year`、`established_in_year`、`born_in`、`population`、`revenue`、`height` 等。

  **对应题型**："图中左侧品牌和右侧品牌，哪个创立时间更早？"

  **工具序列**：`code_interpreter`（识别两个视觉实体）→ `web_search`（分别搜索）→ `code_interpreter`（Python 代码比较数值）

  **实例**（Times Square 图）：McDonald's `→[founded_in_year]→` 1940 vs Bank of America `→[founded_in_year]→` 1904

---

  ### 3c. Motif → 结构化骨架 JSON（motif_to_skeleton）

  每个 Motif 转为标准化的骨架 JSON，传给 LLM 润色。**图中实体在此处完成代码层模糊化，LLM 从未见到真实名称。**

  ```json
  {
    "motif_type": "bridge_intersection",
    "difficulty": "L2",
    "visual_anchors": {
      "Entity_A": "[图中: 画面下方最右侧]",
      "Entity_B": "[图中: 画面底部]"
    },
    "reasoning_graph": {
      "condition_1": "[图中: 画面下方最右侧] →[has_billboard_presence_in]→ [隐藏目标]",
      "condition_2": "[图中: 画面底部] →[located_in]→ [隐藏目标]",
      "fact_1": "Bank of America has billboard advertising presence in Times Square",
      "fact_2": "Father Duffy Square is the northern triangle of Times Square"
    },
    "target_answer": "Times Square",
    "tool_plan": [
      {"tool": "code_interpreter", "reason": "裁剪识别 [图中: 画面下方最右侧] 和 [图中: 画面底部]"},
      {"tool": "web_search", "reason": "分别搜索两个关系，寻找共同交集"}
    ]
  }
  ```

  ### 3d. LLM 润色（polish_level）

  将同类型 Motif 的骨架打包，连同图片一起交给 VLM，按 Motif 专属规则润色为自然语言问题。

  **每种 Motif 有独立的润色指令：**

  - **Bridge**：呈现为「满足条件一 AND 条件二的是什么」结构
  - **MultiHop**：discriminator 嵌入为定语从句，限定中间节点唯一性
  - **Comparative**：answer 字段写出明确比较结论（含两个值）

  **共同规则：**

  - `[图中: ...]` 用自然位置描述引用，**绝对禁止写出真实名称**
  - `[隐藏]`/`[隐藏中间节点]` 用关系描述引用，**绝对禁止写出真实名称**
  - 只能有一个问号，允许「铺垫句 + 问句」结构
  - 严格按骨架预规划的工具链填写，禁止添加额外步骤

  **L1 纯视觉题**（独立生成，不经过 Motif 探测）：

  - 2 道 `code_interpreter` 题（OCR/裁剪/放大/计数）
  - 2 道 `image_search` 题（反向图搜识别视觉元素）

  **映射关系：** L1 = 纯视觉题 | L2 = Bridge + Comparative | L3 = MultiHop


  ### 聚合输出

  将所有图片的问题聚合为 JSONL 格式：

  - `output/final/level_1_all.jsonl`
  - `output/final/level_2_all.jsonl`
  - `output/final/level_3_all.jsonl`
  - `output/final/all_questions.jsonl`
  - `output/stats/question_stats.json`

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

  ​                        ↓ (逐张提交)

  ┌─── step3 线程池（workers 数，无 API 限制）──────────┐

  │  img_01: 找链 → 骨架 → LLM润色 → 输出问题          │

  │  img_02: 找链 → 骨架 → LLM润色 → 输出问题          │

  └─────────────────────────────────────────────────────┘

  ### 检查点机制

  每步每张图片完成后保存检查点到 `output/.checkpoints/step{N}/{img_id}.json`，重跑时自动跳过已完成的。需要重新生成时，手动删除对应检查点。

  ## 数据质量保障机制总结

  | 问题                          | 解决方案                                                | 实现位置                         |
  | ----------------------------- | ------------------------------------------------------- | -------------------------------- |
  | 实体名不匹配（VLM vs 三元组） | 类型感知的模糊关键词匹配，品牌前缀/人名宽松             | _build_entity_index / _match_entity |
  | LLM 泄露被模糊化的实体名      | 代码层预模糊化，LLM 从未见过真实名                      | motif_to_skeleton                   |
  | 虚假/凑数的工具序列           | 工具序列由 Motif 拓扑结构决定，LLM 只填 action/input   | motif_to_skeleton                   |
  | 布尔/太短/纯数字答案          | _answer_ok() 过滤不合格答案节点                         | _find_bridge/multihop_motifs        |
  | 重复答案                      | 各探测器内部 seen_answers 去重                          | _find_bridge/multihop_motifs        |
  | L2/L3 不引用图片              | Bridge/Comparative 强制双视觉锚点；MultiHop 强制起点    | 探测器约束条件                      |
  | L3 图片参与度低               | MultiHop 要求起点必须为图中实体                         | _find_multihop_motifs               |
  | Comparative 误匹配非数值关系  | 可比关系白名单精确匹配，排除 founded_by 等关系          | _is_comparable_relation             |
  | L3 工具种类单一               | MultiHop 工具链固定为 code_interpreter→web_search→visit | motif_to_skeleton                   |
  | 问题句式不自然                | 允许铺垫句+问句，只限一个问号                           | POLISH_PROMPT                      |
  | 品牌关键词误匹配人名          | 品牌关键词仅前缀匹配（startswith）                      | _match_image_entity                |
  | 三元组图深度不够              | 多轮扩展搜索直到出现 L3 链或达 3 轮                     | step2._extend_search               |
  | VLM 重复输出同一实体          | Prompt 约束 + 解析后按名称去重（保留首个）              | extract_entities_vlm               |
  | 三元组实体名漂移（变体/后缀） | Prompt 命名约束 + 后处理启发式 canonical name 替换      | _normalize_triple_entities         |
  | 位置描述不准确/含外观信息     | 位置描述由代码从 bbox 计算，不依赖 VLM 感知             | _bbox_to_location                  |
  | 实体间缺乏跨域关联            | 跨实体关联发现阶段，VLM 分析共现原因并搜索实体对关系    | step2._find_cross_entity_relations |

  ## 每张图片的预期产出

  | 级别 | 类型       | 推理跳数         | 目标数量 | 模糊化                          |
  | ---- | ---------- | ---------------- | -------- | ------------------------------- |
  | L1   | 纯视觉题       | 0 跳    | ~4  | 无                              | code_interpreter / image_search     |
  | L2   | 交集桥接题     | 1-2 跳  | ≤3  | 双视觉锚点均用位置描述          | code_interpreter → web_search       |
  | L2   | 平行对比题     | 1 跳×2  | ≤3  | 双视觉锚点均用位置描述          | code_interpreter → web_search × 2 → code_interpreter |
  | L3   | 深度跳跃题     | 2 跳    | ≤3  | 图中位置描述 + 中间节点关系描述 | code_interpreter → web_search → visit |
  | 总计 |                |         | ~13 |                                 |                                     |

  级别由 **Motif 类型**决定，工具序列由骨架中的 `tool_plan` 字段代码确定性生成，不由 LLM 自由规划。实际数量取决于知识图谱中 Motif 的存在情况，宁缺勿滥。

  ## 输出文件结构

  output/

  ├── images/                     # 筛选后的图片

  │   ├── img_0010.jpg

  │   └── ...

  ├── entities/                   # 每张图的实体 + 三元组

  │   ├── img_0010.json           # 含 entities, triples, search_results

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

  ​    ├── step1/

  ​    ├── step2/

  ​    ├── step3/

  ​    └── step4/

  ## 单条最终数据格式（JSONL 每行）

  {

  ​    "id": "img_0010_L2_01",

  ​    "image_id": "img_0010",

  ​    "image_path": "output/images/img_0010.jpg",

  ​    "level": 2,

  ​    "question": "图中左下角带有金色拱门标志的品牌，其总部位于哪个美国城市？",

  ​    "answer": "Chicago, Illinois",

  ​    "tool_sequence": [

  ​        {"step": 1, "tool": "code_interpreter", "action": "裁剪并识别左下角金色拱门标志", "input": "...", "expected_output": "McDonald's"},

  ​        {"step": 2, "tool": "web_search", "action": "搜索该品牌总部位置", "input": "McDonald's headquarters", "expected_output": "Chicago, Illinois"}

  ​    ],

  ​    "reasoning_path": {

  ​        "chain": "McDonald's →[headquartered_in] Chicago, Illinois",

  ​        "start": "McDonald's",

  ​        "end": "Chicago, Illinois",

  ​        "hops": ["headquartered_in→Chicago, Illinois"],

  ​        "depth": 1,

  ​        "in_image_count": 1

  ​    },

  ​    "obfuscation_applied": true,

  ​    "obfuscated_entities": ["McDonald's"],

  ​    "verified": true,

  ​    "domain": "retail"

  }

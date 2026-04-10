# Agentic Multimodal Training Data Pipeline — Introduction

## 我们在解决什么问题

训练一个能够真正"使用工具看世界"的多模态智能体，需要这样的训练样本：

- 模型**必须看图**才能开始推理（图片是信息来源，而非装饰）
- 模型**必须调用工具**才能完成回答（答案不能仅靠看图或仅靠记忆）
- 推理链**跨越多个知识跳**（不是单步查找，而是 A → B → C 的串联）
- 每一步工具调用都有明确的**必要性**（不是为了用工具而用工具）

这样的数据几乎不存在：

- **VQA 数据集**：看图回答，无工具调用
- **Agent 数据集**：大多是纯文本任务，没有图片
- **人工标注**：成本极高，规模受限，知识截止日期固定
- **纯 LLM 合成**：知识是幻觉出来的，没有真实佐证，容易被模型"记住答案"

我们的 pipeline 自动从任意图片集合出发，生成**有真实知识佐证、强制视觉感知、多跳工具推理**的训练数据。

---

## 核心思路

**图片是入口，知识图谱是桥梁，工具调用是手段。**

一张街景图里有麦当劳的招牌——这不只是"图里有个 logo"，而是通向一整张知识网络的入口：
麦当劳的创始人 → 创始人的出生地 → 那个城市现在的市长 → ……

我们的做法：
1. **识别图中实体**，获取它们在图中的精确位置和边界框
2. **对高价值实体做真实网络搜索**，从搜索结果中提取事实三元组，构建知识图谱
3. **在三元组层面补充语义字段**，包括 `tail_type`，并尽量产出 `normalized_value / unit`
4. **把图谱封装成 GraphEnv，在图上搜索 QuestionProgram**，而不是手写固定 motif
5. **由 verifier 做硬约束筛选，再由 utility ranker 做题目价值排序**
6. **对同一 program 搜索 visibility / microplan / paraphrase 变体**，最后只保留自然且高价值的问题

---

## 与现有方案的对比

| 维度 | 现有方案 | 本方案 |
|------|----------|--------|
| 知识来源 | LLM 内部记忆（可能幻觉） | 真实网络搜索 + 原文佐证 |
| 视觉参与 | 可选，图片常作背景 | 强制——问题中实体名被替换为位置描述，模型必须看图 |
| 推理深度 | 多为单跳 | 三级：L1（1跳）/ L2（2跳）/ L3（3跳以上） |
| 工具序列 | LLM 自由生成，易幻觉 | 由知识图谱结构**代码确定性生成**，LLM 无法凭空添加工具 |
| 数据规模 | 人工标注受限 | 全自动，任意图片集合均可扩展 |
| 跨实体关联 | 通常缺失 | 专门搜索图中实体对之间的真实关联，生成桥接三元组 |

---

## 关键设计决策及其理由

### 1. 为什么把实体名替换为位置描述（而非直接问"图中有什么"）

直接问"图中麦当劳的创始人是谁"，模型可以靠记忆回答，不需要看图。

把问题改成"图中**画面左下角**的品牌的创始人是谁"，模型就**必须**先识别那个位置的实体是什么，再去搜索，才能回答。这是真正的视觉感知驱动的推理。

位置描述由代码从 bbox 坐标精确计算（而非让 VLM 描述），保证描述准确、格式统一，不会带入外观信息泄露实体身份。

### 2. 为什么工具序列由代码生成而非 LLM 生成

让 LLM 自由规划工具序列，它可能：
- 为了"看起来更像 agent"而凑工具步骤
- 使用格式正确但逻辑错误的工具调用
- 在没有图中实体需要识别时仍加入 `image_search`

工具序列直接由知识链结构决定：有几个图中实体就有几次图像识别调用，有几跳就有几次搜索调用。这保证了工具使用的**必要性**和**可验证性**。

### 3. 为什么需要多轮扩展搜索

一张图的实体（如 6 个品牌）各自的知识图往往只有 1-2 跳深度，不足以构建 L3 链（3跳以上）。

通过迭代地搜索已有三元组的"尾实体"（如搜到"创始人"，再搜"创始人的母校"），知识图不断向外延伸，直到出现满足 L3 要求的路径。

### 4. 为什么加入跨实体关联发现

单独搜索每个实体时，搜索计划天然围绕该实体自身展开，不会去查"这两个品牌有什么关系"。

Times Square 里同时出现的 Toshiba 和 Maxell——分别搜索时，无法发现它们都是日本电子品牌、都曾是 Pioneer 的竞争对手等跨实体事实。跨实体关联发现阶段专门搜索图中实体对，为知识图补充横向连接，让 L3 链有更多路径可选。

---

## 数据难度分级

| 级别 | 当前主要题型 | 模糊化策略 | 工具示例 | 意图 |
|------|-------------|-----------|---------|------|
| **L1** | 纯视觉题 | 不做知识模糊化 | `code_interpreter` / `image_search` | OCR、计数、局部识别、反向图搜 |
| **L2** | `lookup` / `compare` / `delta` / `same_target` / `same_group` / `extremum` / `threshold` | 图中实体名 → 位置描述；中间值节点默认隐藏 | `code_interpreter` → `web_search` → `code_interpreter` | 单实体知识、比较、差值、共享目标判断、集合极值、阈值判断 |
| **L3** | `multihop` / `select_then_follow` / `join_then_follow` | 图中实体 → 位置描述；winner/shared target 等关键节点默认 hard hidden | `code_interpreter` → `web_search` → `visit / code_interpreter` | 真正路径依赖的多步题：先选后查、先汇合后查、两跳链 |

L1 仍额外生成 4 道纯视觉题；L2/L3 则由图谱结构和题目程序自动决定，不再是单纯“找一条链就出题”。

---

## 当前 Step2 的关键变化

现在的 Step2 不再只输出最小三元组，而是尽量让图谱“可计算”。

每个三元组至少包含：
- `head`
- `relation`
- `tail`
- `tail_type`
- `fact`
- `source_snippet`
- `provenance`

并尽量补充：
- `normalized_value`：可比较/可计算的标准化值，主要用于 `TIME` / `QUANTITY`
- `unit`：如 `year`、`$`、`people`

当前 `provenance` 主要有：
- `text_exact`
- `text_rewrite`
- `image_resolved`
- `cross_entity`

Step2 的检索策略也收缩成：
- **exact-name-first**：默认先搜实体精确名和少量 type alias
- **rewrite second**：只有 recall / disambiguation 不足时再让 LLM 改写搜索词
- **image gate only**：`image_search` 不进入主扩展环，只作为高歧义实体的 resolution gate

`relation_family` 已经从 pipeline 中移除。当前 Step3 的主分桶完全依赖：
- `tail_type`
- 原始 `relation` 的轻量自然性过滤
- `normalized_value + unit` 的计算能力

这使得 Step3 可以优先用结构化字段做：
- 时间比较/时间差
- 数值比较/数值差
- 数值求和后与第三个实体做阈值判断

如果旧数据没有这些字段，代码会回退到字符串解析，因此旧数据仍可兼容。

---

## 当前 Step3 的核心思路

现在的 Step3 主流程已经切成：

1. **建图**：把 `triples` 转成带类型与证据的 `DiGraph`
2. **封装 GraphEnv**：只暴露 typed action 级别的访问接口，而不是裸图遍历
3. **给边补 `RelationProfile`**：先判断 relation 值不值得问、能不能自然说成中文
4. **Beam Search 提议 QuestionProgram**：在图上搜索最小证明子图
5. **ProgramVerifier 硬筛**：过滤空 proof、无视觉锚点、结构退化、冗余步
6. **Editor 层**：visibility variant search + utility ranker + microplanning
7. **Draft / Corrector / Round-trip Verify**：先出多个问句草稿，再局部修句，再回查是否语义漂移
8. **全局选题**：跨 family 做实体/锚点/答案去重与覆盖控制

当前核心对象是 `QuestionProgram`，而不是 motif。每个 program 至少包含：
- `family`
- `reasoning_family`
- `goal`
- `difficulty`
- `operations`
- `proof_graph`
- `anchors`
- `answer_node / answer_value`
- `visibility_plan`
- `surface_plan`
- `semantic_intent`
- `realization_schema`
- `semantic_pivot`
- `tool_requirements`
- `tool_irreducibility_score`
- `tool_plan`
- `scores`

当前 family 主要有：
- `lookup`
- `compare`
- `delta`
- `same_target`
- `same_group`
- `extremum`
- `threshold`
- `multihop`
- `select_then_follow`
- `join_then_follow`

其中真正的路径依赖题主要是：
- `select_then_follow`：先比较/筛选，再追后续属性
- `join_then_follow`：先汇合到共享隐藏节点，再继续追下一步

---

## 当前质量控制

除了 program 内部的证据评分，当前版本还做了四类质量控制：

1. **Beam Search 的启发式搜索**
- seed state 不再按组合顺序硬截断，而是按 `single / pair / triple` 分层保底
- pair/triple 会按潜在信息量排序，优先保留更可能长出高价值 program 的状态

2. **ProgramVerifier 硬约束**
- proof graph 非空
- 至少一个图中实体 anchor
- `L3` 的操作复杂度达标
- `select_then_follow / join_then_follow / multihop` 的结构必须完整
- 对明显冗余的 select/join/group 路径做近似拒绝

3. **Editor 层 utility ranker**
- 不再只按结构有效性打分，还额外计算：
  - `answer_uniqueness`
  - `clue_minimality`
  - `reasoning_compactness`
  - `tool_irreducibility`
  - `lexicalizability`
  - `anchor_referability`
  - `question_intent_specificity`
  - `recoverable_abstraction`
  - `ontologyese_penalty`
- 对同答案 / 同 anchor cluster 的候选再做 pairwise rerank，避免“结构对但味道差”的题排在前面

4. **联合选择器**
- 不只限制 family 数量
- 还限制同一图中实体、同一实体对、同一视觉锚点、同一答案被反复问
- `L2` / `L3` 联合纳入，不再各选各的

这一步的目标不是“尽量多出题”，而是“让最终题单更像一份平衡的训练样本”。

---

## 当前 Editor 层

Step3 现在的主要提升不在 planner，而在 editor。当前 editor 层由 5 部分组成：

1. **RelationProfile**
- 每条可问边都会被编译成 relation affordance，至少包含：
  - `wh_type`
  - `question_head_zh`
  - `role_phrase_zh`
  - `askability`
  - `lexicalizability`
  - `hideability`
  - `generic_penalty`
  - `tool_affordance`
- 作用是把“结构可行的边”收缩成“结构可行且适合问的边”

2. **visibility variant search**
- `visibility_plan` 不再只有一版固定输出
- 对 `select_then_follow / join_then_follow` 等路径题，会生成多个 hard/soft hidden 变体
- 再按 `recoverable_abstraction` 选择最适合被自然角色化的隐藏方案

3. **microplanning**
- `surface_plan` 不再只是扁平 slot
- 现在会进一步编译成 `microplan`，至少包含：
  - `viewpoint`
  - `head_noun`
  - `selection_form`
  - `refer_style`
  - `wh_form`

4. **draft / corrector**
- 第一个 LLM 只做 constrained paraphrase，生成多个候选问句
- 第二个 LLM 只做局部修句，不允许改语义条件

5. **round-trip verify**
- realization 后再次检查：
  - 是否泄露 hard hidden
  - family / semantic_intent / wh_form 是否保持一致
  - 是否退化成 ontology 腔

当前明确 ban 的 ontology 腔包括：
- `对应的`
- `所对应的`
- `实体`
- `对象`
- `相关信息`
- `组织或机构是什么`

---

## 一个完整例子

**输入图片**：纽约时代广场街景，图中可见麦当劳招牌（画面左下角）

**Pipeline 处理**：
1. VLM 识别出实体"McDonald's"，bbox 在画面左下角，`_bbox_to_location` 输出"画面下方最左侧"
2. 搜索"McDonald's history founding" → 提取三元组：`(McDonald's, founded_by, Dick and Mac McDonald)`
3. 搜索"Dick and Mac McDonald" → 提取：`(Dick and Mac McDonald, born_in, San Bernardino)`
4. 搜索"San Bernardino" → 提取：`(San Bernardino, located_in, California)`
5. 形成 L3 链：McDonald's → Dick and Mac McDonald → San Bernardino → California

**生成的 L3 问题**：
> 图中**画面下方最左侧**有一家全球知名餐饮品牌的标志。该品牌的创始人兄弟出生于美国哪个州？

**工具序列**：
1. `code_interpreter`：裁剪画面左下角区域，识别品牌名
2. `web_search`：搜索该品牌创始人信息
3. `web_search`：搜索创始人出生地所在州
4. `visit`：访问详情页确认州名

**答案**：California

---

## 当前状态与局限

- **主流程已经是 GraphEnv 版**：不再依赖旧 motif 主路径
- **Editor 层已经接上，但还在继续打磨**：当前已加入 `RelationProfile / utility ranker / visibility variant / microplan / draft-corrector / round-trip verify`
- **题面比上一轮更自然，但还没彻底追平最佳旧版样本**：现在已经能稳定避免 `对应的组织或机构是什么` 这类差表达，但部分锚点指代仍偏机械
- **纯视觉关系仍未利用**：实体之间的视觉空间关系（遮挡、并排、层次）目前未进入 GraphEnv
- **动态知识依赖搜索覆盖**：新兴实体或小众领域可能检索不足
- **低信息密度图片仍不适合**：风景、抽象画等会在 step1/step2 早期被过滤

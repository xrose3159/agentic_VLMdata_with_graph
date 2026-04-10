# 弱约束随机游走 + 后验归纳题型

## 核心思路

不预设题型，不预设轨迹骨架。题型是游走的**输出**，不是输入。

1. 从 step2 输出构建**异构求解图**（region / entity / fact 三层）
2. 从图中某个视觉锚点出发，在证据图上**自由扩展子图**（不是线性链）
3. 每一步用**连续打分函数**偏置方向，难度由偏好向量控制
4. 走到"已经能问出自然问题"时停下
5. **后验归纳题型**：从子图枚举所有可闭合意图（lookup / compare / rank / set_merge），选最佳
6. 编译成 QuestionFrame，单次 LLM 调用语言化

## 与之前方案的区别

| | 旧版线性随机游走 | 主流程 GraphEnv | 当前方案 |
|---|---|---|---|
| 题型来源 | 走完链再硬凑 | 先选 family 再填槽 | **后验归纳，自然长出** |
| 图结构 | 同构知识图 | 同构知识图 | **异构图（region→entity→fact）** |
| 难度控制 | hop 数 / length bucket | family 复杂度 | **偏好向量 + budget deficit** |
| 多分支 | bridge 边硬模拟 | 固定 pair 枚举 | **spawn_new_anchor 自然分叉** |
| 停止条件 | gate 概率 | beam 深度 | **STOP 作为 action，由 closure 质量驱动** |
| L1/L2/L3 | L1 走旁路 | L1 走旁路 | **统一在同一引擎** |

## 架构

```
HeteroSolveGraph          异构求解图（4 类节点 × 3 类边）
    ↓
SubgraphWalker            弱约束自由游走
  ├── DifficultyProfile   难度偏好向量（easy / hard）
  ├── Move scoring        连续打分函数
  └── STOP as action      closure 质量驱动停止
    ↓
ClosureCompiler           后验枚举可闭合意图
  ├── read (L1)
  ├── lookup (L2)
  ├── compare (L3)
  ├── compare_then_follow (L3)
  ├── rank (L3)
  └── set_merge (L3)
    ↓
IrreducibilityChecker     5 个不可约性检查
    ↓
QuestionFrame + Realize   结构化语言化（单次 LLM 调用）
    ↓
ToolPlanCompiler          确定性工具序列编译
```

### 1. 异构求解图（HeteroSolveGraph）

从 step2 entity JSON 构建。

**节点 4 类：**

| 类型 | 含义 | 来源 |
|------|------|------|
| `full_image` | 整图（1 个） | — |
| `region` | 图中实体裁剪区 | step2 entities（bbox / location / type / crop_path） |
| `entity` | 实体 canonical name | step2 entities |
| `fact` | 知识事实值 | step2 triples 的 tail |

**边 3 类：**

| 类型 | 含义 | 工具映射 |
|------|------|---------|
| `observe` | full_image → region | code_interpreter（裁剪） |
| `resolve` | region → entity | code_interpreter（OCR）/ image_search（反向搜索） |
| `retrieve` | entity → fact | web_search / visit |

### 2. 游走状态与动作

**状态**只保留四样东西：

```python
state = {
    subgraph,       # 逐步长大的证据子图（DAG，不是链）
    frontier,       # 可继续扩展的节点
    used_anchors,   # 已使用的视觉锚点
    steps_taken,    # 步数
}
```

**三种动作**（包括 STOP）：

| 动作 | 含义 |
|------|------|
| `expand` | 从 frontier 节点展开一条 resolve/retrieve 边 |
| `spawn_anchor` | 引入新的图中视觉锚点（开新分支） |
| `stop` | 停止游走 |

所有动作放在一起用 softmax 采样，没有单独的 gate。

### 3. 难度偏好向量

不同难度只是权重不同：

```python
score(move) = w1 × visual_novelty      # 引入新锚点
            + w2 × fact_gain            # 引入新可问事实
            + w3 × compute_affordance   # 形成可比较/排序的值
            + w4 × branch_novelty       # 形成新分支
            + w5 × closure_gain         # 离"可问"更近
            - w6 × shortcut_penalty     # 答案太容易直达
            - w7 × redundancy           # 凑数步骤
            - w8 × generic_penalty      # 答案太泛
```

| 偏好 | easy | hard |
|------|------|------|
| visual_novelty | 低 | **高** |
| compute_affordance | 0 | **高** |
| branch_novelty | 0 | **高** |
| closure_gain | **高** | 中 |
| min_anchors | 1 | **2** |
| require_compute | false | **true** |

easy 倾向于快速闭合单分支题，hard 倾向于多锚点 + 多分支 + Python 比较。

### 4. 后验归纳题型（ClosureCompiler）

游走过程中和停止时，枚举当前子图中所有可闭合的问题意图：

| 闭合类型 | 条件 | Level | 典型题 |
|----------|------|-------|--------|
| `read` | region 实体可直接识别 | L1 | "这个广告牌写的什么？" |
| `lookup` | region → entity → fact 链路完整 | L2 | "这个品牌的总部在哪？" |
| `compare` | 两个 entity 有同类 TIME/QUANTITY fact | L3 | "哪个首演更晚？" |
| `compare_then_follow` | compare + 赢家还有别的 fact | L3 | "更早的那个，作曲者是谁？" |
| `rank` | 3+ 个 entity 有同类可排序 fact | L3 | "三个品牌哪个成立最早？" |
| `set_merge` | 两个 entity 共享同一 fact target | L3 | "两个品牌共同上市的交易所？" |

STOP 的得分由最佳 closure 的质量和 budget 满足度决定——有好题可问时倾向停，budget 没满时倾向继续。

### 5. 不可约性检查

5 个二值检查，替代复杂打分：

1. **answer_uniqueness** — 答案非 yes/no/unknown/太泛
2. **realizable_question** — 锚点有视觉描述
3. **no_python_shortcut** — L3 compare 的两个分支 follow 值不同（去掉 compare 会改变答案）
4. **answer_not_visible** — 答案不是图中直接可见的实体名
5. **no_branch_shortcut** — 删掉任一分支后答案必须变

### 6. QuestionFrame 语言化

不把原始子图丢给 LLM，先编译成结构化 frame：

```python
QuestionFrame(
    level=3,
    family="compare_then_follow",
    wh_type="who",
    visible_refs=["画面偏左的音乐剧海报", "画面下方偏左的音乐剧海报"],
    criterion="首演时间更早",
    follow_relation="作曲者",
    hidden_entities=["Billy Elliot", "Mamma Mia!"],
    hidden_values=["November 13, 2008", "October 18, 2001"],
    answer="Elton John",
    answer_type="PERSON",
)
```

LLM 只负责把 frame 说成自然中文，单次调用。

### 7. 工具序列（确定性编译）

从轨迹确定性生成，不让 LLM 规划：

| 轨迹步骤 | Tool | Action |
|----------|------|--------|
| observe(region) | code_interpreter | 裁剪 bbox |
| resolve(region) text/brand | code_interpreter | OCR |
| resolve(region) person/landmark | image_search | 反向搜索 |
| retrieve(entity, relation) | web_search | 搜索实体属性 |
| deep_retrieve | visit | 读网页 |
| compute | code_interpreter | Python 比较/排序/集合运算 |

## 文件

| 文件 | 说明 |
|------|------|
| `trajectory_runtime.py` | 主实现（~1400 行） |
| `run_trajectory.py` | CLI runner |
| `runtime.py` | 旧版线性游走（已弃用） |

## 使用

不带 LLM（只看候选）：

```bash
python -m experimental.random_walk_step3.run_trajectory output/entities/img_0010.json
```

带 LLM 语言化：

```bash
python -m experimental.random_walk_step3.run_trajectory output/entities/img_0010.json \
  --image images/img_0010.jpg --realize
```

## 评测

`eval_agent.py` 实现了真实工具调用的 agent loop 评测：

- 模型输出工具调用 JSON → 真实执行（PIL 裁剪 / easyocr OCR / Serper 搜索 / Jina 读网页）→ 结果喂回 → 循环
- 裁剪后的图片会作为图片消息回传给模型
- 结果保存到 `output/eval_*.json`，含完整 trace

```bash
python eval_agent.py
```

## 效果示例

img_0010（时代广场）生成 8 题：

| Level | Family | 问题 | 答案 |
|-------|--------|------|------|
| L1 | read | 画面最左侧、麦当劳标志上方那个红白相间的电子品牌广告牌上写着什么单词？ | Maxell |
| L1 | read | 画面最右侧的大型彩色海报上，那个带有醒目数字的电影名称是什么？ | Toy Story 2 |
| L2 | lookup | 画面左下角那个带有巨大黄色发光拱门标志的餐饮品牌，它的创始人是谁？ | Ray Kroc |
| L2 | lookup | 画面左侧那张写着"WINNER BEST MUSICAL"的音乐剧海报，它的作曲者是谁？ | Elton John |
| L2 | lookup | 画面正中央远处那栋细长高楼上，红色背景的电子品牌广告牌，所属公司的全称是什么？ | Tokyo Shibaura Electric Company |
| L3 | compare | 两张音乐剧海报，哪一个在百老汇首演的时间更晚？ | Billy Elliot (Nov 2008) |

qwen3-vl-30b-a3b-instruct 真实工具调用评测：**5/6 正确**。

# Stochastic Step3 Experiment

这个目录是对当前 `GraphEnv + Beam Search` proposer 的独立实验版。

目标不是替换现有主流程，而是单独验证一种新的搜索策略：

- `think(state)`：根据当前状态列出合法 typed actions，并给每个 action 一个概率。
- `action`：在高分 action 中保留一部分确定性 top 选择，再按概率采样一部分 action。
- `new_state`：执行 action 后进入新的 `SearchState`，或直接产出 `QuestionProgram`。

## 设计原则

- 保留现有主流程的稳定部分：
  - `GraphEnv`
  - `QuestionProgram`
  - `ProgramVerifier`
  - editor 层（utility / visibility / microplanning / realization）
- 只替换 proposer，不动当前主入口。
- 不是纯随机游走，而是：
  - typed action 约束
  - state-conditioned probability
  - stochastic beam

## 文件

- `runtime.py`
  - 实验版 proposer：`StochasticPolicyProposer`
  - 对外入口：`build_question_programs_stochastic(...)`
- `run_compare.py`
  - 对比 deterministic proposer 和 stochastic proposer

## 使用

```bash
python -m experimental.stochastic_step3.run_compare output/entities/img_0010.json
```

可调参数：

```bash
python -m experimental.stochastic_step3.run_compare \
  output/entities/img_0010.json \
  --seed 13 \
  --beam-size 24 \
  --max-depth 3 \
  --top-p 0.9 \
  --temperature 0.85
```

## 当前状态

- 这是实验分支，不接入 `step3_generate.py` 主流程。
- 目的是观察：
  - 程序候选是否更有多样性
  - `select_then_follow` / `join_then_follow` 是否不再总是长成同一种题
  - editor 层在更“活”的 proposer 上能不能挑出更像人问的问题

# Deep Learning Agent Workflow

本文档定义本项目深度学习部分的本地自迭代 agent 工作流。它借鉴 AutoResearch 的思路：固定目标、固定评估、固定预算、限制可改文件、完整记录实验，并且只有在指标明确变好时才保留修改。

本规范是安全边界，不是建议。agent 执行实验时必须遵守这里的规则；任何放宽权限、扩大可改范围、改变评估标准的行为都需要人工明确批准。

## 1. Goal

agent 的目标是在当前本地分支内迭代改进 `src/deep_learning/` 的 Tiny CNN 语音数字识别方案。

固定任务：

- 训练集：`speaker1` 到 `speaker5`
- 验证集：`speaker6` 和 `speaker7`
- 标签顺序：每个 speaker 文件依次为 `0,1,2,3,4,5,6,7,8,9`
- clean 评估：验证集准确率和 10x10 混淆矩阵
- noise 评估：mixed noise 下 SNR `-10,-5,0,5,10,15,20` dB 的准确率曲线

非目标：

- 不追求使用 Whisper、AST、BC-ResNet 等大模型替代当前主线。
- 不改传统方法和机器学习同学的代码。
- 不自动生成最终报告、slide 或合并 PR。

## 2. Operating Mode

agent 只允许在本地自迭代。默认权限如下：

- 可以：在当前分支内小范围修改允许文件，运行训练和评估，记录实验结果，本地 commit 可保留的实验。
- 不可以：自动 push、自动开 PR、自动 merge、自动改 GitHub issue/PR 状态。
- 不可以：自动安装依赖、联网下载模型、删除用户文件、修改原始数据。

GitHub 插件只用于只读查询仓库、PR 或 issue 状态，或准备人工审阅摘要。任何 GitHub 写操作必须由用户另行明确授权。

## 3. Current Branch And Workspace Gate

启动前必须通过以下门禁。

```bash
git status --short --branch
```

规则：

- agent 可以在当前本地分支运行，不要求创建 `agent/dl-autoresearch-*` 分支。
- agent 启动时必须记录当前分支名，并在每轮结果日志中写入该分支名。
- 如果 `git status --short` 显示未提交改动，agent 停止并列出文件。
- agent 不得自动 `stash`、`reset`、`checkout --` 或删除这些改动。
- agent 不自动切换分支，不自动创建分支。
- 如果当前分支是 `main` 或 `master`，agent 仍可执行只读门禁检查，但不得开始实验性代码修改，除非用户明确要求允许在该分支实验。

## 4. Dependency Gate

启动前必须验证依赖可用。

```bash
python -B -c "import torch, numpy, scipy, matplotlib, librosa, soundfile; print('deps ok')"
python -B -c "import src.deep_learning.train, src.deep_learning.evaluate, src.deep_learning.noise_eval; print('pipeline imports ok')"
python -m src.deep_learning.train --help
```

规则：

- 任一命令失败，agent 停止。
- agent 不自动运行 `pip install`。
- agent 不新增依赖，不修改 `requirements.txt`。
- 如果缺少依赖，agent 只报告缺失模块和建议命令，由人工决定是否安装。

## 5. Allowed Files

agent 单轮实验只允许修改以下文件：

- `src/deep_learning/model.py`
- `src/deep_learning/train.py`
- `src/deep_learning/dataset.py`
- `src/deep_learning/features.py`
- `src/deep_learning/config.py`

只读文件：

- `src/deep_learning/evaluate.py`
- `src/deep_learning/noise_eval.py`
- `src/deep_learning/metrics.py`
- `src/deep_learning/segment.py`
- `docs/agent_workflow.md`
- `docs/deep_learning.md`

禁止修改：

- `reference/`
- `data/raw/`
- `requirements.txt`
- `.gitignore`
- `README.md`
- `src/audio_io.py`
- `src/features.py`
- `src/train.py`
- `src/evaluate.py`
- `src/noise_test.py`
- 任意报告、slide、原始数据或 Git 配置文件

安全检查命令：

```bash
git diff --name-only <base_commit> HEAD
```

输出必须全部属于允许文件列表。否则该轮实验判定为越界，必须 discard。

## 6. Experiment Budget

每一轮实验必须小而可解释。

硬限制：

- 每轮最多修改 3 个文件。
- 单轮 diff 建议不超过 200 行。
- 模型参数量不得超过 150,000。
- 每轮优先只改变一个研究因素。
- 每轮最多训练一次主模型，除非该轮明确是随机种子稳定性检查。

允许研究因素：

- CNN 通道数、卷积层数、dropout、池化方式。
- log-Mel 参数，例如 `n_mels`、`max_seconds`、window/hop。
- augmentation 参数，例如随机增益、时间平移、训练噪声概率。
- optimizer 超参数，例如 learning rate、weight decay、patience。

禁止研究因素：

- 改验证集 speaker。
- 改 digit label 顺序。
- 改 SNR 列表。
- 改 clean/noise 评估脚本以提高分数。
- 引入预训练模型或联网下载权重。

## 7. Single-Round Loop

每轮实验必须按以下顺序执行。

1. 记录基线 commit。

```bash
git rev-parse HEAD
```

记为 `base_commit`。

2. 写出实验假设。

假设必须包含：

- `run_id`
- 修改点
- 预期影响
- 风险
- 回滚标准

3. 修改允许文件。

修改必须与假设一致，不能顺手重构无关代码。

4. 检查参数量。

```bash
python -B -c "from src.deep_learning.model import TinyKeywordCNN, count_parameters; print(count_parameters(TinyKeywordCNN()))"
```

如果参数量超过 150,000，停止并 discard。

5. 本地 commit。

```bash
git add <allowed_files>
git commit -m "agent-exp: <run_id> <idea>"
```

6. 运行固定训练和评估。

```bash
python -m src.deep_learning.train --output-dir outputs/agent_research/runs/<run_id> --checkpoint outputs/agent_research/runs/<run_id>/tiny_cnn.pt --epochs 120 --patience 20 --seed <seed>
python -m src.deep_learning.evaluate --checkpoint outputs/agent_research/runs/<run_id>/tiny_cnn.pt --output-dir outputs/agent_research/runs/<run_id>
python -m src.deep_learning.noise_eval --checkpoint outputs/agent_research/runs/<run_id>/tiny_cnn.pt --output-dir outputs/agent_research/runs/<run_id> --noise-kinds mixed --snr -10 -5 0 5 10 15 20
```

7. 计算分数并记录结果。

结果写入未跟踪文件：

```text
outputs/agent_research/results.tsv
```

8. 决定保留或丢弃。

- 通过保留规则：保留 commit。
- 未通过保留规则：优先用 `git revert <experiment_commit>` 撤销本轮实验；不得默认 `reset --hard`。如果需要 reset，必须先向人工说明目标 commit 和影响范围。

## 8. Scoring

主得分固定为：

```text
score = 0.5 * clean_accuracy + 0.5 * mean_mixed_noise_accuracy
```

其中：

- `clean_accuracy` 来自 `clean_metrics.json`
- `mean_mixed_noise_accuracy` 是 `noise_accuracy.csv` 中 mixed noise 七个 SNR 准确率的平均值
- 分数范围为 `0.0` 到 `1.0`

保留规则：

- 新 `score` 比当前最佳高至少 `0.02`，且 clean accuracy 没有下降超过 `0.05`，才自动保留。
- 如果 `score` 持平，只有在参数更少、代码更简单或低 SNR 表现更稳定时才保留。
- 如果训练失败、评估失败、输出缺失、参数量越界或修改文件越界，一律 discard。

当前最佳分数必须来自同一评估命令和同一数据划分，不能混用不同设置。

## 9. Result Log Schema

`outputs/agent_research/results.tsv` 使用 tab 分隔，必须包含以下列：

```text
run_id	commit	base_commit	seed	status	score	clean_accuracy	mean_mixed_noise_accuracy	param_count	files_changed	idea	notes
```

`status` 只允许以下值：

- `kept`
- `discarded_no_improvement`
- `discarded_failed_eval`
- `discarded_safety_violation`
- `discarded_over_budget`

每轮实验还应在对应 run 目录保存：

- `train_summary.json`
- `training_history.csv`
- `clean_metrics.json`
- `clean_confusion_matrix.csv`
- `clean_confusion_matrix.png`
- `noise_accuracy.csv`
- `accuracy_snr_curve.png`

## 10. Self-Update Policy

agent 可以维护未跟踪状态文件：

```text
outputs/agent_research/strategy_state.md
```

允许写入内容：

- 已尝试实验和结果摘要。
- 失败假设。
- 有效假设。
- 下一轮候选实验。
- 对数据或分段质量的观察。

禁止写入内容：

- 放宽本规范的建议作为已生效规则。
- 自动改变评分公式。
- 自动扩大可改文件列表。
- 自动改变 GitHub 权限。

每 5 轮实验生成一次：

```text
outputs/agent_research/strategy_review.md
```

该文件只提出建议。是否把建议写入正式规范，必须由人工决定。

## 11. Failure Handling

遇到以下情况必须停止循环并报告：

- 依赖缺失。
- 工作区不干净。
- 真实 speaker 音频无法解码。
- 自动分段无法为每个 speaker 产生 10 个片段。
- 训练或评估命令连续两轮失败。
- 评估结果与预期 schema 不一致。
- 发现修改范围越界。

agent 不得通过修改评估脚本、跳过测试、删除结果文件来绕过失败。

## 12. Git And GitHub Boundaries

允许的本地 Git 操作：

- `git status`
- `git diff`
- `git rev-parse`
- `git log`
- `git add` 允许文件
- `git commit`
- 实验失败后，用 `git revert <experiment_commit>` 撤销本轮 commit

禁止的默认操作：

- `git push`
- `git pull --rebase`
- `git merge`
- `git checkout -- <file>`
- `git reset --hard`，除非用户明确批准且目标是本轮 `base_commit`
- 删除文件或清理目录

GitHub 插件默认只读：

- 可读取仓库、PR、issue、review 状态。
- 可准备 PR 摘要或实验报告草稿。
- 不可自动 comment、label、request review、close issue、open PR、merge PR。

任何 GitHub 写操作都必须由用户单独明确授权，并且要先说明目标仓库、分支、PR 或 issue 编号。

## 13. Human Review Checklist

人工审查一个保留实验时，至少检查：

- 是否只改了允许文件。
- 参数量是否小于 150,000。
- 评估命令是否使用固定 train/evaluate/noise_eval 命令。
- clean accuracy 和 noise mean accuracy 是否都合理。
- SNR 曲线是否存在明显异常。
- 代码是否仍然能解释为课程要求内的小模型。
- 是否没有新增依赖、没有修改数据、没有改评估脚本。

建议人工通过后，再由人决定是否把保留 commit cherry-pick 到正式开发分支或开 PR。

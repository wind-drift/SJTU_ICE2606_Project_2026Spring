# SJTU ICE2606 Project 2026 Spring

本仓库用于 SJTU ICE2606 信号与系统课程大作业。

## 1. Project Structure

```text
.
├── README.md
├── .gitignore
├── requirements.txt
├── data/          # 数据文件
├── src/           # 源代码
├── notebooks/     # 实验分析与可视化
├── figures/       # 实验图像
├── reports/       # 报告与展示材料
└── docs/          # 项目文档与笔记
```

## 2. Branch Rules

`main` 分支作为稳定主分支，原则上不直接在 `main` 上开发。

每位成员在自己的分支上工作，其他分支命名建议：

```text
feature/*   # 功能开发
docs/*      # 文档修改
fix/*       # 问题修复
chore/*     # 工程配置、依赖更新、目录整理等
```

## 3. First-Time Setup

第一次创建自己的分支：

```bash
git checkout main
git pull
git checkout -b feature/your-name
git push -u origin feature/your-name
```

## 4. Daily Workflow

每天开始工作时：

```bash
# 1. 更新 main
git checkout main
git pull

# 2. 回到自己的分支
git checkout feature/your-name

# 3. 把 main 的更新同步到自己的分支
git pull origin main

# 4. 开始工作
# ... 修改代码 ...

# 5. 提交并推送自己的分支
git status
git add .
git commit -m "type: brief description"
git push
```

## 5. Pull Request

个人分支完成阶段性工作后，在 GitHub 上创建 Pull Request，将自己的分支合并到 `main`。

建议在一个功能基本完成、代码可以运行后再合并。

## 6. Commit Message

提交信息建议使用：

```text
type: brief description
```

常见类型：

```text
feat: 新增功能
fix: 修复问题
docs: 修改文档
chore: 工程配置或杂项修改
refactor: 代码重构
test: 测试相关修改
```

示例：

```bash
git commit -m "feat: add MFCC feature extraction"
git commit -m "fix: handle audio loading error"
git commit -m "docs: update README"
git commit -m "chore: update requirements"
```

## 7. Files Not Tracked by Git

相关规则已写入 `.gitignore`。如果后续产生新的临时文件或大型中间结果文件，需要及时补充 `.gitignore`。

## 8. Notes

- *README内容由GPT辅助生成*

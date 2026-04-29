# 信号与系统语音识别项目

本仓库用于 SJTU ICE2606 信号与系统课程大作业，任务目标是完成数字语音识别系统的信号分析、特征提取、模型训练、噪声鲁棒性测试与实验报告整理。

## Project Structure

```text
.
├── README.md
├── .gitignore
├── requirements.txt
│
├── data/
│   ├── raw/              # 原始语音数据
│   ├── noise/            # 噪声文件
│   └── processed/        # 预处理后的数据
│
├── src/                  # Python 源代码
│   ├── audio_io.py       # 音频读取与保存
│   ├── features.py       # 特征提取，如 FFT、STFT、MFCC
│   ├── train.py          # 模型训练
│   ├── evaluate.py       # 识别结果评估
│   └── noise_test.py     # 噪声鲁棒性测试
│
├── notebooks/            # 实验分析 notebook
├── figures/              # 实验图像与结果图
├── reports/              # 实验报告
└── docs/                 # 项目说明与笔记
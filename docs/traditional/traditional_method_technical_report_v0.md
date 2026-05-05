# 传统方法语音识别技术报告（阶段版 v0）

> 项目：SJTU ICE2606 Signals and Systems Project 2026 Spring  
> 任务：0~9 孤立数字语音识别  
> 当前版本：传统方法 baseline 阶段性报告  
> 当前负责人：传统方法基础工程闭环

---

## 1. 当前工作概述

本阶段完成了传统信号处理 + 机器学习路线的基础闭环，主要包括：

1. 原始语音数据读取、归一化与数字切分；
2. 单个数字语音的信号分析，包括时域波形、FFT 幅度谱、STFT 语谱图与 MFCC 特征图；
3. 基于 MFCC 统计特征的 KNN baseline 分类器；
4. clean 条件下的验证集识别准确率与混淆矩阵；
5. 基于给定混合噪声的不同 SNR 鲁棒性测试；
6. 重复噪声实验，输出平均准确率与标准差。

当前方法的定位是：建立一条稳定、可解释、可复现的传统方法 baseline，为后续加入动态特征、DTW-KNN、GMM 或深度学习方法提供对比基准。

---

## 2. 数据集与工程结构

### 2.1 原始数据

原始数据包含 7 个说话人音频文件：

```text
speaker1.mp3
speaker2.mp3
speaker3.mp3
speaker4.mp3
speaker5.mp3
speaker6.mp3
speaker7.mp3
```

每个文件中包含数字 0~9 的连续孤立词语音。

按照任务要求，数据划分为：

```text
训练集：speaker1 ~ speaker5，共 5 × 10 = 50 个样本
验证集：speaker6 ~ speaker7，共 2 × 10 = 20 个样本
```

### 2.2 切分后数据结构

预处理后，每个 speaker 文件被切分为 10 个独立数字语音片段，并保存为 wav 文件：

```text
data/processed/
├─train/
│  ├─speaker1_digit0.wav
│  ├─speaker1_digit1.wav
│  └─...
├─val/
│  ├─speaker6_digit0.wav
│  ├─speaker6_digit1.wav
│  └─...
└─metadata.csv
```

其中 `metadata.csv` 记录每个样本的路径、说话人编号、数字标签和训练/验证划分。

---

## 3. 数据预处理方法

### 3.1 音频读取与统一格式

每个原始 mp3 文件经过如下处理：

1. 读取音频；
2. 转为单声道；
3. 重采样到 16000 Hz；
4. 按最大绝对值做幅值归一化；
5. 基于短时能量进行端点检测；
6. 切分出 10 段数字语音；
7. 保存为 wav 文件。

### 3.2 数字切分方法

切分基于短时 RMS 能量。对音频分帧后计算每帧 RMS，使用相对阈值判断是否为有效语音区域。连续的有效帧合并为一个语音片段。

主要参数如下：

```text
采样率：16000 Hz
frame_length：1024 samples
hop_length：256 samples
top_db：25 dB
min_duration：0.25 s
pad_duration：0.08 s
```

其中 `min_duration = 0.25 s` 用于滤除误切出的短噪声片段；`pad_duration = 0.08 s` 用于保留语音前后边缘，避免发音起止被截断。

### 3.3 预处理结果

最终成功得到：

```text
训练样本：50
验证样本：20
总样本数：70
```

人工抽听约 20 个切分样本，未发现明显标签错位或切分错误。

---

## 4. 信号分析

为理解语音信号的时域、频域和时频域特性，选取一个数字样本进行分析，输出如下图像：

```text
results/traditional/signal_analysis/figures/waveform.png
results/traditional/signal_analysis/figures/fft_spectrum.png
results/traditional/signal_analysis/figures/spectrogram.png
results/traditional/signal_analysis/figures/mfcc.png
```

### 4.1 时域波形

时域波形展示语音信号幅值随时间变化的情况。当前样本中可以观察到：

- 语音前后存在较短静音段；
- 主发声区域能量明显增强；
- 主发声段具有一定周期性，说明包含浊音成分；
- 端点检测保留了完整发音，没有明显截断。

时域波形主要用于观察语音起止位置、能量包络和发音强弱变化。

### 4.2 FFT 幅度谱

FFT 幅度谱反映整段语音在频域上的总体能量分布。当前图像中，主要能量集中在低频区域，并在若干频率位置出现峰值。这些峰值与语音的基频、谐波结构及声道共振特性有关。

在最终报告图中，频率范围限制在 0~4000 Hz，以突出主要语音频段。

### 4.3 STFT 语谱图

语音信号是非平稳信号，直接对整段语音做一次 FFT 只能得到整体频谱，无法反映频谱随时间的变化。因此采用短时傅里叶变换（STFT）获得语谱图。

语谱图中：

- 横轴表示时间；
- 纵轴表示频率；
- 颜色表示频率能量强弱。

当前图像显示语音能量主要集中在发声区域，低频部分能量较强，并可以观察到随时间变化的谐波和共振峰结构。

### 4.4 MFCC 特征图

MFCC 用于提取更适合语音识别的特征。其基本流程为：

```text
预加重 → 分帧 → 加窗 → FFT → 功率谱 → Mel 滤波器组 → log 能量 → DCT → 取前 13 维
```

MFCC 图展示不同倒谱系数随时间变化的情况。相比直接频谱，MFCC 更突出语音的谱包络信息，并压缩了冗余频域细节。

---

## 5. MFCC 特征提取原理

### 5.1 预加重

预加重滤波器为：

```text
y[n] = x[n] - 0.97 x[n-1]
```

其作用是增强高频成分，减弱语音低频能量过强带来的影响，使后续频谱分析更加均衡。

### 5.2 分帧与加窗

语音在整体上是非平稳信号，但在短时间内可近似为平稳信号。因此将语音分成短帧处理。

当前参数为：

```text
frame_size = 25 ms
frame_stride = 10 ms
window = Hamming window
```

分帧后对每帧乘 Hamming 窗，以减少硬截断造成的频谱泄漏。

### 5.3 FFT 与功率谱

对每一帧进行 FFT，将时域信号转换为频域表示。随后计算功率谱：

```text
power ∝ |FFT(x)|²
```

功率谱表示每个频率位置上的能量强弱。

### 5.4 Mel 滤波器组

Mel 滤波器组模拟人耳对频率的非线性感知：人耳对低频差异更敏感，对高频差异相对不敏感。因此 Mel 滤波器组在低频区域分布更密，在高频区域分布更疏。

具体实现中，先在 Mel 轴上等间距取点，再转换回 Hz 轴。由于 Mel-Hz 变换是非线性的，转换回 Hz 后会形成低频窄、高频宽的一组三角滤波器。

每个三角滤波器的峰值通常为 1，低频和高频的区别主要体现在宽度和间距，而不是峰值高度。

### 5.5 log 能量

Mel 滤波器组得到的是每个 Mel 频带中的线性能量。随后取对数：

```text
log_filter_banks = log(filter_banks)
```

其作用是模拟人耳对响度的非线性感知，同时压缩能量动态范围。可以理解为：Mel 处理频率轴，log 处理能量轴。

### 5.6 DCT 与前 13 维 MFCC

对 log-Mel 能量进行 DCT，得到倒谱系数。DCT 的作用是压缩信息并降低相邻 Mel 频带之间的相关性。低阶 MFCC 系数主要描述语音谱包络，因此取前 13 维作为每帧特征。

---

## 6. Baseline 分类方法：MFCC-stat + KNN

### 6.1 固定长度统计特征

每段语音长度不同，MFCC 帧数也不同。为了输入 KNN 分类器，需要将变长 MFCC 矩阵转换为固定长度向量。

当前采用：

```text
13 维 MFCC 均值 + 13 维 MFCC 标准差 = 26 维特征向量
```

其中：

- mean 表示整段语音的平均谱包络特征；
- std 表示语音特征随时间变化的幅度。

### 6.2 StandardScaler 标准化

KNN 基于距离进行分类，不同特征维度的数值范围会影响距离计算。因此在 KNN 前使用 StandardScaler：

```text
z = (x - μ) / σ
```

使各维特征具有相近尺度，避免数值范围较大的维度支配欧氏距离。

### 6.3 KNN 分类器

当前使用 1-NN 分类器：

```text
KNeighborsClassifier(n_neighbors=1, metric="euclidean")
```

对于验证样本，计算其与所有训练样本在 26 维特征空间中的欧氏距离，并将最近训练样本的标签作为预测结果。

1-NN 的优点是简单、直观、适合小样本 baseline；缺点是对异常样本和噪声扰动较敏感。

---

## 7. Clean 条件识别结果

### 7.1 实验设置

训练集：speaker1~speaker5，共 50 个样本。  
验证集：speaker6~speaker7，共 20 个样本。  
特征维度：26 维。  
分类器：StandardScaler + 1-NN。

### 7.2 结果

Clean 验证集准确率为：

```text
Accuracy = 0.8000
```

即 20 个验证样本中识别正确 16 个。

### 7.3 混淆矩阵分析

结果文件：

```text
results/traditional/baseline_knn/figures/clean_confusion_matrix.png
```

主要错误包括：

```text
0 → 7
2 → 8
4 → 3
6 → 9
```

这些错误说明：MFCC mean/std 特征能够区分多数数字，但由于该统计特征丢失了时间顺序信息，部分发音在特征空间中仍可能接近其他数字。

---

## 8. 噪声鲁棒性测试

### 8.1 噪声来源

使用教师提供的 MATLAB 脚本生成混合噪声 `mixed_noise.wav`。该噪声由以下成分混合得到：

```text
白噪声
粉红噪声 / 1/f 噪声
50 Hz 工频干扰
```

噪声采样率为 16000 Hz，时长 10 s，并进行了 RMS 归一化。

### 8.2 SNR 加噪方法

对验证集语音添加指定 SNR 的噪声。SNR 定义为：

```text
SNR = 10 log10(P_signal / P_noise)
```

根据目标 SNR 调整噪声缩放系数：

```text
alpha = sqrt(P_signal / (P_noise × 10^(SNR/10)))
noisy = clean + alpha × noise
```

实验测试 SNR 为：

```text
-10 dB, -5 dB, 0 dB, 5 dB, 10 dB, 15 dB, 20 dB
```

### 8.3 Repeated noise test

为降低单次随机噪声片段带来的偶然性，采用重复测试：

- 每个 SNR 重复 20 次；
- 所有重复均使用同一个 `mixed_noise.wav` 文件；
- 每次重复中，为每个验证样本截取固定噪声片段；
- 在同一次重复内，不同 SNR 使用同一批噪声片段，仅改变缩放系数；
- 最终统计每个 SNR 下的平均准确率与标准差。

这样可以保证不同 SNR 条件之间主要比较的是噪声强度变化，而不是噪声片段随机差异。

### 8.4 结果

Clean reference accuracy：

```text
0.8000
```

噪声鲁棒性测试结果如下：

| SNR (dB) | Accuracy Mean | Accuracy Std |
|---:|---:|---:|
| -10 | 0.1575 | 0.0373 |
| -5  | 0.2425 | 0.0494 |
| 0   | 0.3500 | 0.0229 |
| 5   | 0.2825 | 0.0335 |
| 10  | 0.2950 | 0.0276 |
| 15  | 0.3675 | 0.0335 |
| 20  | 0.5025 | 0.0302 |

结果文件：

```text
results/traditional/noise_robustness_v2/accuracy_snr_curve_mean_std.png
results/traditional/noise_robustness_v2/snr_accuracy_summary.csv
results/traditional/noise_robustness_v2/noise_test_repeated_metrics.txt
```

### 8.5 结果分析

从结果可以看出，baseline 在 clean 条件下准确率为 80%，但加入混合噪声后准确率明显下降。即使在 20 dB 条件下，平均准确率也只有约 50%。

这说明当前 MFCC mean/std + 1-NN baseline 对噪声较敏感。可能原因包括：

1. 统计特征对噪声较敏感；
2. 切分样本前后存在 padding，clean 条件下接近静音，但加噪后这些区域会变成纯噪声；
3. 当前 mean/std 统计将所有帧纳入计算，静音段噪声会污染整体特征；
4. KNN 基于欧氏距离，噪声导致特征偏移后容易改变最近邻关系；
5. 验证集样本数较少，每个样本对准确率影响较大。

整体趋势上，高 SNR 条件下准确率有所恢复，但曲线存在局部波动。这与小样本验证集和 KNN 对扰动敏感有关。

---

## 9. 当前局限性

当前 baseline 存在以下局限：

1. MFCC 统计特征丢失时间顺序信息；
2. 没有使用 Δ / Δ² 动态特征；
3. 没有进行有效语音帧筛选，静音段噪声可能影响统计结果；
4. KNN 对噪声和异常样本敏感；
5. 验证集规模较小，单个样本对准确率影响较大；
6. 尚未与 SVM、DTW-KNN、GMM 或深度学习方法进行系统对比。

---

## 10. 下一步计划

### 10.1 Baseline+ 改进

优先考虑以下基础改进：

1. 有效语音帧筛选：只对短时能量较高的帧计算 MFCC mean/std；
2. 对比保留 C0 与去除 C0 的效果；
3. 尝试简单滤波预处理，如 50 Hz 陷波或低频高通滤波；
4. 小范围比较 KNN 参数，如 k=1/3/5 和 distance weighting。

其中最推荐首先进行有效语音帧筛选，因为当前噪声退化很可能与静音段噪声污染有关。

### 10.2 Advanced 方法接口

当前 baseline 已经形成完整闭环，后续可以与 advanced 方法对比。Advanced 方法可包括：

```text
MFCC + Δ + Δ²
CMVN
DTW-KNN
GMM
模板匹配
SVM baseline
```

对比指标包括：

```text
clean accuracy
clean confusion matrix
不同 SNR 下 mean accuracy ± std
最低可接受 SNR
错误样本分析
```

---

## 11. 当前阶段结论

本阶段完成了传统语音识别方法的基础工程闭环。通过 MFCC mean/std 特征与 1-NN 分类器，在 clean 验证集上达到 80% 准确率，说明 MFCC 统计特征能够有效提取数字语音的内容相关信息。

噪声鲁棒性实验表明，当前 baseline 在混合噪声下性能明显下降，20 dB 时平均准确率约为 50%，说明简单统计特征与 KNN 分类器对噪声较敏感。这一结果为后续 baseline+ 改进和 advanced 算法设计提供了明确方向。

当前版本可作为后续算法改进的 baseline 参考。


## project part file

part developer: Jing Xu

2026spring

---

本部分说明：
- 放弃了“均值/方差”压缩，保留了时序信息，将语音视为一个变长特征序列，形状为(n_frames, 39)
- 39维动态特征提取 (MFCC + $\Delta$ + $\Delta^2$)，compute_delta函数中实现了一阶差分和二阶差分算法，将原始的13维MFCC扩展到了39维语音特征
- CMVN倒谱均值方差归一化，在获得39维特征后，直接加入了基于句子的均值归零和方差归一化((feat - mean) / std)
- 分类器替换为GMM(高斯混合模型)，代码中为0-9每个数字独立训练了一个拥有4个分量的 GMM
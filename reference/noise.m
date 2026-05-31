%% 生成混合噪声文件（白噪声 + 粉红噪声 + 50Hz工频）
% 用于语音识别任务的统一噪声测试
clear; clc;

% 参数设置
fs = 16000;                 % 采样率（与语音数据一致）
duration = 10;              % 噪声时长（秒）
t = (0:duration*fs-1)' / fs;

%% 1. 生成各噪声分量
% 白噪声
white_noise = randn(length(t), 1);
white_noise = white_noise / rms(white_noise);   % 归一化RMS=1

% 粉红噪声（频域法）
N = length(t);
freq = (0:N-1)' / duration;
freq(1) = 1e-12;
pink_amp = 1 ./ sqrt(freq);
pink_amp(1) = 0;
phase = 2*pi*rand(N,1);
X_pink = pink_amp .* exp(1j*phase);
for k = 2:floor(N/2)
    X_pink(N-k+2) = conj(X_pink(k));
end
if mod(N,2) == 0
    X_pink(N/2+1) = real(X_pink(N/2+1));
end
pink_noise = real(ifft(X_pink));
pink_noise = pink_noise / rms(pink_noise);

% 50Hz工频
power_noise = sin(2*pi*50 * t);

%% 2. 混合（等功率）
mixed_noise = white_noise + pink_noise + power_noise;
% 归一化RMS=1（可选，便于后续缩放）
mixed_noise = mixed_noise / rms(mixed_noise);

%% 3. 保存
audiowrite('mixed_noise.wav', mixed_noise, fs);
fprintf('混合噪声文件已保存为 mixed_noise.wav\n');
disp('该文件包含白噪声、粉红噪声和50Hz工频干扰，RMS已归一化。');
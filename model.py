from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt, resample
import math


FS_ORIG    = 86    # 原始采样率
FS_TARGET  = 25    # 下采样率 (Hz)
WINDOW_SEC = 5
SLIDE_SEC  = 1
EPS        = 1e-6  # 防除零
CUTOFF     = 12

def lowpass_filter(x, cutoff, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff/nyq, btype='low')
    return filtfilt(b, a, x)

def highpass_filter(x, cutoff, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff/nyq, btype='high')
    return filtfilt(b, a, x)

class PPGRegressionDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        split_df: pd.DataFrame,
        fold: int,
        mode: str = 'trainval',   # 'trainval' or 'test'
        fold_mode: str = 'train', # only for trainval: 'train' or 'val'
    ):
        self.X, self.y = [], []

        # 根据 split_df 选 encounter_id
        if mode == 'test':
            encs = split_df[split_df['set']=='test']['encounter_id']
        else:
            tv   = split_df[split_df['set']=='trainval']
            encs = (tv[tv['fold'] != fold]['encounter_id']
                    if fold_mode=='train' else
                    tv[tv['fold'] == fold]['encounter_id'])

        win_len = FS_TARGET * WINDOW_SEC
        slide   = FS_TARGET * SLIDE_SEC

        for enc in encs:
            fn = os.path.join(dataset_dir, f"{enc}.npz")
            if not os.path.exists(fn):
                continue

            data    = np.load(fn)
            red_arr = data['red'][::2]    # (seconds, FS_ORIG)
            ir_arr  = data['ir'][::2]     # (seconds, FS_ORIG)
            spo2    = data['spo2'][::2]   # (seconds, ...)

            # 1) 确保尺寸
            secs, fs0 = red_arr.shape
            assert fs0 == FS_ORIG, f"FS mismatch: {fs0} vs {FS_ORIG}"

            # 2) 降采样前先扁平化
            red_flat = red_arr.reshape(-1)
            ir_flat  = ir_arr.reshape(-1)

            # 3) 低通滤波 
            red_flat = lowpass_filter(red_flat, cutoff=CUTOFF, fs=FS_ORIG)
            ir_flat  = lowpass_filter(ir_flat,  cutoff=CUTOFF, fs=FS_ORIG)

            red_dc = lowpass_filter(red_flat, cutoff=0.5, fs=FS_ORIG)
            ir_dc  = lowpass_filter(ir_flat,  cutoff=0.5, fs=FS_ORIG)

            # 4) 分离 AC && 构造比值特征 AC/DC
            red_ac   = red_flat - red_dc
            ir_ac    = ir_flat  - ir_dc
            red_feat = red_ac  / (red_dc  + EPS)
            ir_feat  = ir_ac   / (ir_dc   + EPS)

            # 5) 下采样到 FS_TARGET
            n_orig = red_feat.shape[0]
            n_tgt  = int(secs * FS_TARGET)
            red_ds = resample(red_feat, n_tgt)
            ir_ds  = resample(ir_feat,  n_tgt)

            # 6) 处理 spo2 → 每秒取平均，再重复到 sample 级，
            #    下采样则直接 repeat FS_TARGET 次
            if spo2.ndim > 1:
                spo2_sec = np.nanmean(spo2, axis=1)
            else:
                spo2_sec = spo2.copy()
            spo2_ds = np.repeat(spo2_sec, FS_TARGET)

            # 7) 三条曲线长度校验
            assert len(red_ds) == len(ir_ds) == len(spo2_ds), (
                f"Length mismatch: {len(red_ds)}, {len(ir_ds)}, {len(spo2_ds)}"
            )
            n = len(red_ds)

            # 8) 滑动窗口
            for start in range(0, n - win_len + 1, slide):
                end   = start + win_len
                seg_r = red_ds[start:end]
                seg_i = ir_ds[start:end]
                seg_o = spo2_ds[start:end]

                # 丢掉含 NaN 的窗口
                if np.isnan(seg_r).any() or np.isnan(seg_i).any() or np.isnan(seg_o).any():
                    continue

                # 合成 (win_len,2) 输入，目标 y∈[0,1]
                seq    = np.stack([seg_r, seg_i], axis=1).astype(np.float32)
                target = float(seg_o.mean() / 100.0)

                self.X.append(seq)
                self.y.append(target)

        # 转成 Tensor
        self.X = torch.from_numpy(np.stack(self.X))  # (N, win_len, 2)
        self.y = torch.tensor(self.y).float()         # (N,)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, return_features=False):
        lstm_out, _ = self.lstm(x)

        context = lstm_out.mean(dim=1)

        if return_features:
            return context  # Return latent features
        return self.fc(context)

class AttentionBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        num_heads=4
        # self.attention = nn.Sequential(
        #     nn.Linear(hidden_dim * 2, hidden_dim),
        #     nn.Tanh(),
        #     nn.Linear(hidden_dim, 1),
        #     nn.Softmax(dim=1)  # Softmax over timesteps
        # )
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim*2,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True   # PyTorch ≥1.11 支持 batch_first
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, return_features=False):
        # LSTM output: (batch_size, seq_len, hidden_dim * 2)
        lstm_out, _ = self.lstm(x)
        # Attention weights: (batch_size, seq_len, 1)

        # attn_weights = self.attention(lstm_out)
        # context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch_size, hidden_dim * 2)

        attn_out, attn_weights = self.self_attn(lstm_out, lstm_out, lstm_out)
        context = attn_out.mean(dim=1)

        if return_features:
            return context  # Return latent features
        return self.fc(context)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Precompute the positional encodings once
        pe = torch.zeros(max_len, d_model)   # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        returns: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        # 1) project raw 2-ch input to d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        # 2) positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # 3) transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,    # so inputs are (B, L, d_model)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4) attention pooling (learned over timesteps)
        self.attn_pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1)    # weights sum to 1 over seq_len
        )

        # 5) final regressor
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        x: (batch, seq_len, 2)
        returns: (batch, 1) or if return_features then (batch, d_model)
        """
        # embed + scale
        x = self.input_proj(x) * math.sqrt(self.input_proj.out_features)  # (B, L, d_model)
        # add positional info
        x = self.input_norm(x) 
        x = self.pos_encoder(x)
        # transformer stack
        x = self.transformer_encoder(x)  # (B, L, d_model)
        # attention pooling
        attn_w = self.attn_pool(x)       # (B, L, 1)
        context = torch.sum(attn_w * x, dim=1)  # (B, d_model)
        if return_features:
            return context
        # regression head
        return self.fc(context)          # (B, 1)

class CNNBaseline(nn.Module):
    def __init__(self, input_dim, conv_channels=[64, 128], kernel_size=3, dropout=0.3):
        """
        input_dim: 每个时间步的特征维度（比如 PPG 通道数）
        conv_channels: 每层 Conv1d 的输出通道数列表
        kernel_size: 卷积核大小
        """
        super().__init__()
        layers = []
        in_ch = input_dim
        for out_ch in conv_channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ]
            in_ch = out_ch
        # 全局池化到长度 1
        layers += [nn.AdaptiveAvgPool1d(1)]
        self.encoder = nn.Sequential(*layers)
        self.fc = nn.Linear(conv_channels[-1], 1)
    
    def forward(self, x, return_features=False):
        # x: (batch, seq_len, input_dim) → 转成 (batch, input_dim, seq_len)
        x = x.transpose(1, 2)
        feat = self.encoder(x).squeeze(-1)    # (batch, conv_channels[-1])
        if return_features:
            return feat
        return self.fc(feat)                 # (batch, 1)
    

class Chomp1d(nn.Module):
    """
    裁剪因果卷积（causal conv）中右侧多余的 padding，使输出长度与输入长度一致。
    """
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # x 的 shape: (batch, channels, length_padded)
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    TCN 的基本模块（Temporal Block）：
    - 两层 1D 因果卷积（带 dilation），每层后面接 ReLU + Dropout
    - 残差连接：若输入通道 != 输出通道，则使用 1x1 卷积对齐
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, dilation, padding, dropout):
        super().__init__()
        # 第一层因果卷积
        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # 第二层因果卷积
        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # 残差分支：如果 in_channels != out_channels，需要 1×1 卷积
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else None
        self.relu = nn.ReLU()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (batch, in_channels, seq_len)
        out = self.conv1(x)      # (batch, out_channels, seq_len + padding_left)
        out = self.chomp1(out)   # 裁剪到 (batch, out_channels, seq_len)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        # 残差连接
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    """
    多层 TCN（Temporal Convolutional Network）：
    - num_channels: 列表，指定每层 TemporalBlock 的输出通道数
    - kernel_size: 卷积核大小
    - dropout: Dropout 概率
    """
    def __init__(self, input_channels, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            in_ch = input_channels if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            dilation_size = 2 ** i
            padding = (kernel_size - 1) * dilation_size
            layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=dropout
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: (batch, input_channels, seq_len)
        :return: (batch, num_channels[-1], seq_len)
        """
        return self.network(x)


class TCNBaseline(nn.Module):
    """
    将 CNNBaseline 中的卷积编码器改为 TCN 编码器：
      - 输入：x (batch, seq_len, input_dim)
      - 输出：标量回归 / 分类 (或 return_features=True 时返回特征向量)
    """
    def __init__(
        self,
        input_dim,
        tcn_channels=[64, 128],
        kernel_size=3,
        dropout=0.3
    ):
        """
        :param input_dim: 每个时间步的特征维度（比如 PPG 通道数）
        :param tcn_channels: 列表，每个元素表示该层 TCN 的 out_channels
        :param kernel_size: TCN 中卷积核大小
        :param dropout: TCN 中的 Dropout 概率
        """
        super().__init__()

        # 1) 用 TCN 取代原先的 Conv1d+BN+ReLU+Dropout 模块
        # 输入到 TCN 的通道数= input_dim，输出通道数序列由 tcn_channels 决定
        self.tcn = TCN(
            input_channels=input_dim,
            num_channels=tcn_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )

        # 2) 全局池化层：AdaptiveAvgPool1d(1) → (batch, tcn_channels[-1], 1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 3) 全连接层：把 (batch, tcn_channels[-1]) 映射到 1 维
        self.fc = nn.Linear(tcn_channels[-1], 1)

    def forward(self, x, return_features=False):
        """
        :param x: (batch, seq_len, input_dim)
        :param return_features: 如果为 True，则在全连接前就返回特征向量 (batch, tcn_channels[-1])
        :return:
            - 如果 return_features=True: 返回 (batch, tcn_channels[-1])
            - 否则：返回 (batch, 1)
        """
        # 先把 (batch, seq_len, input_dim) → (batch, input_dim, seq_len)
        x = x.transpose(1, 2)

        # TCN 提取时序特征: (batch, input_dim, seq_len) → (batch, tcn_channels[-1], seq_len)
        tcn_out = self.tcn(x)

        # 全局池化到长度 1: (batch, tcn_channels[-1], seq_len) → (batch, tcn_channels[-1], 1)
        pooled = self.global_pool(tcn_out)

        # 去掉最后一维: (batch, tcn_channels[-1])
        feat = pooled.squeeze(-1)

        if return_features:
            return feat

        # 用全连接映射到标量: (batch, 1)
        return self.fc(feat)
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt, resample

FS_ORIG    = 86    # 原始采样率
FS_TARGET  = 25    # 下采样率 (Hz)
WINDOW_SEC = 6
SLIDE_SEC  = 1
EPS        = 1e-6  # 防除零
CUTOFF     = 12

def lowpass_filter(x, cutoff, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff/nyq, btype='low')
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
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)  # Softmax over timesteps
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, return_features=False):
        # LSTM output: (batch_size, seq_len, hidden_dim * 2)
        lstm_out, _ = self.lstm(x)
        # Attention weights: (batch_size, seq_len, 1)
        attn_weights = self.attention(lstm_out)
        # Compute context vector as weighted sum of lstm_out
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch_size, hidden_dim * 2)
        if return_features:
            return context  # Return latent features
        return self.fc(context)
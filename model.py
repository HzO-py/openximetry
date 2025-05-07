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
WINDOW_SEC = 10
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

            mean_r, std_r = red_flat.mean(), red_flat.std(ddof=0)
            mean_i, std_i = ir_flat .mean(), ir_flat .std(ddof=0)
            red_flat = (red_flat - mean_r) / (std_r + EPS)
            ir_flat  = (ir_flat  - mean_i) / (std_i + EPS)

            # 3) 低通滤波 
            red_flat = lowpass_filter(red_flat, cutoff=CUTOFF, fs=FS_ORIG)
            ir_flat  = lowpass_filter(ir_flat,  cutoff=CUTOFF, fs=FS_ORIG)

            red_dc = lowpass_filter(red_flat, cutoff=0.5, fs=FS_ORIG)
            ir_dc  = lowpass_filter(ir_flat,  cutoff=0.5, fs=FS_ORIG)
            red_ac = highpass_filter(red_flat, cutoff=0.5, fs=FS_ORIG)
            ir_ac  = highpass_filter(ir_flat,  cutoff=0.5, fs=FS_ORIG)

            n_tgt  = int(secs * FS_TARGET)
            red_ac_ds = resample(red_ac, n_tgt)
            red_dc_ds = resample(red_dc, n_tgt)
            ir_ac_ds  = resample(ir_ac,  n_tgt)
            ir_dc_ds  = resample(ir_dc,  n_tgt)
            if spo2.ndim > 1:
                spo2_sec = np.nanmean(spo2, axis=1)
            else:
                spo2_sec = spo2.copy()
            spo2_ds = np.repeat(spo2_sec, FS_TARGET)

            # 三者长度校验
            assert len(red_ac_ds)==len(red_dc_ds)==len(ir_ac_ds)==len(ir_dc_ds)==len(spo2_ds), (
                f"Length mismatch: {len(red_ac_ds)}, {len(spo2_ds)}"
            )
            n = len(spo2_ds)

            # 滑动窗口
            for start in range(0, n - win_len + 1, slide):
                end = start + win_len
                segs = [
                    red_ac_ds[start:end],
                    red_dc_ds[start:end],
                    ir_ac_ds[start:end],
                    ir_dc_ds[start:end],
                ]
                # 如果含 NaN 就跳过
                if any(np.isnan(s).any() for s in segs) or np.isnan(spo2_ds[start:end]).any():
                    continue

                # stack 成 (win_len, 4)
                seq    = np.stack(segs, axis=1).astype(np.float32)
                target = float(spo2_ds[start:end].mean() / 100.0)

                self.X.append(seq)
                self.y.append(target)

        # 转成 Tensor
        self.X = torch.from_numpy(np.stack(self.X))  # (N, win_len, 4)
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
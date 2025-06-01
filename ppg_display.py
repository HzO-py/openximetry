import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, resample

# ———— 参数 ————
FS_ORIG    = 86
FS_TARGET  = 25
CUTOFF     = 12
EPS        = 1e-6
DATASET_DIR= "dataset"  # 请根据实际修改
TARGET_ENC = "02093f14af09488db9e650db6130e55e214c1831adf4f60a8685ef3f0ef2839d"

# 要绘制的时间段（秒）
segments_sec = [(1010, 1020)]

# 给每条曲线分配颜色
colors = {
    'raw_red'  : 'tab:red',
    'raw_ir'   : 'tab:pink',
    'dc_red'   : 'tab:red',
    'dc_ir'    : 'tab:pink',
    'feat_red' : 'tab:red',
    'feat_ir'  : 'tab:pink',
    'ac_red' : 'tab:red',
    'ac_ir'  : 'tab:pink',
    'spo2'     : 'tab:blue',
}

def lowpass_filter(x, cutoff, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff/nyq, btype='low')
    return filtfilt(b, a, x)

def highpass_filter(x, cutoff, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff/nyq, btype='high')
    return filtfilt(b, a, x)

# 1) 载入原始数据
fn   = os.path.join(DATASET_DIR, f"{TARGET_ENC}.npz")
data = np.load(fn)
raw_red = data['red'][::2].reshape(-1)
raw_ir  = data['ir'][::2].reshape(-1)

# 2) 计算 AC/DC
red_filt = lowpass_filter(raw_red, CUTOFF, FS_ORIG)
ir_filt  = lowpass_filter(raw_ir,  CUTOFF, FS_ORIG)
red_dc   = lowpass_filter(raw_red, 0.5, FS_ORIG)
ir_dc    = lowpass_filter(raw_ir,  0.5, FS_ORIG)
red_feat = (red_filt - red_dc) / (red_dc + EPS)
ir_feat  = (ir_filt  - ir_dc)  / (ir_dc  + EPS)
red_ac    = highpass_filter(red_filt,  0.5, FS_ORIG)
ir_ac    = highpass_filter(ir_filt,  0.5, FS_ORIG)

# 3) 下采样到 FS_TARGET
n_tgt       = int(len(red_feat) * FS_TARGET / FS_ORIG)
raw_red_ds  = resample(raw_red,   n_tgt)
raw_ir_ds   = resample(raw_ir,    n_tgt)
dc_red_ds   = resample(red_dc,    n_tgt)
dc_ir_ds    = resample(ir_dc,     n_tgt)
red_ds      = resample(red_feat,  n_tgt)
ir_ds       = resample(ir_feat,   n_tgt)
red_ac      = resample(red_ac,  n_tgt)
ir_ac       = resample(ir_ac,   n_tgt)

# 4) SpO₂ 数据处理
spo2 = data['spo2'][::2]
if spo2.ndim > 1:
    spo2_sec = np.nanmean(spo2, axis=1)
else:
    spo2_sec = spo2.copy()
spo2_ds = np.repeat(spo2_sec, FS_TARGET)[:n_tgt]

# 5) 构造时间轴并过滤 NaN
t_ds = np.arange(n_tgt) / FS_TARGET
mask = (
    ~np.isnan(raw_red_ds) &
    ~np.isnan(raw_ir_ds)  &
    ~np.isnan(dc_red_ds)   &
    ~np.isnan(dc_ir_ds)    &
    ~np.isnan(red_ds)      &
    ~np.isnan(ir_ds)       &
    ~np.isnan(spo2_ds)     &
    ~np.isnan(red_ac)      &
    ~np.isnan(ir_ac)       
)
t_ds        = t_ds[mask]
raw_red_ds  = raw_red_ds[mask]
raw_ir_ds   = raw_ir_ds[mask]
dc_red_ds   = dc_red_ds[mask]
dc_ir_ds    = dc_ir_ds[mask]
red_ds      = red_ds[mask]
ir_ds       = ir_ds[mask]
spo2_ds     = spo2_ds[mask]
red_ac       = red_ac[mask]
ir_ac     = ir_ac[mask]

# 6) 按时间段画图
for start_sec, end_sec in segments_sec:
    i0 = int(start_sec * FS_TARGET)
    i1 = int(end_sec   * FS_TARGET)

        
    fig, axes = plt.subplots(
        3, 1,
        figsize=(8, 4),
        sharex=True,
        facecolor='white'
    )

    # 隐藏所有边框和刻度
    for ax in axes:
        # 隐藏四条脊线
        for spine in ax.spines.values():
            spine.set_visible(False)
        # 隐掉刻度线和刻度标签
        ax.set_xticks([])
        ax.set_yticks([])

    # PPG_red
    fontsize=16
    axes[0].plot(t_ds[i0:i1], red_ds[i0:i1],
                 color=colors['feat_red'], linewidth=1)
    axes[0].set_ylabel('PPG_red',
                       rotation=0,
                       labelpad=15,
                       va='center',
                       fontsize=fontsize)

    # PPG_IR
    axes[1].plot(t_ds[i0:i1], ir_ds[i0:i1],
                 color=colors['feat_ir'], linewidth=1)
    axes[1].set_ylabel('PPG_IR',
                       rotation=0,
                       labelpad=15,
                       va='center',
                       fontsize=fontsize)

    # SpO₂
    axes[2].plot(t_ds[i0:i1], spo2_ds[i0:i1],
                 color=colors['spo2'], linewidth=2)
    axes[2].set_ylabel('SpO₂ (%)',
                       rotation=0,
                       labelpad=15,
                       va='center',
                       fontsize=fontsize)
    axes[2].set_xlabel('Time', fontsize=fontsize)

    fig.suptitle("Public Dataset (Openoximetry Repository)", fontsize=fontsize)
    # fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('imgs/ppg_display.svg')


# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import butter, filtfilt, resample

# # ———— 参数 ————
# FS_ORIG    = 86
# FS_TARGET  = 25
# CUTOFF     = 12
# EPS        = 1e-6
# DATASET_DIR= "dataset"  # 请根据实际修改
# TARGET_ENC = "02093f14af09488db9e650db6130e55e214c1831adf4f60a8685ef3f0ef2839d"

# # 要绘制的时间段（秒）
# segments_sec = [(1015, 1020)]

# # 给每条曲线分配颜色
# colors = {
#     'raw_red'  : 'tab:red',
#     'raw_ir'   : 'tab:pink',
#     'dc_red'   : 'tab:red',
#     'dc_ir'    : 'tab:pink',
#     'feat_red' : 'tab:red',
#     'feat_ir'  : 'tab:pink',
#     'ac_red' : 'tab:red',
#     'ac_ir'  : 'tab:pink',
#     'spo2'     : 'tab:blue',
# }

# def lowpass_filter(x, cutoff, fs, order=4):
#     nyq = 0.5 * fs
#     b, a = butter(order, cutoff/nyq, btype='low')
#     return filtfilt(b, a, x)

# def highpass_filter(x, cutoff, fs, order=4):
#     nyq = 0.5 * fs
#     b, a = butter(order, cutoff/nyq, btype='high')
#     return filtfilt(b, a, x)

# # 1) 载入原始数据
# fn   = os.path.join(DATASET_DIR, f"{TARGET_ENC}.npz")
# data = np.load(fn)
# raw_red = data['red'][::2].reshape(-1)
# raw_ir  = data['ir'][::2].reshape(-1)

# # 2) 计算 AC/DC
# red_filt = lowpass_filter(raw_red, CUTOFF, FS_ORIG)
# ir_filt  = lowpass_filter(raw_ir,  CUTOFF, FS_ORIG)
# red_dc   = lowpass_filter(raw_red, 0.5, FS_ORIG)
# ir_dc    = lowpass_filter(raw_ir,  0.5, FS_ORIG)
# red_filt = red_filt - red_dc
# ir_filt  = ir_filt  - ir_dc
# red_feat = red_filt/ (red_dc + EPS)
# ir_feat  = ir_filt/ (ir_dc  + EPS)

# # 3) 下采样到 FS_TARGET
# n_tgt       = int(len(red_feat) * FS_TARGET / FS_ORIG)
# raw_red_ds  = resample(raw_red,   n_tgt)
# raw_ir_ds   = resample(raw_ir,    n_tgt)
# filt_red_ds = resample(red_filt,  n_tgt)   # <<< 新增
# filt_ir_ds  = resample(ir_filt,   n_tgt)   # <<< 新增
# dc_red_ds   = resample(red_dc,    n_tgt)
# dc_ir_ds    = resample(ir_dc,     n_tgt)
# red_ds      = resample(red_feat,  n_tgt)
# ir_ds       = resample(ir_feat,   n_tgt)

# # 4) SpO₂ 数据处理
# spo2 = data['spo2'][::2]
# if spo2.ndim > 1:
#     spo2_sec = np.nanmean(spo2, axis=1)
# else:
#     spo2_sec = spo2.copy()
# spo2_ds = np.repeat(spo2_sec, FS_TARGET)[:n_tgt]

# # 5) 构造时间轴并过滤 NaN
# t_ds = np.arange(n_tgt) / FS_TARGET
# mask = (
#     ~np.isnan(raw_red_ds) &
#     ~np.isnan(raw_ir_ds)  &
#     ~np.isnan(filt_red_ds) &
#     ~np.isnan(filt_ir_ds)  &
#     ~np.isnan(dc_red_ds)   &
#     ~np.isnan(dc_ir_ds)    &
#     ~np.isnan(red_ds)      &
#     ~np.isnan(ir_ds)       &
#     ~np.isnan(spo2_ds)          
# )
# t_ds        = t_ds[mask]
# raw_red_ds  = raw_red_ds[mask]
# raw_ir_ds   = raw_ir_ds[mask]
# dc_red_ds   = dc_red_ds[mask]
# dc_ir_ds    = dc_ir_ds[mask]
# red_ds      = red_ds[mask]
# ir_ds       = ir_ds[mask]
# spo2_ds     = spo2_ds[mask]

# # 6) 按时间段画图
# for start_sec, end_sec in segments_sec:
#     i0 = int(start_sec * FS_TARGET)
#     i1 = int(end_sec   * FS_TARGET)

        
#     fig, axes = plt.subplots(9, 1,
#                              figsize=(8, 12),
#                              sharex=True,
#                              facecolor='white')

#     # raw_red
#     axes[0].plot(t_ds[i0:i1], raw_red_ds[i0:i1], color='black')
#     # axes[0].set_ylabel('raw_red', rotation=0, va='center')

#     # filt_red
#     axes[1].plot(t_ds[i0:i1], filt_red_ds[i0:i1], color='black')
#     # axes[1].set_ylabel('filt_red', rotation=0, va='center')

#     # dc_red
#     axes[2].plot(t_ds[i0:i1], dc_red_ds[i0:i1],  color='black')
#     # axes[2].set_ylabel('dc_red', rotation=0, va='center')

#     # feat_red
#     axes[3].plot(t_ds[i0:i1], red_ds[i0:i1],     color='black')
#     # axes[3].set_ylabel('feat_red', rotation=0, va='center')

#     # raw_ir
#     axes[4].plot(t_ds[i0:i1], raw_ir_ds[i0:i1],  color='black')
#     # axes[4].set_ylabel('raw_ir', rotation=0, va='center')

#     # filt_ir
#     axes[5].plot(t_ds[i0:i1], filt_ir_ds[i0:i1],  color='black')
#     # axes[5].set_ylabel('filt_ir', rotation=0, va='center')

#     # dc_ir
#     axes[6].plot(t_ds[i0:i1], dc_ir_ds[i0:i1],    color='black')
#     # axes[6].set_ylabel('dc_ir', rotation=0, va='center')

#     # feat_ir
#     axes[7].plot(t_ds[i0:i1], ir_ds[i0:i1],       color='black')
#     # axes[7].set_ylabel('feat_ir', rotation=0, va='center')

#     # SpO₂
#     axes[8].plot(t_ds[i0:i1], spo2_ds[i0:i1],     color=colors['spo2'], linewidth=2)
#     axes[8].set_ylabel('SpO₂ (%)', rotation=0, va='center')
#     axes[8].set_xlabel('Time (s)')

#     for ax in axes:
#         ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False)
#         for spine in ax.spines.values():
#             spine.set_visible(False)
#     # axes[-1].tick_params(labelbottom=True)

#     # fig.suptitle("Public Dataset (Openoximetry Repository)", fontsize=fontsize)
#     # fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('imgs/ppg_preprocess.svg')
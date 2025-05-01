import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt,resample
from scipy.stats import pearsonr
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import mean_absolute_error

# —— 配置区 ——
DATASET_DIR = 'dataset'
FS = 86
FS_TARGET  = 25
# SEGMENT_SEC = 6            # 每段长度（秒）
STRIDE_SEC = 1              # 滑动步长（秒）
LOWPASS_CUTOFF = 12
HIGHPASS_CUTOFF = 0.5
SMOOTH_WIN = 1             # R值平滑窗口大小（单位：点）

from scipy.signal import find_peaks

def plot_peaks(signal,peaks):
    t = np.arange(len(signal)) / FS_TARGET
    plt.figure(figsize=(10, 4))
    plt.plot(t, signal, label='Signal')
    plt.plot(t[peaks], signal[peaks], 'ro', label='Peaks')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def maxim_ratio_extract(ir_segment, red_segment):
    peaks, _ = find_peaks(-ir_segment, distance=FS_TARGET//2)

    ratio_list = []

    for i in range(len(peaks) - 1):
        start = peaks[i]
        end = peaks[i + 1]

        # IR 峰值和插值 valley
        ir_max_idx = np.argmax(ir_segment[start:end]) + start
        ir_valley_est = ir_segment[start] + (ir_segment[end] - ir_segment[start]) * ((ir_max_idx - start) / (end - start))
        ir_ac = ir_segment[ir_max_idx] - ir_valley_est
        ir_dc = ir_segment[ir_max_idx]

        # RED 同理
        red_max_idx = np.argmax(red_segment[start:end]) + start
        red_valley_est = red_segment[start] + (red_segment[end] - red_segment[start]) * ((red_max_idx - start) / (end - start))
        red_ac = red_segment[red_max_idx] - red_valley_est
        red_dc = red_segment[red_max_idx]

        if ir_dc > 0 and red_dc > 0:
            ratio = (red_ac / red_dc) / (ir_ac / ir_dc)
            ratio_list.append(ratio)

    if len(ratio_list) == 0:
        return np.nan
    return np.mean(ratio_list)

def maxim_ratio_extract_log(ir_segment, red_segment):
    # ir_filt=highpass_filter(ir_segment)
    peaks, _ = find_peaks(-ir_segment, distance=FS_TARGET//2)  # find troughs of IR
    # plot_peaks(ir_segment,peaks)
    # exit(1)
    ratio_list = []

    for i in range(len(peaks) - 1):
        start = peaks[i]
        end = peaks[i + 1]

        ir_max = np.max(ir_segment[start:end])
        ir_min = (ir_segment[start] + ir_segment[end]) / 2.0
        red_max = np.max(red_segment[start:end])
        red_min = (red_segment[start] + red_segment[end]) / 2.0

        if ir_max > 0 and ir_min > 0 and red_max > 0 and red_min > 0:
            ratio = np.log(red_max / red_min) / np.log(ir_max / ir_min)
            ratio_list.append(ratio)

    if len(ratio_list) == 0:
        return np.nan

    return np.mean(ratio_list)


def lowpass_filter(signal, fs=FS, cutoff=LOWPASS_CUTOFF, order=4):
    b, a = butter(order, cutoff / (fs / 2), btype='low')
    return filtfilt(b, a, signal)

def highpass_filter(signal, fs=FS, cutoff=HIGHPASS_CUTOFF, order=4):
    b, a = butter(order, cutoff / (fs / 2), btype='high')
    return filtfilt(b, a, signal)

def extract_features_from_ppg(ppg_segment):
    # ac = np.std(ppg_segment)
    ac= np.max(ppg_segment)-np.min(ppg_segment)
    dc = np.mean(ppg_segment)
    return ac, dc

def moving_average(x, window_size):
    return np.convolve(x, np.ones(window_size)/window_size, mode='valid')

def remove_outliers_per_spo2(R_all: np.ndarray, y_all: np.ndarray):
    """
    For each distinct SpO₂ level in y_all (e.g. 100,99,...,70),
    compute the mean µ and std s of the corresponding R_all values,
    and drop any points where |R - µ| > 2s.
    
    Returns:
        R_clean, y_clean: the masked arrays with outliers removed.
    """
    # Make a boolean mask, start with everything kept
    mask = np.ones_like(y_all, dtype=bool)
    
    # Loop over the integer levels 70..100
    for spo2_level in range(100, 0, -1):
        # find indices with exactly that SPO2
        idx = (y_all == spo2_level)
        if not np.any(idx):
            continue
        
        vals = R_all[idx]
        mu, sigma = vals.mean(), vals.std(ddof=0)
        # build keep‐mask for this level
        keep = np.abs(vals - mu) <= 2 * sigma
        
        # apply it
        mask[idx] = keep
    
    return R_all[mask], y_all[mask]

def process_file(filepath,SEGMENT_SEC):
    data = np.load(filepath)
    red = data['red']
    ir = data['ir']
    spo2 = data['spo2']
    if spo2.shape[0]<10:
        return
    spo2_avg = np.nanmean(spo2, axis=1)

    secs, f0 = red[::2].shape
    n_tgt = int(secs * FS_TARGET)
    # 下采样
    red_seq = red[::2].reshape(-1)
    ir_seq = ir[::2].reshape(-1)
    spo2_avg = spo2_avg[::2].reshape(-1)

    # 滤波
    red_filt = lowpass_filter(red_seq)
    ir_filt = lowpass_filter(ir_seq)

    red_filt = resample(red_filt, n_tgt)
    ir_filt  = resample(ir_filt,  n_tgt)

    spo2_avg = np.repeat(spo2_avg, FS_TARGET)

    assert len(red_filt) == len(ir_filt) == len(spo2_avg), \
    f"Length mismatch: {len(red_filt)}, {len(ir_filt)}, {len(spo2_avg)}"

    stride_samples = FS_TARGET * STRIDE_SEC
    seg_len_samples = FS_TARGET * SEGMENT_SEC

    R = []
    t = []
    y_list = []
    for i in range(0, len(red_filt) - seg_len_samples+1, stride_samples):
        seg_r = red_filt[i:i+seg_len_samples]
        seg_i = ir_filt[i:i+seg_len_samples]
        seg_o = spo2_avg[i:i+seg_len_samples]
        if np.isnan(seg_r).any() or np.isnan(seg_i).any() or np.isnan(seg_o).any():
            continue
        tgt = float(np.mean(seg_o))
        # ac_r, dc_r = extract_features_from_ppg(seg_r)
        # ac_i, dc_i = extract_features_from_ppg(seg_ir)

        # if dc_r == 0 or dc_i == 0:
        #     continue

        # r_val = (ac_r / dc_r) / (ac_i / dc_i)
        r_val = maxim_ratio_extract(seg_i, seg_r)
        R.append(r_val)
        y_list.append(tgt)
        # t.append(i / FS_TARGET)

    

    R = np.array(R)
    try:
        R_smooth = moving_average(R, SMOOTH_WIN)
    except:
        # print(filepath,red.shape,ir.shape)
        return
    t_smooth = np.arange(len(R_smooth)) #t[:len(R_smooth)]
    t_spo2 = np.arange(len(y_list))

    return R_smooth,t_smooth,np.array(y_list),t_spo2
    # —— 可视化 ——（作为子图）
    # fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    # fig.suptitle(f'{os.path.basename(filepath)}', fontsize=14)

    # spo2_pred=-9.822494*(-R)*(-R) + 9.224996*(-R) + 106.180838
    # spo2_pred = np.clip(spo2_pred, 0, 100)

    # spo2_pred = moving_average(spo2_pred, SMOOTH_WIN)
    # t_smooth = t[:len(spo2_pred)]

    # axs[0].plot(t_smooth, spo2_pred, color='red', label='spo2_pred')
    # axs[0].set_ylabel('spo2_pred')
    # axs[0].grid(True)
    # axs[0].legend()
    # axs[0].set_ylim(0,110)

    # axs[1].plot(t_spo2,spo2_avg, label='SpO₂')
    # axs[1].set_xlabel('Sample index')
    # axs[1].set_ylabel('SpO₂ (%)')
    # axs[1].grid(True)
    # axs[1].legend()
    # axs[1].set_ylim(0,110)

    # common_xlim = (0, min(t_smooth[-1], t_spo2[-1]))
    # axs[0].set_xlim(common_xlim)
    # axs[1].set_xlim(common_xlim)

    # plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.show()

    return R_smooth,t_smooth,spo2_avg,t_spo2

def get_best_pcc_with_lag(X, y, max_lag):
    best_pcc = 0
    best_lag = 0

    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            X_shifted = X[:-lag]
            y_shifted = y[lag:]
        elif lag < 0:
            X_shifted = X[-lag:]
            y_shifted = y[:lag]
        else:
            X_shifted = X
            y_shifted = y

        if len(X_shifted) < 10:  # 太短就跳过
            continue

        mask2 = np.nanstd(X_shifted) > 1e-6 and np.nanstd(y_shifted) > 1e-6
        if not mask2:
            continue
        r, _ = pearsonr(X_shifted, y_shifted)
        if abs(r) > abs(best_pcc):
            best_pcc = r
            best_lag = lag

    return best_lag, best_pcc

def Traverse(SEGMENT_SEC):
    coeffs_a = []
    coeffs_b = []
    coeffs_c = []
    cnt=0
    encounter_id=[]
    # pdf = PdfPages('20samples_plots.pdf')

    splits = pd.read_csv('encounter_5folds.csv')
    splits['fold'] = splits['fold'].fillna(-1).astype(int)
    for fold in range(5):
        # 2) 拆 train / val
        train_encs = set(
            splits[(splits['set']=='trainval') & (splits['fold'] != fold)]['encounter_id']
        )
        val_encs = set(
            splits[(splits['set']=='trainval') & (splits['fold'] == fold)]['encounter_id']
        )
        X_all = []
        y_all = []


        for fname in sorted(os.listdir(DATASET_DIR)):
            x_sub=[]
            y_sub=[]
            if fname.endswith('.npz'):
                enc_id = fname[:-4]
                if enc_id not in train_encs:
                    continue

                res =process_file(os.path.join(DATASET_DIR, fname),SEGMENT_SEC)

                if res is None:
                    # print(fname)
                    continue

                R_list, idx_list, spo2, spo2_time  = res

                # for r, idx in zip(R_list, idx_list):
                #     if np.isnan(r): 
                #         continue
                #     idx=int(idx)
                #     if idx < 0 or idx >= len(spo2): 
                #         continue
                #     val = spo2[idx]
                #     if np.isnan(val): 
                #         continue

                #     x_sub.append(-r)
                #     y_sub.append(val)
                X = np.array(R_list)
                y = np.array(spo2)
                # mask = (~np.isnan(X)) & (~np.isnan(y))
                # X = X[mask]
                # y = y[mask]
                # max_lag=300
                # if len(X[mask])<=max_lag:
                #     # print(f"len(X[mask])<=")
                #     continue

                # lag, pcc = get_best_pcc_with_lag(-X, y,max_lag)
                # pcc, p_value = pearsonr(X[mask], y[mask])
                # if pcc<0.9:
                #     print(f"Lag: {lag:.4f}")
                #     print(f"Pearson correlation coefficient (PCC): {pcc:.4f}")
                #     continue

                # if lag > 0:
                #     Xs = X[:-lag]
                #     ys = y[ lag:]
                # elif lag < 0:
                #     Xs = X[-lag:]
                #     ys = y[: lag]
                # else:
                #     Xs = X
                #     ys = y

                # 6) 累加到全局
                # a, b, c = np.polyfit(Xs, ys, 2)
                # coeffs_a.append(a)
                # coeffs_b.append(b)
                # coeffs_c.append(c)
                X_all.extend(X)
                y_all.extend(y)
                # cnt += 1

        X_all = np.array(X_all)
        y_all = np.array(y_all)
        R_clean, y_clean = remove_outliers_per_spo2(X_all, y_all)
        a, b, c = np.polyfit(R_clean, y_clean, 2)
        # print(f"处理了 {cnt} 个文件")


        y_val_trues, y_val_preds = [], []
        for fname in sorted(os.listdir(DATASET_DIR)):
            if not fname.endswith('.npz'): 
                continue
            enc = fname[:-4]
            if enc not in val_encs:
                continue

            res = process_file(os.path.join(DATASET_DIR, fname),SEGMENT_SEC)
            if res is None:
                continue
            R_list, idx_list, spo2, _ = res

            Xv = np.array(R_list)
            yv = np.array(spo2)

            # 用当前 (a,b,c) 预测
            y_pred = a * Xv**2 + b * Xv + c
            y_pred[y_pred>100]=100
            y_pred[y_pred<0]=0
            y_val_trues.extend(yv)
            y_val_preds.extend(y_pred)

        y_val_trues = np.array(y_val_trues)
        y_val_preds = np.array(y_val_preds)

        val_mae = mean_absolute_error(y_val_trues, y_val_preds)

        print(f"SEGMENT_SEC {SEGMENT_SEC}, Fold {fold}: a={a:.4f}, b={b:.4f}, c={c:.4f}, Val MAE={val_mae:.4f}")


    # coeffs_a = np.array(coeffs_a)
    # coeffs_b = np.array(coeffs_b)
    # coeffs_c = np.array(coeffs_c)

    # print(f"共处理了 {cnt} 个文件。")
    # print("a 系数 → mean: {:.6f}, var: {:.6f}".format(coeffs_a.mean(), coeffs_a.var()))
    # print("b 系数 → mean: {:.6f}, var: {:.6f}".format(coeffs_b.mean(), coeffs_b.var()))
    # print("c 系数 → mean: {:.6f}, var: {:.6f}".format(coeffs_c.mean(), coeffs_c.var()))
    # df = pd.DataFrame({'encounter_id': encounter_id})
    # csv_path = 'encounter_id.csv'
    # df.to_csv(csv_path, index=False)
# Traverse()

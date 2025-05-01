import os
import pandas as pd
import numpy as np
import wfdb


def get_ppg_start_time(hea_path):
    """从 .hea 文件第一行解析 PPG 起始时间"""
    with open(hea_path, 'r') as f:
        first = f.readline().strip().split()
        # 通常格式：<rec_name> <n_signals> <fs> <n_samples> <HH:MM:SS.xxxxxx>
        time_str = first[4]
        return pd.to_datetime(time_str)

def main():
    WAVEFORMS_DIR = '../waveforms'
    WINDOW_SEC    = 0.5

    features = []
    labels   = []

    # 用 pulse_df 里的 encounter_id 遍历
    pulse_df = pd.read_csv('../pulseoximeter.csv')
    for enc_id, enc_group in pulse_df.groupby('encounter_id'):
        bucket = enc_id[0]
        # 这里的 base 正好指向 waveforms/<bucket>/<enc_id>_*
        base = os.path.join(WAVEFORMS_DIR, bucket, enc_id)

        hea_path = base + '_ppg.hea'
        dat_path = base + '_ppg.dat'
        csv2hz   = base + '_2hz.csv'

        # 1) 读取 PPG 波形及元数据
        try:
            rec = wfdb.rdrecord(base + '_ppg')  
        except FileNotFoundError:
            continue

        sig  = rec.p_signal
        names= rec.sig_name
        fs   = rec.fs  # e.g. 86
        # 定位红光/红外通道
        if len(names) in (4, 8):
            i_red = names.index("Red Signal")
            i_ir  = names.index("IR Signal")
        else:
            i_red = names.index("RED")
            i_ir  = names.index("IR")

        red_f = sig[:,i_red]
        ir_f  = sig[:,i_ir]

        # 2) 读取 PPG 起始时间
        try:
            ppg_start = get_ppg_start_time(hea_path)
        except Exception:
            # 如果 .hea 没有起始时间，就退回用 2Hz 的第一个 Timestamp
            print('hea has no timestamp')
            continue

        # 3) 读取 2Hz 文件
        try:
            df2 = pd.read_csv(csv2hz)
        except FileNotFoundError:
            continue

         # 把它转换成从当天0点开始的秒数
        ppg_start_sec = ppg_start.hour*3600 + ppg_start.minute*60 \
                        + ppg_start.second + ppg_start.microsecond/1e6
        ppg_end = ppg_start_sec + len(red_f) / fs
        # 3) 读 2Hz
        try:
            df2 = pd.read_csv(csv2hz)
        except FileNotFoundError:
            continue
        half = int(WINDOW_SEC * fs)
        df2['Timestamp'] = pd.to_datetime(df2['Timestamp'])
        # 新增：只要时分秒，不管日期
        df2['sec_of_day'] = (
            df2['Timestamp'].dt.hour*3600 +
            df2['Timestamp'].dt.minute*60 +
            df2['Timestamp'].dt.second +
            df2['Timestamp'].dt.microsecond/1e6
        )

        df2 = df2[(df2['sec_of_day'] >= ppg_start_sec) & (df2['sec_of_day'] <= ppg_end)]
        spo2_cols = [c for c in df2.columns if c.endswith('_SpO2')]
        red_segs = []
        ir_segs  = []
        spo2_mat = []
        # 3) 再遍历 df2，就完全和 PPG 范围对齐了
        for _, row in df2.iterrows():
            delta = row['sec_of_day'] - ppg_start_sec   
            if delta < 0:
                continue
            idx   = int(delta * fs)
            if idx-half < 0 or idx+half > len(red_f):
                continue

            red_segs.append(red_f[idx-half:idx+half])
            ir_segs.append(ir_f [idx-half:idx+half])
            # 构造该行的 spo2 向量（nan 表示缺失）
            spo2_vec = row[spo2_cols].astype(float).values
            spo2_mat.append(spo2_vec)

            # —— 取所有设备的 SpO2 平均值 —— 
            
        if not red_segs or not spo2_mat:
            continue
        spo2_arr = np.vstack(spo2_mat)             # (n_samples, n_devices)
        if np.isnan(spo2_arr).all():
            continue
        red_arr  = np.stack(red_segs)              # (n_samples, window_len)
        ir_arr   = np.stack(ir_segs)
        cols_arr = np.array(spo2_cols, dtype='<U32')
        
        outp = os.path.join('dataset', f"{enc_id}.npz")
        np.savez(
            outp,
            encounter_id=enc_id,
            red=red_arr,
            ir=ir_arr,
            spo2=spo2_arr,
            spo2_cols=cols_arr
        )
        print(red_arr.shape,ir_arr.shape,spo2_arr.shape,cols_arr.shape)

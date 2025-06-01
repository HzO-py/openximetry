import os
from matplotlib import ticker
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import *
from sklearn.metrics import mean_absolute_error
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from spo2 import *

def ml():
    enc_id='0b9291e9836c9094225411caac27ee83efcbf08643f864179cef708e79e9474d'

    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_dir= 'dataset'
    split_csv  = 'encounter_5folds.csv'
    batch_size = 64

    splits     = pd.read_csv(split_csv)

    model = AttentionBiLSTM(input_dim=2, hidden_dim=128, num_layers=2, dropout=0.3)
    # model = TransformerRegressor(input_dim=2,d_model=128,
    #                                 nhead=4,num_layers=2,dim_feedforward=128,dropout=0)
    # model     = BiLSTM(input_dim=2, hidden_dim=128,
    #                         num_layers=2, dropout=0.3)
    # model=CNNBaseline(input_dim=2)
    model.load_state_dict(torch.load(f'models/25hz_lstm_att_fold1_win5.pth', map_location=device))
    model.to(device).eval()

    metrics = []
    df_enc = splits[splits['encounter_id']==enc_id].copy()

    ds = PPGRegressionDataset(
        dataset_dir, df_enc, 
        fold=None, mode='test'
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    y_true, y_pred = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            out = model(Xb).squeeze(1).cpu().numpy()
            y_pred.extend(out)
            y_true.extend(yb.numpy())

    y_true = np.array(y_true)*100
    y_pred = np.array(y_pred)*100

    times = np.arange(len(y_true))   # SLIDE_SEC = 1
    y_pred[y_pred>100]=100
    y_pred[y_pred<0]=0
    y_pred = moving_average(y_pred, 5)
    times=times[:len(y_pred)]
    y_true=y_true[:len(y_pred)]

    np.savez(
    'imgs/bilstm_att.npz',
    times   = times.astype('datetime64[ns]').astype(np.int64),  # 转成 int64 存储
    y_true  = y_true,
    y_pred  = y_pred
)


def abc():
# Polynomial coefficients from training
    # a,b,c=3.582110, -32.101080, 114.750785
    SEGMENT_SEC=5
    enc_id='0b9291e9836c9094225411caac27ee83efcbf08643f864179cef708e79e9474d'
    a=2.1552
    b=-28.4494
    c=112.8707
    # Iterate over each test file and plot curves
    npz_path = os.path.join('dataset', f"{enc_id}.npz")
    
    res = process_file(npz_path,SEGMENT_SEC)
    R_smooth, t_smooth, spo2_avg, t_spo2 = res
    
    # Ensure equal length
    n = min(len(R_smooth), len(spo2_avg))
    # print(len(R_smooth),len(spo2_avg))
    R = np.array(R_smooth[:n])
    true_spo2 = np.array(spo2_avg[:n])
    times = np.array(t_smooth[:n])
    
    # Predict
    pred_spo2 = a * R**2 + b * R + c
    pred_spo2[pred_spo2>100]=100
    pred_spo2[pred_spo2<0]=0
    pred_spo2 = moving_average(pred_spo2, 5)
    times=times[:len(pred_spo2)]
    true_spo2=true_spo2[:len(pred_spo2)]

    np.savez(
    'imgs/abc.npz',
    times   = times.astype('datetime64[ns]').astype(np.int64),  # 转成 int64 存储
    y_true  = true_spo2,
    y_pred  = pred_spo2
)

def compare_and_report(npz_files: dict):

    times_list = []
    for p in npz_files.values():
        data = np.load(p)
        times = pd.to_datetime(data['times'])
        times_list.append(times)

    common_times = times_list[0]
    for ts in times_list[1:]:
        common_times = common_times.intersection(ts)
    common_times = common_times.unique().sort_values()

    # 3) 构建相对秒数横轴
    t0     = common_times[0]
    x_sec  = (common_times - t0) / pd.Timedelta(seconds=1)

    # 4) 画布
    fig, ax = plt.subplots(figsize=(6,2.5))
    cmap = plt.get_cmap('tab10')

    # 5) 先画公共的真值（用第一个模型的 y_true）
    data0       = np.load(list(npz_files.values())[0])
    y_true_full = pd.Series(data0['y_true'],
                            index=pd.to_datetime(data0['times']))
    y_true_comm = y_true_full.reindex(common_times)
    ax.plot(np.arange(len(y_true_comm.values)), y_true_comm.values,
            color='tab:blue', linestyle='-',
            linewidth=1, label='Reference')

    # 6) 画每个模型的预测
    records = []
    for idx, (name, p) in enumerate(npz_files.items()):
        data     = np.load(p)
        y_pred_full = pd.Series(data['y_pred'],
                                index=pd.to_datetime(data['times']))
        # 只重索引到 common_times，不做插值，缺失处即 NaN
        y_pred_comm = y_pred_full.reindex(common_times)

        # 计算 MAE/RMSE 只在非 NaN 处
        mask = ~y_pred_comm.isna()
        y_t = y_true_comm[mask]
        y_p = y_pred_comm[mask]
        mae  = mean_absolute_error(y_t, y_p)
        rmse = mean_squared_error(y_t, y_p, squared=False)
        pcc  = pearsonr(y_t, y_p)[0]

        records.append({
            'model'     : name,
            'MAE'       : mae,
            'RMSE'      : rmse,
            'PCC'       : pcc
        })

        ax.plot(np.arange(len(y_pred_comm.values)), y_pred_comm.values,
                linestyle='--',
                linewidth=1.2,
                color=cmap(idx+2),
                label=name)

    import matplotlib.patches as mpatches

    # 用 fig.legend 在画布顶端外面重画 legend
    # 7) 美化 & 图例
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('SpO₂ (%)', fontsize=10)
    ax.ticklabel_format(style='plain', axis='x')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune='both'))


    fig.tight_layout(rect=[0,0,1,0.8])
    fig.legend(loc='upper center',
            bbox_to_anchor=(0.5,0.95),
            ncol=len(npz_files)+1,
            frameon=False,
            fontsize=9)
    fig.savefig('imgs/public_spo2_compare_paper.pdf',
                dpi=300,
                bbox_inches='tight')
    plt.show()
    plt.close(fig)

    # 8) 指标表
    metrics_df = pd.DataFrame.from_records(records).set_index('model')
    print(metrics_df)

# abc()

# ml()
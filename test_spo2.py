import os
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
from spo2 import moving_average

def main():
    # 1) 配置
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_dir= 'dataset'
    split_csv  = 'encounter_5folds.csv'
    batch_size = 64

    # 2) 读分割表，拿 test 里所有 encounter_id
    splits     = pd.read_csv(split_csv)
    test_ids   = splits[splits['set']=='test']['encounter_id'].tolist()

    # 3) 创建模型并加载最佳权重
    # model = AttentionBiLSTM(input_dim=2, hidden_dim=128, num_layers=2, dropout=0.3)
    model = TransformerRegressor(input_dim=2,d_model=128,
                                    nhead=4,num_layers=2,dim_feedforward=128,dropout=0)
    # model     = BiLSTM(input_dim=2, hidden_dim=128, num_layers=2, dropout=0.3)
    # model=CNNBaseline(input_dim=2)
    # model = TCNBaseline(
    #     input_dim=2,             # 与 BiLSTM 的 input_dim=2 对应
    #     tcn_channels=[64, 128],  # 两层 TCN：第一层 64 通道、第二层 128 通道
    #     kernel_size=3,           # 卷积核大小
    #     dropout=0.3              # dropout 概率
    # )
    model.load_state_dict(torch.load(f'models/25hz_lstm_att_fold0_win5.pth', map_location=device))
    model.to(device).eval()

    y_trues=[]
    y_preds=[]
    metrics = []
    # 4) 对每个 encounter 单独推理并画图
    pdf_path = f'imgs/25hz_lstm_att_win{WINDOW_SEC}.pdf'
    with PdfPages(pdf_path) as pdf:
        for enc_id in test_ids:
            # 筛出单文件的 split_df
            df_enc = splits[splits['encounter_id']==enc_id].copy()

            # 构造 dataset + loader
            ds = PPGRegressionDataset(
                dataset_dir, df_enc, 
                fold=None, mode='test'
            )
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

            # 收集 y_true, y_pred
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
            t_pred=times[:len(y_pred)]
            y_true=y_true[:len(y_pred)]

            y_trues.extend(y_true)
            y_preds.extend(y_pred)



            mae     = mean_absolute_error(y_true, y_pred)

            rmse = mean_squared_error(y_true, y_pred, squared=False)
            pcc  = pearsonr(y_true, y_pred)[0]
            metrics.append((mae, rmse, pcc))

            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(t_pred, y_true, label='True SpO₂',      linewidth=2)
            ax.plot(t_pred, y_pred, label='Predicted SpO₂', linewidth=2, alpha=0.8)
            ax.set_title   (f"Encounter {enc_id} — MAE: {mae:.2f}")
            ax.set_xlabel  ('Time (s)')
            ax.set_ylabel  ('SpO₂ (%)')
            ax.set_ylim    (50, 105)
            ax.legend     ()
            ax.grid       (True)
            fig.tight_layout()

            # save this page into the PDF
            pdf.savefig(fig)
            # plt.show()
            # plt.close(fig)


    # 5) 计算 MAE 并转换到百分比

        arr = np.array(metrics)

        print("MAE  {:.3f}±{:.3f}".format(arr[:,0].mean(), arr[:,0].std()))
        print("RMSE {:.3f}±{:.3f}".format(arr[:,1].mean(), arr[:,1].std()))
        print("PCC  {:.3f}±{:.3f}".format(arr[:,2].mean(), arr[:,2].std()))

main()
########################################################################################################################
# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import torch
# from sklearn.metrics import mean_absolute_error
# from matplotlib.backends.backend_pdf import PdfPages
# from sklearn.metrics import mean_squared_error
# from scipy.stats import pearsonr
# # Make sure process_file and model coefficients are imported
# from spo2 import *
# SEGMENT_SEC=5
# def main():
# # Polynomial coefficients from training
#     a=3.1182
#     b=-30.4533
#     c=113.7072
#     # a=2.1552
#     # b=-28.4494
#     # c=112.8707
#     pdf_path = f'imgs/old86hz_win{SEGMENT_SEC}.pdf'
#     with PdfPages(pdf_path) as pdf:
#         # Load test split
#         splits = pd.read_csv('encounter_5folds.csv', dtype=str)
#         test_encs = set(splits.loc[splits['set']=='test', 'encounter_id'])
#         all_true = []
#         all_pred = []
#         metrics=[]
#         # Iterate over each test file and plot curves
#         for enc_id in sorted(test_encs):
#             npz_path = os.path.join('dataset', f"{enc_id}.npz")
#             if not os.path.exists(npz_path):
#                 continue
            
#             res = process_file(npz_path,SEGMENT_SEC)
#             if res is None:
#                 continue
#             R_smooth, t_smooth, spo2_avg, t_spo2 = res
            
#             # Ensure equal length
#             n = min(len(R_smooth), len(spo2_avg))
#             # print(len(R_smooth),len(spo2_avg))
#             R = np.array(R_smooth[:n])
#             true_spo2 = np.array(spo2_avg[:n])
#             times = np.array(t_smooth[:n])
            
#             # Predict
#             pred_spo2 = a * R**2 + b * R + c
#             pred_spo2[pred_spo2>100]=100
#             pred_spo2[pred_spo2<0]=0
#             pred_spo2 = moving_average(pred_spo2, 5)
#             t_pred=times[:len(pred_spo2)]
#             true_spo2=true_spo2[:len(pred_spo2)]
#             all_true.append(true_spo2)
#             all_pred.append(pred_spo2)

#             # Compute MAE for this file
#             # mae = mean_absolute_error(true_spo2, pred_spo2)

#             mae     = mean_absolute_error(true_spo2, pred_spo2)

#             rmse = mean_squared_error(true_spo2, pred_spo2, squared=False)
#             pcc  = pearsonr(true_spo2, pred_spo2)[0]
#             metrics.append((mae, rmse, pcc))

#             # create a new Figure
#             fig, ax = plt.subplots(figsize=(10,4))
#             ax.plot(t_pred, true_spo2, label='True SpO₂', linewidth=2)
#             ax.plot(t_pred, pred_spo2, label='Predicted SpO₂', linewidth=2, alpha=0.8)
#             ax.set_title(f"Encounter {enc_id} — MAE: {mae:.2f}")
#             ax.set_xlabel("Time (s)")
#             ax.set_ylabel("SpO₂ (%)")
#             ax.set_ylim(50, 105)
#             ax.legend()
#             ax.grid(True)
#             fig.tight_layout()

#             # save this figure into the PDF
#             pdf.savefig(fig)
#             # plt.show()
#             # plt.close(fig)   # free memory

#         # all_true = np.concatenate(all_true)
#         # all_pred = np.concatenate(all_pred)

#         # 计算全局 MAE
#         # mae = mean_absolute_error(all_true, all_pred)
#         # print(f"Overall Test MAE: {mae:.4f} (i.e. {mae:.2f} percentage points)")
#         arr = np.array(metrics)

#         print("MAE  {:.3f}±{:.3f}".format(arr[:,0].mean(), arr[:,0].std()))
#         print("RMSE {:.3f}±{:.3f}".format(arr[:,1].mean(), arr[:,1].std()))
#         print("PCC  {:.3f}±{:.3f}".format(arr[:,2].mean(), arr[:,2].std()))

# main()
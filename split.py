import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

def main():
    # —— 1. 读入数据 —— 
    enc_ids = pd.read_csv('encounter_id.csv', dtype=str)
    mapping = pd.read_csv('../encounter.csv', dtype=str)

    df = (
        mapping
        .loc[mapping['encounter_id'].isin(enc_ids['encounter_id']), ['patient_id', 'encounter_id']]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # —— 2. 第一步：先划 TrainVal / Test —— 
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    trainval_idx, test_idx = next(gss.split(df, groups=df['patient_id']))

    df['set'] = None
    df.loc[test_idx, 'set'] = 'test'
    df.loc[trainval_idx, 'set'] = 'trainval'

    # —— 3. 第二步：在 TrainVal 里做 5 折 GroupKFold —— 
    trainval = df[df['set'] == 'trainval'].reset_index(drop=True)
    trainval['fold'] = -1  # 初始化

    gkf = GroupKFold(n_splits=5)
    for fold_number, (_, val_idx) in enumerate(gkf.split(trainval, groups=trainval['patient_id'])):
        trainval.loc[val_idx, 'fold'] = fold_number

    # —— 4. 合并回 df —— 
    df = df.merge(trainval[['encounter_id', 'fold']], on='encounter_id', how='left')

    # —— 5. 检查 —— 
    assert (df.loc[df['set'] == 'trainval', 'fold'] != -1).all(), "有 trainval 数据没有 fold！"
    assert df['set'].isnull().sum() == 0, "有 encounter 没有被分配 set！"

    # —— 6. 保存 —— 
    df.to_csv('encounter_5folds.csv', index=False)

    # —— 7. 打印结果 —— 
    print(df['set'].value_counts())
    print(df.groupby('fold')['patient_id'].nunique())
    print(df.groupby('set')['patient_id'].nunique())
    print(df.head())

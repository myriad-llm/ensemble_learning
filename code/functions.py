import pandas as pd
from tqdm import tqdm
import time
from utils.augmentation import Augmentation

def augment_data(train_data, train_labels, test_labels):
    """
    对每个 msisdn 进行数据增强，并将增强后的数据合并到原始数据中。

    Args:
        train_data (pd.DataFrame): 训练数据。
        train_labels (pd.DataFrame): 训练标签。
        test_labels (pd.DataFrame): 测试标签。

    Returns:
        pd.DataFrame: 增强后的训练数据。
        pd.DataFrame: 增强后的训练标签。
        pd.DataFrame: 增强后的标签。
    """
    addition_train_data = []
    addition_train_labels = []

    times=2
    ratio_range=0.1
    pbar = tqdm(train_data.groupby('msisdn'))
    for msisdn, group in pbar:
        if msisdn == 0:
            continue
        pbar.set_description(f"Augmenting msisdn {msisdn}")
        label = train_labels[train_labels['msisdn'] == msisdn].iloc[0]['is_sa']
        aug = Augmentation(group, label, 'msisdn', 'is_sa')
        if label == 1:
            res_df, res_labels = aug.times(ratio=ratio_range, times=3+4*times, method='mask')
            addition_train_data.append(res_df)
            addition_train_labels.append(res_labels)
        else:
            res_df, res_labels = aug.times(ratio=ratio_range, times=times, method='mask')
            addition_train_data.append(res_df)
            addition_train_labels.append(res_labels)

    addition_train_data = pd.concat(addition_train_data)
    addition_train_labels = pd.concat(addition_train_labels)

    # 将新数据加入到train_data中
    train_data = pd.concat([train_data, addition_train_data], ignore_index=True).reset_index(drop=True)
    train_labels = pd.concat([train_labels, addition_train_labels], ignore_index=True).reset_index(drop=True)

    # 按照 msisdn, start_time 排序
    sort_start_time = time.time()
    train_data = train_data.sort_values(by=['msisdn', 'start_time']).reset_index(drop=True)
    train_labels = train_labels.sort_values(by=['msisdn']).reset_index(drop=True)
    print('sort time:', time.time() - sort_start_time)

    labels_aug = pd.concat([train_labels, test_labels], ignore_index=True).reindex()

    return train_data, train_labels, labels_aug

# 聚合特征
def aggregate_features(data):
    agg_features = data.groupby('msisdn').agg({
    'call_duration': [
        ('sum', 'sum'), 
        ('mean', 'mean'), 
        ('max', 'max'), 
        ('std', 'std'),
        ('quantile_25', lambda x: x.quantile(0.25)), 
        ('quantile_50', lambda x: x.quantile(0.50)), 
        ('quantile_75', lambda x: x.quantile(0.75)),
    ],
    'cfee': [
        ('sum', 'sum'),
        ('std', 'std'), 
        ('mean', 'mean'),
    ],
    'lfee': [
        ('sum', 'sum'), 
        ('mean', 'mean'),
        ('std', 'std'),
    ],
    'hour': [
        ('mean', 'mean'), 
        ('std', 'std'), 
        ('max', 'max'), 
        ('min', 'min'),
    ],
    'dayofweek': [
        ('std', 'std'), 
        ('magic', lambda x: x.value_counts().mean()), 
        ('work_day_num', lambda x: x[x.isin([1,2,3,4,5])].count()), 
        ('weekend_num', lambda x: x[x.isin([6,7])].count()),
        ('mode', lambda x: x.mode().values[0]),
        ('work_day_weekend_diff', lambda x: (x[x.isin([1,2,3,4,5])].count() - x[x.isin([6,7])].count()) / (x[x.isin([1,2,3,4,5])].count() + x[x.isin([6,7])].count())),
    ],
    # 'home_area_code': [
    #     ('home_area_code_nunique', 'nunique')
    # ],
    'visit_area_code': [
        ('nunique', 'nunique'),
        ('times_not_at_home_area', lambda x: x[x != x.shift()].count()/x.count())
    ],
    'called_home_code': [
        ('nunique', 'nunique'),
        ('called_diff_home_code', lambda x: x[x != x.shift()].count() / x.count())
    ],
    'called_code': [
        ('nunique', 'nunique'),
        ('diff', lambda x: x[x != x.shift()].count()/ x.count())
    ],
    'open_datetime': [
        ('open_count', 'nunique')
    ],
    'other_party': [
        ('account_person_num', 'nunique'),
        ('called_diff_home_code', lambda x: x[x != x.shift()].count() / x.count())
    ],
    'a_serv_type': [
        ('call_num', lambda x: x[x.isin([1, 3])].count()), 
        ('called_num', lambda x: x[x == 2].count()),
        ('call_called_normalized_diff', lambda x: (x[x.isin([1, 3])].count() - x[x == 2].count()) /  (x[x.isin([1, 3])].count() + x[x == 2].count())),
    ],
    'start_time_diff': [
        ('start_time_diff_mean', 'mean'), 
        ('start_time_diff_std', 'std'), 
        ('max', 'max'), 
        ('coefficient_of_variation', lambda x: x.std() / x.mean()),
    ], 
    # 'phone1_type': [
    #     ('nunique', 'nunique'),
    #     ('mode', lambda x: x.mode().values[0])
    # ],
    # 'distance': [
    #     ('sum', 'sum'), 
    #     ('std', 'std'), 
    #     ('max', 'max'), 
    #     ('quantile_25', lambda x: x.quantile(0.25)), 
    #     ('quantile_50', lambda x: x.quantile(0.50)), 
    #     ('quantile_75', lambda x: x.quantile(0.75)),
    # ]
    })

    agg_features.columns = ['+'.join(col).strip() for col in agg_features.columns.values]
    agg_features.columns = agg_features.columns.str.replace('[', '').str.replace(']', '').str.replace('<', '').str.replace('>', '').str.replace('(', '').str.replace(')', '').str.replace(',', '').str.replace(' ', '_')
    agg_features = agg_features.reset_index()
    return agg_features

def get_nan(train):
    # 获取 train 中的 nan值
    train_nan = train[train.isnull().T.any()]
    # 统计 每列含有的 nan 数量
    for col in train.columns:
        if train[col].isnull().sum() > 0:
            print(col, train[col].isnull().sum())

    return train_nan

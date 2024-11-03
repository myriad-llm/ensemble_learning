import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm
import time
from utils.augmentation import Augmentation
import warnings
from functools import partial

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
    warnings.warn("This function will be deprecated in the future, use augment_data_parallel instead.", DeprecationWarning)

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

def _augment_group(group_data, train_labels):
    addition_train_data = []
    addition_train_labels = []
    times = 2
    ratio_range = 0.1
    for msisdn, group in group_data.groupby('msisdn'):
        if msisdn == 0:
            continue
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
    return pd.concat(addition_train_data), pd.concat(addition_train_labels)

def augment_data_parallel(train_data, train_labels, test_labels, max_workers=0):
    if max_workers == 0:
        max_workers = os.cpu_count()

    # Split the data into max_workers groups
    groups = [group for _, group in train_data.groupby('msisdn')]
    print("len(groups):", len(groups))
    chunk_size = len(groups) // max_workers
    chunks = [pd.concat(groups[i - chunk_size:i]) for i in range(chunk_size, len(groups), chunk_size)]
    # BUG: 获取 len(groups) % max_workers 个数据合入到最后一个 chunk 中
    if len(groups) % max_workers != 0:
        rest_df = pd.concat(groups[-(len(groups) % max_workers):])
        chunks[-1] = pd.concat([chunks[-1], rest_df])

    addition_train_data = []
    addition_train_labels = []

    print("parallel augmenting data...")
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_augment_group, chunk, train_labels) for chunk in chunks]
        print("collecting results...")
        for future in as_completed(futures):
            res_data, res_labels = future.result()
            addition_train_data.append(res_data)
            addition_train_labels.append(res_labels)
    print("augmenting done, time:", time.time() - start_time)

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


def get_nan(train):
    # 获取 train 中的 nan值
    train_nan = train[train.isnull().T.any()]
    # 统计 每列含有的 nan 数量
    for col in train.columns:
        if train[col].isnull().sum() > 0:
            print(col, train[col].isnull().sum())

    return train_nan

# 聚合特征
def aggregate_features(data):
    warnings.warn("This function will be deprecated in the future, use aggregate_features_parallel instead.", DeprecationWarning)
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

class _CustomAggregator:

    @staticmethod
    def quantile(x, q):
        return x.quantile(q)

    @staticmethod
    def quantile_25(x):
        return x.quantile(0.25)

    @staticmethod
    def quantile_50(x):
        return x.quantile(0.5)

    @staticmethod
    def quantile_75(x):
        return x.quantile(0.75)

    # just for column 'dayofweek'
    @staticmethod
    def magic(x):
        return x.value_counts().mean()

    # just for column 'dayofweek'
    @staticmethod
    def work_day_num(x):
        return x[x.isin([1, 2, 3, 4, 5])].count()

    @staticmethod
    def weekend_num(x):
        return x[x.isin([6, 7])].count()

    @staticmethod
    def mode(x):
        return x.mode().values[0]

    @staticmethod
    def work_day_weekend_diff(x):
        work_days = x[x.isin([1, 2, 3, 4, 5])].count()
        weekends = x[x.isin([6, 7])].count()
        return (work_days - weekends) / (work_days + weekends)

    @staticmethod
    def times_not_at_home_area(x):
        return x[x != x.shift()].count() / x.count()

    @staticmethod
    def called_diff_home_code(x):
        return x[x != x.shift()].count() / x.count()

    @staticmethod
    def diff(x):
        return x[x != x.shift()].count() / x.count()

    @staticmethod
    def call_num(x):
        return x[x.isin([1, 3])].count()

    @staticmethod
    def called_num(x):
        return x[x == 2].count()

    @staticmethod
    def call_called_normalized_diff(x):
        calls = x[x.isin([1, 3])].count()
        called = x[x == 2].count()
        return (calls - called) / (calls + called)

    @staticmethod
    def coefficient_of_variation(x):
        return x.std() / x.mean()

def _aggregate_features(data, agg_params):
    agg_features = data.groupby('msisdn').agg(agg_params)
    agg_features.columns = ['+'.join(col).strip() for col in agg_features.columns.values]
    agg_features.columns = agg_features.columns.str.replace('[', '').str.replace(']', '').str.replace('<', '').str.replace('>', '').str.replace('(', '').str.replace(')', '').str.replace(',', '').str.replace(' ', '_')
    agg_features = agg_features.reset_index()
    return agg_features

# 并行处理函数
def aggregate_features_parallel(data, max_workers=0):

    agg_params = {
        'call_duration': [
            ('sum', 'sum'), 
            ('mean', 'mean'), 
            ('max', 'max'), 
            ('std', 'std'),
            ('quantile_25', partial(_CustomAggregator.quantile, q=0.25)),
            ('quantile_50', partial(_CustomAggregator.quantile, q=0.5)),
            ('quantile_75', partial(_CustomAggregator.quantile, q=0.75)),
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
            ('magic', _CustomAggregator.magic),
            ('work_day_num', _CustomAggregator.work_day_num),
            ('weekend_num', _CustomAggregator.weekend_num),
            ('mode', _CustomAggregator.mode),
            ('work_day_weekend_diff', _CustomAggregator.work_day_weekend_diff),
        ],
        'visit_area_code': [
            ('nunique', 'nunique'),
            ('times_not_at_home_area', _CustomAggregator.times_not_at_home_area)
        ],
        'called_home_code': [
            ('nunique', 'nunique'),
            ('called_diff_home_code', _CustomAggregator.called_diff_home_code)
        ],
        'called_code': [
            ('nunique', 'nunique'),
            ('diff', _CustomAggregator.diff)
        ],
        'open_datetime': [
            ('open_count', 'nunique')
        ],
        'other_party': [
            ('account_person_num', 'nunique'),
            ('called_diff_home_code', _CustomAggregator.called_diff_home_code)
        ],
        'a_serv_type': [
            ('call_num', _CustomAggregator.call_num), 
            ('called_num', _CustomAggregator.called_num),
            ('call_called_normalized_diff', _CustomAggregator.call_called_normalized_diff),
        ],
        'start_time_diff': [
            ('start_time_diff_mean', 'mean'), 
            ('start_time_diff_std', 'std'), 
            ('max', 'max'), 
            ('coefficient_of_variation', _CustomAggregator.coefficient_of_variation),
        ], 
        'phone1_loc_city_lat': [
            ('mean', 'mean'), 
            ('std', 'std'), 
            ('max', 'max'), 
            ('min', 'min'),
        ],
        'phone1_loc_city_lon': [
            ('mean', 'mean'), 
            ('std', 'std'), 
            ('max', 'max'), 
            ('min', 'min'),
        ],
        'phone2_loc_city_lat': [
            ('mean', 'mean'), 
            ('std', 'std'), 
            ('max', 'max'), 
            ('min', 'min'),
        ],
        'phone2_loc_city_lon': [
            ('mean', 'mean'), 
            ('std', 'std'), 
            ('max', 'max'), 
            ('min', 'min'),
        ],
        'start_day': [
            ('mean', 'mean'), 
            ('std', 'std'), 
            ('max', 'max'), 
            ('min', 'min'),
        ],
        'start_day_diff': [
            ('mean', 'mean'), 
            ('std', 'std'), 
            ('max', 'max'), 
            ('min', 'min'),
        ],
    }

    if max_workers == 0:
        max_workers = min(os.cpu_count(), len(agg_params))
    print(f'Using {max_workers} workers to aggregate features...')

    columns = [col for col in data.columns if col in agg_params.keys()]
    results = []

    unused_params = [i for i in agg_params.keys() if i not in columns]
    if unused_params:
        print(f"The following aggregate params are not found in the dataframe and will be ignored:\n{', '.join(unused_params)}")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_aggregate_features, data[['msisdn', col]], agg_params[col]): col for col in columns}
        for future in as_completed(futures):
            col = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f'{col} generated an exception: {exc}')

    # # 验证每个结果的 msisdn 是否一致
    # for i in range(1, len(results)):
    #     pd.testing.assert_series_equal(results[i]['msisdn'], results[i-1]['msisdn'])

    # 拼接结果
    final_result = pd.concat(results, axis=1)
    final_result = final_result.loc[:,~final_result.columns.duplicated()]

    return final_result

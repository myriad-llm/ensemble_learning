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

    # 通过 agg 使用

    @staticmethod
    def quantile(x, q):
        return x.quantile(q)

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
    def daily_avg_call_num(x):
        return x.dt.date.value_counts().mean()

    @staticmethod
    def monthly_avg_call_num(x):
        return x.dt.to_period('M').value_counts().mean()

    @staticmethod
    def night_call_count(x):
        return x[(x >= 21) | (x < 6)].count()

    @staticmethod
    def work_time_call_count(x):
        return x[(x >= 9) & (x < 18)].count()

    @staticmethod
    def other_time_call_count(x):
        return x[(x < 9) | (x >= 18)].count()
    
    @staticmethod
    def count_value(x, value):
        return (x == value).count()

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
    def coefficient_of_variation(x):
        return x.std() / x.mean()

class _CustomAggFunc:

    # 通过 apply 使用
    @staticmethod
    def count_recent_users(group):
        assert 'end_time' in group.columns and 'open_datetime' in group.columns
        return pd.Series({
            'count_recent_users': (group['end_time'].max() - group['open_datetime']).dt.days.mean()
        })

    @staticmethod
    def average_update_time_diff(group):
        assert 'update_time' in group.columns and 'start_time' in group.columns
        return pd.Series({
            'average_update_time_diff': (group['update_time'] - group['start_time']).dt.total_seconds().mean()
        })



def _aggregate_features(data, agg_params):
    """单列聚合函数

    Args:
        data (pd.DataFrame): 单列数据
        agg_params (dict): agg 参数

    Returns:
        pd.DataFrame: 聚合后的特征
    """
    agg_features = data.groupby('msisdn').agg(agg_params)
    agg_features.columns = ['+'.join(col).strip() for col in agg_features.columns.values]
    agg_features.columns = agg_features.columns.str.replace('[', '').str.replace(']', '').str.replace('<', '').str.replace('>', '').str.replace('(', '').str.replace(')', '').str.replace(',', '').str.replace(' ', '_')
    agg_features = agg_features.reset_index()
    return agg_features

def _apply_agg_features(data, custom_agg_func: callable):
    """无法通过agg实现的复杂或者多列聚合函数

    Args:
        data (pd.DataFrame): 数据
        custom_agg_func (function): 自定义聚合函数

    Returns:
        pd.DataFrame: 聚合后的特征
    """
    agg_features = data.groupby('msisdn').apply(custom_agg_func)
    agg_features.columns = agg_features.columns.str.replace('[', '').str.replace(']', '').str.replace('<', '').str.replace('>', '').str.replace('(', '').str.replace(')', '').str.replace(',', '').str.replace(' ', '_')
    agg_features = agg_features.reset_index()
    return agg_features

# 并行处理函数
def aggregate_features_parallel(data, max_workers=0):
    agg_params = {
        
        #时间特征开始
        #——————————————————————————#
    
        #通话时长统计：计算每个用户的通话时长均值、最大值和最小值，以捕捉异常的通话时间
         'call_duration': [
            ('sum', 'sum'), 
            ('mean', 'mean'), 
            ('max', 'max'), 
            ('std', 'std'),
            ('quantile_25', partial(_CustomAggregator.quantile, q=0.25)),
            ('quantile_50', partial(_CustomAggregator.quantile, q=0.5)),
            ('quantile_75', partial(_CustomAggregator.quantile, q=0.75)),
        ],
        
        # 通话频率：按天、月、周或月统计通话次数，以发现异常的高频通话模式。
    
        # 按周 统计工作日和非工作日 通话次数的平均值
      
        'dayofweek': [
            ('std', 'std'), 
            ('magic', _CustomAggregator.magic),
            ('work_day_num', _CustomAggregator.work_day_num),
            ('weekend_num', _CustomAggregator.weekend_num),
            ('mode', _CustomAggregator.mode),
            ('work_day_weekend_diff', _CustomAggregator.work_day_weekend_diff),
        ],
        
    
         'start_time': [
             # 按天统计通话次数的平均值
            ('daily_avg_call_num', _CustomAggregator.daily_avg_call_num),
             # 按月统计通话次数的平均值
            ('monthly_avg_call_num', _CustomAggregator.monthly_avg_call_num),
        ],

         'hour': [
            # 按夜间通话次数统计
            ('night_call_count', _CustomAggregator.night_call_count),
            # 按工作时间通话次数统计
            ('work_time_call_count', _CustomAggregator.work_time_call_count),
            # 按其他时间段通话次数统计
            ('other_time_call_count', _CustomAggregator.other_time_call_count),
    ],

        #时间特征结束
        #——————————————————————————#
        
        
        
        #用户行为多样性开始
        #——————————————————————————#
        
        # 统计每个用户与不同对端的通话数量
        'other_party': [
            ('account_person_num', 'nunique'),
            ('called_diff_home_code', _CustomAggregator.called_diff_home_code)
        ],
        
      
        'phone1_loc_city': [
            # 统计每个用户在不同城市的通话数量
            ('unique_city_count','nunique'),  # 计算每个用户在不同城市的通话数量
        ],

        'phone1_loc_province': [
        # 统计每个用户在不同省份的通话数量
        ('unique_province_count','nunique'),  # 计算每个用户在不同省份的通话数量
        ],
        
        #用户行为多样性结束
        #——————————————————————————#
        
        
        
        # 位置的特征开始
        #——————————————————————————#
        
          
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
        
        
        #不同地区代码的数量
        'home_area_code': [
            ('home_area_code_nunique', 'nunique')
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
        
        
        
        #统计用户的漫游行为（roam_type）
        'roam_type': [
                ('roam_type_nunique', 'nunique'),  # 不同漫游类型的数量
                ('roam_type_0_count',   partial(_CustomAggregator.count_value,value=0)),  # 非漫游次数
                ('roam_type_1_count',  partial(_CustomAggregator.count_value,value=1)),  # 省内漫游次数
                ('roam_type_4_count',  partial(_CustomAggregator.count_value,value=4)),  # 省际出访漫游次数
                ('roam_type_5_count',  partial(_CustomAggregator.count_value,value=5)),  # 国际漫游次数
                ('roam_type_6_count',  partial(_CustomAggregator.count_value,value=6)),  # 群内通话次数
                ('roam_type_7_count',  partial(_CustomAggregator.count_value,value=7)),  # 区域通话次数
                ('roam_type_8_count',  partial(_CustomAggregator.count_value,value=8)),  # 视频通话次数
                ('roam_type_9_count',  partial(_CustomAggregator.count_value,value=9)),  # 密话次数
            ],
        
        # 国内与国际通话
        'long_type1': [
            ('long_type1_0_count', partial(_CustomAggregator.count_value,value=0)),  # 本地、区内、区间、海事微星
            ('long_type1_1_count', partial(_CustomAggregator.count_value,value=1)),  # 省内通话次数
            ('long_type1_2_count', partial(_CustomAggregator.count_value,value=2)),  # 国内通话次数
            ('long_type1_3_count', partial(_CustomAggregator.count_value,value=3)),  # 香港、澳门、台湾、国际长途
        ],

        #位置的特征结束
        #——————————————————————————#
    
    
    
        # 通话类型和媒体特征
        #——————————————————————————#
 

        'a_serv_type': [
            # 统计主叫的数量
            ('call_src_count', partial(_CustomAggregator.count_value,value=1)),  # 主叫的数量
            # 统计被叫的数量
            ('call_dst_count', partial( _CustomAggregator.count_value,value=2)),  # 被叫的数量
            # 统计呼转的数量
            ('call_forward_count',partial( _CustomAggregator.count_value,value=3))  # 呼转的数量
        ],

        #费用相关特征
        #——————————————————————————#
        #通话费统计：对cfee和lfee进行统计，生成每个用户的通话费总额、均值和标准差，识别高消费用户。
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

        #费用相关特征
        #——————————————————————————#



    # 用户历史特征
    #——————————————————————————#
    # 开户时间：计算从open_datetime到当前时间的账户年龄，判断是否为新用户，诈骗行为可能更常出现在新用户中。
            
        'open_datetime': [
            ('open_count', 'nunique'),
            # 计算 end_time的最大值和 开户时间的
            
            # ('new_user_count',  partial(_CustomAggregator.count_recent_users,data=data)), 
            
            # 计算入库时间差的平均值  入库时间差：计算update_time与start_time的时间差均值，若数据更新频率异常，可能存在风险。
            # ('average_update_time_diff',partial( _CustomAggregator.average_update_time_diff,data=data))
        ],

            
    # 用户历史特征开始
    #——————————————————————————#
        'start_time_diff': [
            ('start_time_diff_mean', 'mean'), 
            ('start_time_diff_std', 'std'), 
            ('max', 'max'), 
            ('coefficient_of_variation', _CustomAggregator.coefficient_of_variation),
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
    
    # 用户历史特征结束
    #——————————————————————————#

        
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
        # 简单聚合
        futures_agg = {executor.submit(_aggregate_features, data[['msisdn', col]], agg_params[col]): col for col in columns}
        # 通过 apply 实现的复杂聚合
        futures_apply_agg = {
            executor.submit(_apply_agg_features, data[['msisdn', 'end_time', 'open_datetime']], _CustomAggFunc.count_recent_users): 'count_recent_users',
            executor.submit(_apply_agg_features, data[['msisdn', 'update_time', 'start_time']], _CustomAggFunc.average_update_time_diff): 'average_update_time_diff'
        }

        futures = {**futures_agg, **futures_apply_agg}
        for future in as_completed(futures):
            col = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f'{col} generated an exception: {exc}')

    # 验证每个结果的 msisdn 是否一致
    for i in range(1, len(results)):
        pd.testing.assert_series_equal(results[i]['msisdn'], results[i-1]['msisdn'])

    # 拼接结果
    final_result = pd.concat(results, axis=1)
    final_result = final_result.loc[:,~final_result.columns.duplicated()]

    return final_result

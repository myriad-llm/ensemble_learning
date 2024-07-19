# %%
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix,classification_report
from xgboost import XGBClassifier
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer

ALL = True
NO_SMOTE = False

# 判断 processed 文件夹是否存在
import os
if not os.path.exists('../data/processed'):
    print("Creating processed data folder...")
    # 读取CSV文件
    train_data = pd.read_csv('../data/raw/trainSet_res_with_distances.csv', dtype={'msisdn': 'str'})
    train_labels = pd.read_csv('../data/raw/trainSet_ans.csv', dtype={'msisdn': 'str'})

    validation_data = pd.read_csv('../data/raw/validationSet_res_with_distances.csv', dtype={'msisdn': 'str'})


    # 遍历 groupby('msisdn') 的结果，对每个 msisdn 进行数据增强
    # ------
    from tqdm import tqdm
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname('./'), '../'))
    from utils.augmentation import Augmentation

    addition_train_data = []
    addition_train_labels = []

    times = 4
    ratio_range = 0.1
    pbar = tqdm(train_data.groupby('msisdn'))
    for msisdn, group in pbar:
        if msisdn == 0:
            continue
        # print(f"Augmenting msisdn {msisdn}")
        pbar.set_description(f"Augmenting msisdn {msisdn}")
        label = train_labels[train_labels['msisdn'] == msisdn].iloc[0]['is_sa']
        aug = Augmentation(group, label, 'msisdn', 'is_sa')
        # 对正负样本进行平衡 样本比 1:4
        if label == 1:
            res_df, res_labels = aug.times(ratio=ratio_range, times=times * 4, method='mask')
        else:
            res_df, res_labels = aug.times(ratio=ratio_range, times=times, method='mask')
        addition_train_data.append(res_df)
        addition_train_labels.append(res_labels)
    addition_train_data = pd.concat(addition_train_data)
    addition_train_labels = pd.concat(addition_train_labels)

    # 将新数据加入到train_data中
    train_data = pd.concat([train_data, addition_train_data], ignore_index=True).reset_index(drop=True)
    train_labels = pd.concat([train_labels, addition_train_labels], ignore_index=True).reset_index(drop=True)
    # ------------------

    # save
    print("Saving processed data...")
    os.makedirs('../data/processed', exist_ok=True)
    train_data.to_csv('../data/processed/train_data.csv', index=False)
    train_labels.to_csv('../data/processed/train_labels.csv', index=False)
    validation_data.to_csv('../data/processed/validation_data.csv', index=False)

else:
    print("Reading processed data...")
    train_data = pd.read_csv('../data/processed/train_data.csv', dtype={'msisdn': 'str'})
    train_labels = pd.read_csv('../data/processed/train_labels.csv', dtype={'msisdn': 'str'})

    # 读取验证集
    validation_data = pd.read_csv('../data/processed/validation_data.csv', dtype={'msisdn': 'str'})


# 转换时间格式
train_data['start_time'] = pd.to_datetime(train_data['start_time'], format='%Y%m%d%H%M%S')
train_data['end_time'] = pd.to_datetime(train_data['end_time'], format='%Y%m%d%H%M%S')
train_data['open_datetime'] = pd.to_datetime(train_data['open_datetime'], format='%Y%m%d%H%M%S')
train_data['update_time'] = pd.to_datetime(train_data['update_time'])
train_data['date'] = pd.to_datetime(train_data['date'])

validation_data['start_time'] = pd.to_datetime(validation_data['start_time'], format='%Y%m%d%H%M%S')
validation_data['end_time'] = pd.to_datetime(validation_data['end_time'], format='%Y%m%d%H%M%S')
validation_data['open_datetime'] = pd.to_datetime(validation_data['open_datetime'], format='%Y%m%d%H%M%S',errors='coerce')
validation_data['update_time'] = pd.to_datetime(validation_data['update_time'])
validation_data['date'] = pd.to_datetime(validation_data['date'])

# %%
len(train_data), len(validation_data)

# %%
# 为每条记录添加start_time_diff，记录 start_time 与上一条记录的 start_time 之差 (单位：秒)
start_time_diff = train_data.groupby('msisdn')['start_time'].diff().dt.total_seconds().fillna(0).reset_index(drop=True)
# 将该列加入到数据集中
train_data['start_time_diff'] = start_time_diff.copy()
# time_diff_start2end = train_data.groupby('msisdn')['end_time'].diff().dt.total_seconds().fillna(0)
start_time_diff = validation_data.groupby('msisdn')['start_time'].diff().dt.total_seconds().fillna(0).reset_index(drop=True)
validation_data['start_time_diff'] = start_time_diff.copy()

# %%
train_labels

# %% [markdown]
# 数据特征处理

# %%
# 聚合特征
def aggregate_features(data):
    return data.groupby('msisdn').agg({
    'call_duration': [
        # ('call_duration_sum', 'sum'), 
        ('call_duration_mean', 'mean'), 
        ('call_duration_max', 'max'), 
        ('call_duration_std', 'std'),
        ('call_duration_quantile_25', lambda x: x.quantile(0.25)), 
        ('call_duration_quantile_50', lambda x: x.quantile(0.50)), 
        ('call_duration_quantile_75', lambda x: x.quantile(0.75))
    ],
    'cfee': [
        # ('cfee_sum', 'sum'),
        ('cfee_std', 'std'), 
        ('cfee_mean', 'mean'),
    ],
    'lfee': [
        # ('lfee_sum', 'sum'), 
        ('lfee_mean', 'mean'),
        ('lfee_std', 'std'),
    ],
    'hour': [
        ('hour_mean', 'mean'), 
        ('hour_std', 'std'), 
        # ('hour_max', 'max'), 
        ('hour_min', 'min'),
    ],
    'dayofweek': [
        ('dayofweek_std', 'std'), 
        ('magic_dayofweek', lambda x: x.value_counts().mean()), 
        # ('work_day_num', lambda x: x[x.isin([1,2,3,4,5])].count()), 
        # ('weekend_num', lambda x: x[x.isin([6,7])].count()),
        ('dayofweek_mode', lambda x: x.mode().values[0]),
        ('work_day_weekend_diff', lambda x: (x[x.isin([1,2,3,4,5])].count() - x[x.isin([6,7])].count()) / (x[x.isin([1,2,3,4,5])].count() + x[x.isin([6,7])].count())),
    ],
    # 'home_area_code': [
    #     ('home_area_code_nunique', 'nunique')
    # ],
    'visit_area_code': [
        ('visit_area_code_nunique', 'nunique'),
        ('times_not_at_home_area', lambda x: x[x != x.shift()].count())
    ],
    'called_home_code': [
        ('called_home_code_nunique', 'nunique'),
        ('called_diff_home_code', lambda x: x[x != x.shift()].count())
    ],
    'called_code': [
        # ('called_code_nunique', 'nunique')
        ('called_code_diff', lambda x: x[x != x.shift()].count())
    ],
    'open_datetime': [
        ('open_count', 'nunique')
    ],
    'other_party': [
        ('account_person_num', 'nunique'),
        ('called_diff_home_code', lambda x: x[x != x.shift()].count())
    ],
    'a_serv_type': [
        # ('call_num', lambda x: x[x.isin([1, 3])].count()), 
        # ('called_num', lambda x: x[x == 2].count()),
        ('call_called_normalized_diff', lambda x: (x[x.isin([1, 3])].count() - x[x == 2].count()) /  (x[x.isin([1, 3])].count() + x[x == 2].count())),
    ],
    'start_time_diff': [
        ('start_time_diff_mean', 'mean'), 
        ('start_time_diff_std', 'std'), 
        ('start_time_diff_max', 'max'), 
    ], 
    'distance': [
        # ('distance_sum', 'sum'), 
        ('distance_std', 'std'), 
        # ('distance_max', 'max'), 
        # ('distance_quantile_25', lambda x: x.quantile(0.25)), 
        ('distance_quantile_50', lambda x: x.quantile(0.50)), 
        ('distance_quantile_75', lambda x: x.quantile(0.75)),
    ]
})

train_features = aggregate_features(train_data)
validation_features = aggregate_features(validation_data)


train_features.columns = ['_'.join(col).strip() for col in train_features.columns.values]
validation_features.columns = ['_'.join(col).strip() for col in validation_features.columns.values]
train_features.columns = train_features.columns.str.replace('[', '').str.replace(']', '').str.replace('<', '').str.replace('>', '').str.replace('(', '').str.replace(')', '').str.replace(',', '').str.replace(' ', '_')
validation_features.columns = validation_features.columns.str.replace('[', '').str.replace(']', '').str.replace('<', '').str.replace('>', '').str.replace('(', '').str.replace(')', '').str.replace(',', '').str.replace(' ', '_')
# 重置索引
train_features = train_features.reset_index()
validation_features = validation_features.reset_index()

# 合并标签数据
train_features = train_features.merge(train_labels, on='msisdn', how='left')
# 打印结果
# train_features

# %%
y = train_features['is_sa']
X = train_features.drop(['msisdn', 'is_sa'], axis=1)
X_validation = validation_features.drop(['msisdn'], axis=1)

n_sample = y.shape[0]
n_pos_sample = y[y ==1].shape[0]
n_neg_sample = y[y == 0].shape[0]
print('样本个数：{}; 正样本占{:.2%}; 负样本占{:.2%}'.format(n_sample,
                                                   n_pos_sample / n_sample,
                                                   n_neg_sample / n_sample))
print('特征维数：', X.shape[1])

# %%
# TODO use all_X to impute
imputer = SimpleImputer(strategy='most_frequent')
X = imputer.fit_transform(X)

# %%
imputer2 = SimpleImputer(strategy='most_frequent')
X_validation = imputer2.fit_transform(X_validation)

# %%
if ALL:
    if not NO_SMOTE:
        smote = SMOTE(random_state=42)    # 处理过采样的方法
        X, y = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
    train_len = len(y_train) + len(y_test)
    test_len = 0
else:
    X_train,X_test,y_train,y_test = train_test_split(X,y,stratify = y,test_size= 0.3,random_state=42)
    # X_test_imputed = imputer.transform(X)

    if not NO_SMOTE:
        smote = SMOTE(random_state=42)    # 处理过采样的方法
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print('通过SMOTE方法平衡正负样本后')
        n_sample = y_train.shape[0]
        n_pos_sample = y_train[y_train == 1].shape[0]
        n_neg_sample = y_train[y_train == 0].shape[0]
        print('样本个数：{}; 正样本占{:.2%}; 负样本占{:.2%}'.format(n_sample,
                                                        n_pos_sample / n_sample,
                                                        n_neg_sample / n_sample))
        print('特征维数：', X.shape[1])
    train_len, test_len = len(y_train), len(y_test)

# %%
X_test.shape

# %%
print(y_test.value_counts())

# %%
X.shape

# %%
columns = train_features.columns.tolist()
columns.remove('msisdn')

# %%
assert X_validation.shape[1] == X_train.shape[1]

# %%
# 拼接 X_train 和 y_train np.array 为 dataframe
train_set = np.c_[X_train, y_train]
train_set = pd.DataFrame(train_set, columns=columns)
test_set = np.c_[X_test, y_test]
test_set = pd.DataFrame(test_set, columns=columns)
valid_set = np.c_[X_validation, np.zeros(X_validation.shape[0])]
valid_set = pd.DataFrame(valid_set, columns=columns)
valid_set['is_sa'] = -1

# %%
test_set.describe()

# %%
all_set = pd.concat([train_set, test_set, valid_set], axis=0).reset_index(drop=True)
labeled_data_len = train_set.shape[0] + test_set.shape[0]

# %%
test_set.shape, train_set.shape, valid_set.shape, all_set.shape

# %%
labeled_set, valid_set = all_set.iloc[:labeled_data_len].copy(), all_set.iloc[labeled_data_len:].copy()
labeled_set.reset_index(drop=True, inplace=True)
valid_set.reset_index(drop=True, inplace=True)
# 有一些值在SMOTE后对数变换后为 NaN，需要删除这些数据
print(labeled_set.isnull().sum().sum())
labeled_set = labeled_set.dropna()
print(labeled_set.isnull().sum().sum())
assert valid_set.shape[0] == validation_features.shape[0]

# %%
# 重新划分训练集和测试集
if not ALL:
    train_set, test_set = labeled_set.iloc[:train_len].copy(), labeled_set.iloc[train_len:].copy()
    train_set.reset_index(drop=True, inplace=True)
    test_set.reset_index(drop=True, inplace=True)
    # # 从 test_set 和 y_test 中删除 数据增强获得的数据 无法实现
    # assert y.shape[0] == train_features.shape[0]
    # assert labeled_set.shape[0] == y.shape[0]
    # id_col = train_features['msisdn']
    # train_set_ids = id_col.iloc[train_set.index]
    # test_set_ids = id_col.iloc[test_set.index]

# %%
# 使用 autogluon 训练
from autogluon.tabular import TabularPredictor
# 输入数据X_train, y_train
if not ALL:
    model = TabularPredictor(label='is_sa', eval_metric='f1', problem_type='binary').fit(train_set, presets='medium_quality')
    # model = TabularPredictor(label='is_sa', eval_metric='f1', problem_type='binary').fit(train_set, presets='best_quality', time_limit=3600)
else:
    model = TabularPredictor(label='is_sa', eval_metric='f1', problem_type='binary').fit(labeled_set, presets='best_quality', time_limit=3600)

# %%
if not ALL:
    print(model.evaluate(test_set))

# %%
feature_importance = model.feature_importance(test_set if not ALL else labeled_set)
print(feature_importance)
feature_importance

# %%
# leaderboard
if not ALL:
    leaderboard = model.leaderboard(test_set, silent=True)
    print(leaderboard)
else:
    leaderboard = model.leaderboard(labeled_set, silent=True)
    print(leaderboard)
leaderboard

# %%
# 在testset 上计算指标
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

if not ALL:
    y_pred = model.predict(test_set)
    y_true = test_set['is_sa']
    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))

# %%
valid_set

# %%
# 预测
y_validation_pred = model.predict(valid_set.drop('is_sa', axis=1))

# 将预测结果与 msisdn 对应起来
validation_results = validation_features[['msisdn']].copy()
validation_results['is_sa'] = y_validation_pred.astype(int)

print(validation_results.describe())

# 保存结果到CSV文件
file_name = './valid_large_data.csv' if ALL else './valid_small_data.csv'
validation_results.to_csv(file_name, index=False)
print(file_name)



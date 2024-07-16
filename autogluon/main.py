# %%
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix,classification_report
from xgboost import XGBClassifier
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer



# 读取CSV文件
train_data = pd.read_csv('../data/raw/trainSet_res.csv')
train_labels = pd.read_csv('../data/raw/trainSet_ans.csv')

# 读取验证集
validation_data = pd.read_csv('../data/raw/validationSet_res.csv')

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
train_data.shape

# %%
# 为每条记录添加start_time_diff，记录 start_time 与上一条记录的 start_time 之差 (单位：秒)
start_time_diff = train_data.groupby('msisdn')['start_time'].diff().dt.total_seconds().fillna(0).reset_index(drop=True)
# 将该列加入到数据集中
train_data['start_time_diff'] = start_time_diff.copy()
# time_diff_start2end = train_data.groupby('msisdn')['end_time'].diff().dt.total_seconds().fillna(0)
start_time_diff = validation_data.groupby('msisdn')['start_time'].diff().dt.total_seconds().fillna(0).reset_index(drop=True)
validation_data['start_time_diff'] = start_time_diff.copy()

# %%
train_data.groupby('msisdn')['home_area_code'].nunique()

# %% [markdown]
# 数据特征处理

# %%
# 聚合特征
def aggregate_features(data):
    return data.groupby('msisdn').agg({
    'call_duration': [
        ('call_duration_sum', 'sum'), 
        ('call_duration_mean', 'mean'), 
        ('call_duration_max', 'max'), 
        ('call_duration_quantile_25', lambda x: x.quantile(0.25)), 
        ('call_duration_quantile_50', lambda x: x.quantile(0.50)), 
        ('call_duration_quantile_75', lambda x: x.quantile(0.75))
    ],
    'cfee': [
        ('cfee_sum', 'sum'), 
        ('cfee_mean', 'mean')
    ],
    'lfee': [
        ('lfee_sum', 'sum'), 
        ('lfee_mean', 'mean')
    ],
    'hour': [
        ('hour_mean', 'mean'), 
        ('hour_std', 'std'), 
        ('hour_max', 'max'), 
        ('hour_min', 'min')
    ],
    'dayofweek': [
        ('dayofweek_std', 'std'), 
        ('magic_dayofweek', lambda x: x.value_counts().mean()), 
        ('work_day_num', lambda x: x[x.isin([1,2,3,4,5])].count()), 
        ('weekend_num', lambda x: x[x.isin([6,7])].count())
    ],
    # 'home_area_code': [
    #     ('home_area_code_nunique', 'nunique')
    # ],
    'visit_area_code': [
        ('visit_area_code_nunique', 'nunique')
    ],
    'called_home_code': [
        ('called_home_code_nunique', 'nunique')
    ],
    'called_code': [
        ('called_code_nunique', 'nunique')
    ],
    'open_datetime': [
        ('open_count', 'nunique')
    ],
    'other_party': [
        ('account_person_num', 'nunique')
    ],
    'a_serv_type': [
        ('call_num', lambda x: x[x.isin([1, 3])].count()), 
        ('called_num', lambda x: x[x == 2].count())
    ],
    'start_time_diff': [
        ('start_time_diff_mean', 'mean'), 
        ('start_time_diff_std', 'std'), 
        ('start_time_diff_max', 'max'), 
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
train_features

# %%
y = train_features['is_sa']
X = train_features.drop(['msisdn', 'is_sa'], axis=1)

n_sample = y.shape[0]
n_pos_sample = y[y ==1].shape[0]
n_neg_sample = y[y == 0].shape[0]
print('样本个数：{}; 正样本占{:.2%}; 负样本占{:.2%}'.format(n_sample,
                                                   n_pos_sample / n_sample,
                                                   n_neg_sample / n_sample))
print('特征维数：', X.shape[1])


# %%
X.head()

# %%

imputer = SimpleImputer(strategy='most_frequent')
X= imputer.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y,stratify = y,test_size= 0.3,random_state=42)

# X_test_imputed = imputer.transform(X)

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


# %%
train_features.shape
columns = train_features.columns.tolist()
columns.remove('msisdn')

# %%
# 拼接 X_train 和 y_train np.array 为 dataframe
train_set = np.c_[X_train, y_train]
train_set = pd.DataFrame(train_set, columns=columns)
test_set = np.c_[X_test, y_test]
test_set = pd.DataFrame(test_set, columns=columns)
all_set = pd.concat([train_set, test_set], axis=0).reset_index(drop=True)

# %%
all_set.max()

# %%


# %%
# 使用 autogluon 训练
from autogluon.tabular import TabularPredictor
# 输入数据X_train, y_train
model = TabularPredictor(label='is_sa', eval_metric='f1', problem_type='binary').fit(all_set, num_bag_folds=5, num_bag_sets=1, num_stack_levels=1)

# %%
print(model.evaluate(all_set))

# %%
model.feature_importance(all_set)

# %%
# leaderboard
leaderboard = model.leaderboard(all_set, silent=True)
print(leaderboard)
leaderboard

# %%
# 在testset 上计算指标
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

y_pred = model.predict(all_set)
y_true = all_set['is_sa']
print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))

# %%
X_validation = validation_features.drop(['msisdn'], axis=1)

# %%
# 预测
y_validation_pred = model.predict(X_validation)

# 将预测结果与 msisdn 对应起来
validation_results = validation_features[['msisdn']].copy()
validation_results['is_sa'] = y_validation_pred.astype(int)

print(validation_results.describe())

# 保存结果到CSV文件
validation_results.to_csv('./vaild_large_data.csv', index=False)
print("Validation predictions saved to validation_predictions.csv")



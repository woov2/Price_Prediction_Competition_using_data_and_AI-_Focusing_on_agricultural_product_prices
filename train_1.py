import pandas as pd
import numpy as np
import pickle
import joblib
import os

from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import RANSACRegressor
from xgboost import XGBRegressor
from preprocessing_1 import create_sliding_block_dataset, create_sliding_block_dataset_version2, process_garak_domae, process_national_domae, process_sanji_gongpanjang, process_jungdomae, process_retail, month_sequence_to_sin_cos
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
## Processing Data Load
## 양파
train_양파 = pd.read_csv("preprocessing_train/train_양파.csv")

## 깐마늘
train_깐마늘 = pd.read_csv("preprocessing_train/train_깐마늘.csv")

## 배추
train_배추 = pd.read_csv("preprocessing_train/train_배추.csv")

## 사과
train_사과 = pd.read_csv("preprocessing_train/train_사과.csv")
## Sliding Window

os.makedirs('models', exist_ok=True)
os.makedirs('scalers', exist_ok=True)

###################################### 양파 ###############################################

df = train_양파[['전달 평균가격(원)', '전년 평균가격(원)', 
            '평년 평균가격(원)', '평균가격(원)']]

input_size = 9
output_size = 3
block_size = 180

X, y = create_sliding_block_dataset_version2(df, input_size, output_size, block_size)
X = pd.DataFrame([x.flatten() for x in X])

temp = []
for i in range(10):
    temp.append(pd.concat([train_양파[180*i:180*(i+1)][[
                                                   '품목명', '품종명', '거래단위', '등급', 
                                                   'soon',
                                                   'year', 'month', 
                                                   'sin_month', 'cos_month',
                                                   'is_peak', 'is_bottom']][:169].reset_index(drop=True), 
                              X[169*i:169*(i+1)].reset_index(drop=True)], axis=1))

X = pd.concat(temp, axis=0)
X = X.reset_index(drop=True)
X_양파 = X.copy()
y_양파 = y.copy()

##################################### 깐마늘 #############################################

df = train_깐마늘[['평균가격(원)']]

input_size = 9
output_size = 3
block_size = 180

X, y = create_sliding_block_dataset_version2(df, input_size, output_size, block_size)
X = pd.DataFrame([x.flatten() for x in X])

temp = []
for i in range(10):
    temp.append(pd.concat([train_깐마늘[180*i:180*(i+1)][['품목명', '품종명', '거래단위', '등급', 'soon', '전순 평균가격(원)', '전달 평균가격(원)', '전년 평균가격(원)',
                                                   '순 평균상대습도', '순 평균기온', '순 평균풍속', '순 최고기온', '순 최저상대습도', 
            '순 최저기온', '순 강수량', '순 누적 일조시간', '특보 발효 순통계 개수', 
                                                   'year', 'month', 
                                                   'sin_month', 'cos_month', 
                                                   'sin_date', 'cos_date'
                                                  ]][:169].reset_index(drop=True), 
                              X[169*i:169*(i+1)].reset_index(drop=True)], axis=1))

X = pd.concat(temp, axis=0)
X = X.reset_index(drop=True)

for i in range(9):  
    for j in range(i+1, 9): 
        X[f'Diff_{i}_{j}'] = X[i] - X[j]
        
X_깐마늘 = X.copy()
y_깐마늘 = y.copy()

###################################### 배추 ###############################################

df = train_배추['평균가격(원)']

input_size = 9
output_size = 3
block_size = 180

X, y = create_sliding_block_dataset(df, input_size, output_size, block_size)
X = pd.DataFrame([x.flatten() for x in X])

temp = []
for i in range(10):
    temp.append(pd.concat([train_배추[180*i:180*(i+1)][['품목명','품종명','거래단위','등급','전순 평균가격(원)','전달 평균가격(원)','전년 평균가격(원)','평년 평균가격(원)',
                                                   'year', 'month', 'soon','sin_month','cos_month',
                                                 '가락_평균가격', '가락_전순가격', '가락_전달가격',
       '가락_전년가격', '가락_평년가격', '전국_총반입량', '전국_총거래금액', '전국_평균가', '전국_고가',
       '중가(60%) 평균가 ', '전국_저가', '전국_중간가', '전국_최저가', '전국_최고가', '전국_경매건수',
       '전국_전순가격', '전국_전달가격', '전국_전년가격', '전국_평년가격', 
      '산지_총반입량', '산지_총거래금액',
       '산지_평균가', '산지_중간가', '산지_최저가', '산지_최고가', '산지_경매건수', '산지_전순가격', '산지_전달가격',
       '산지_전년가격', '산지_평년가격', '중도매_평균가격(원)', '중도매_전순가격', '중도매_전달가격', '중도매_전년가격','중도매_평년가격',
       '소매_평균가격(원)', '소매_전순가격', '소매_전달가격', '소매_전년가격', '소매_평년가격',
       'tr_val_1','tr_val_2','tr_val_3','tr_val_4','tr_val_5','tr_val_6','tr_val_7','tr_val_8','tr_val_9','tr_val_10'
                                              
      ]][:169].reset_index(drop=True),X[169*i:169*(i+1)].reset_index(drop=True)],axis=1))

X = pd.concat(temp,axis = 0)
X = X.reset_index(drop=True)

for i in range(9):
    for j in range(i+1, 9): 
        X[f'Diff_{i}_{j}'] = X[i] - X[j]

# 가격 변화율 (Price Change Rate)
X['가락_전순가격_변화율'] = (X['가락_평균가격'] - X['가락_전순가격']) / X['가락_전순가격']
X['가락_전달가격_변화율'] = (X['가락_평균가격'] - X['가락_전달가격']) / X['가락_전달가격']
X['가락_전년가격_변화율'] = (X['가락_평균가격'] - X['가락_전년가격']) / X['가락_전년가격']
X['가락_평년가격_변화율'] = (X['가락_평균가격'] - X['가락_평년가격']) / X['가락_평년가격']

# 전국 평균 가격 대비 가락 평균 가격 비율
X['전국대비_가락_평균가격_비율'] = X['가락_평균가격'] / X['전국_평균가']

# 전국 총 반입량 대비 산지 총 반입량 비율
X['전국대비_산지_반입량_비율'] = X['산지_총반입량'] / X['전국_총반입량']

# 전국 총 거래금액 대비 산지 총 거래금액 비율
X['전국대비_산지_거래금액_비율'] = X['산지_총거래금액'] / X['전국_총거래금액']

# 중도매 및 소매 가격 비교
X['중도매_소매_평균가격_비율'] = X['중도매_평균가격(원)'] / X['소매_평균가격(원)']

# 산지와 전국 최저가 비교
X['산지_전국_최저가_비율'] = X['산지_최저가'] / X['전국_최저가']

# 전국 경매건수 대비 산지 경매건수 비율
X['전국대비_산지_경매건수_비율'] = X['산지_경매건수'] / X['전국_경매건수']

# 전국 대비 중도매 전순가격 비율
X['전국대비_중도매_전순가격_비율'] = X['중도매_전순가격'] / X['전국_전순가격']

# 소매와 중도매 전달가격 비교
X['소매_중도매_전달가격_비율'] = X['소매_전달가격'] / X['중도매_전달가격']

# 산지 중간가 대비 전국 중간가 비율
X['산지_전국_중간가_비율'] = X['산지_중간가'] / X['전국_중간가']

X = X.replace([float('inf'), float('-inf')], 0)
X.fillna(0, inplace=True) 

X_배추 = X.copy()
y_배추 = y.copy()

###################################### 사과 ###############################################
df = train_사과['평균가격(원)']

input_size = 9
output_size = 3
block_size = 180

X, y = create_sliding_block_dataset(df, input_size, output_size, block_size)
X = pd.DataFrame([x.flatten() for x in X])

temp = []
for i in range(10):
    temp.append(pd.concat([train_사과[180*i:180*(i+1)][['품목명','품종명','거래단위','등급','전순 평균가격(원)','전달 평균가격(원)','전년 평균가격(원)','평년 평균가격(원)',
                                                  '순 평균상대습도', '순 평균기온', '순 평균풍속', '순 최고기온', '순 최저상대습도', '순 최저기온', '순 강수량',
                                                  '순 누적 일조시간', '특보 발효 순통계 개수', 
                                                   'year', 'month', 'soon','sin_month','cos_month',
                                                   '전국_총반입량', '전국_총거래금액', '전국_평균가', '전국_고가',
       '중가(60%) 평균가 ', '전국_저가', '전국_중간가', '전국_최저가', '전국_최고가', '전국_경매건수',
       '전국_전순가격', '전국_전달가격', '전국_전년가격', '전국_평년가격'                                             
      ]][:169].reset_index(drop=True),X[169*i:169*(i+1)].reset_index(drop=True)],axis=1))

X = pd.concat(temp,axis = 0)
X = X.reset_index(drop=True)
X_사과 = X.copy()
y_사과 = y.copy()
# 양파 -----------------------------------------------------------
md = {'양파':2}
품종명 = {'양파':['양파']}
거래단위 = {'양파':'1키로'}

k, v = "양파", 2

data = X_양파[(X_양파['품종명'].isin(품종명[k])) & (X_양파['거래단위']==거래단위[k])].reset_index(drop=True)
value = y_양파[X_양파[(X_양파['품종명'].isin(품종명[k])) & (X_양파['거래단위']==거래단위[k])].index]

data.drop(columns = ['품목명','품종명','거래단위','등급'], inplace=True)

le_dict = {}
for col in ['soon']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    le_dict[col] = le
    
with open('encoding/양파_labelencoder.pkl', 'wb') as f:
    pickle.dump(le_dict, f)

data.columns = [str(i) for i in data.columns]

tr_value = value.copy()

model = MultiOutputRegressor(XGBRegressor(max_depth = 3,
                                        learning_rate = 0.03353234595184532,
                                        n_estimators = 436,
                                        colsample_bytree = 0.6238713915347277,
                                        subsample = 0.25325234558787907,
                                        min_child_weight = 4,
                                        reg_alpha = 0.02289191045218996,
                                        reg_lambda = 0.016152136474997993,
                                        seed = 42,
                                        tree_method = "hist"))
model.fit(data, tr_value)
joblib.dump(model, 'models/양파_model.pkl')

# 깐마늘 -----------------------------------------------------------
md = {'깐마늘(국산)':6}
품종명 = {'깐마늘(국산)':['깐마늘(국산)']}
거래단위 = {'깐마늘(국산)':'20 kg'}

k, v = "깐마늘(국산)", 6

data = X_깐마늘[(X_깐마늘['품종명'].isin(품종명[k])) & (X_깐마늘['거래단위']==거래단위[k])].reset_index(drop=True)
value = y_깐마늘[X_깐마늘[(X_깐마늘['품종명'].isin(품종명[k])) & (X_깐마늘['거래단위']==거래단위[k])].index]

scaler = StandardScaler()
value = scaler.fit_transform(value)

with open('scalers/깐마늘_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

le_dict = {}
for col in ['품목명','품종명','거래단위','등급', 'soon']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    le_dict[col] = le
    
with open('encoding/깐마늘_labelencoder.pkl', 'wb') as f:
    pickle.dump(le_dict, f)

data.columns = [str(i) for i in data.columns]

tr_value = value.copy()

state = 42

model = MultiOutputRegressor(XGBRegressor(max_depth = 11,
                                        learning_rate = 0.013399187582766156,
                                        n_estimators = 333,
                                        colsample_bytree = 0.7094139999441733,
                                        subsample = 0.273649976479683,
                                        min_child_weight = 7,
                                        reg_alpha = 0.0016260190378359343,
                                        reg_lambda = 0.0011277688891604676,
                                        seed = 42,
                                        tree_method = "hist"))
model.fit(data, tr_value)
joblib.dump(model, 'models/깐마늘_model.pkl')

# 배추 -----------------------------------------------------------
md = {'배추':0}
품종명 = {'배추':['배추']}
거래단위 = {'배추':'10키로망대'}

k, v = "배추", 0

data = X_배추[(X_배추['품종명'].isin(품종명[k])) & (X_배추['거래단위']==거래단위[k])].reset_index(drop=True)
value = y_배추[X_배추[(X_배추['품종명'].isin(품종명[k])) & (X_배추['거래단위']==거래단위[k])].index]

scaler = MinMaxScaler()
value = scaler.fit_transform(value)

with open('scalers/배추_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

le_dict = {}
for col in ['품목명','품종명','거래단위','등급', 'soon']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    le_dict[col] = le

with open('encoding/배추_labelencoder.pkl', 'wb') as f:
    pickle.dump(le_dict, f)

data.columns = [str(i) for i in data.columns]

tr_value = value.copy()

model = MultiOutputRegressor(XGBRegressor(max_depth = 34,
                                        learning_rate = 0.001636356384562596,
                                        n_estimators = 1528,
                                        colsample_bytree = 0.5541270480824472,
                                        subsample = 0.9998185855299385,
                                        min_child_weight = 5,
                                        reg_alpha = 0.032431438057066254,
                                        reg_lambda = 0.06164686170841902,
                                        seed = 42,
                                        tree_method = "exact"))
model.fit(data, tr_value)
joblib.dump(model, 'models/배추_model_xgb.pkl')

model = MultiOutputRegressor(ExtraTreesRegressor(random_state=42))
model.fit(data, tr_value)
joblib.dump(model, 'models/배추_model_et.pkl')

# 사과 -----------------------------------------------------------
md =  {'사과':8}
품종명 = {'사과':['홍로','후지']}
거래단위 = {'사과':'10 개'}

k, v = "사과", 8

data = X_사과[(X_사과['품종명'].isin(품종명[k])) & (X_사과['거래단위']==거래단위[k])].reset_index(drop=True)
value = y_사과[X_사과[(X_사과['품종명'].isin(품종명[k])) & (X_사과['거래단위']==거래단위[k])].index]

scaler = RobustScaler()
value = scaler.fit_transform(value)

with open('scalers/사과_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

le_dict = {}
for col in ['품목명','품종명','거래단위','등급', 'soon']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    le_dict[col] = le

with open('encoding/사과_labelencoder.pkl', 'wb') as f:
    pickle.dump(le_dict, f)

data.columns = [str(i) for i in data.columns]

tr_value = value.copy()
##사과 parameter
#######################################
params = {
    'max_depth': 25,
    'learning_rate': 0.041234179899240574,
    'n_estimators': 1754,
    'colsample_bytree': 0.9489991959152967,
    'subsample': 0.5107124780769203,
    'min_child_weight': 9,
    'reg_alpha': 0.03484221600894738,
    'reg_lambda': 0.020121209871364277
}
params['seed'] = 42
params['tree_method'] = "exact"
########################################
n_split_list = [5]  
state = 42
for split in n_split_list:
    cv = KFold(n_splits=split, shuffle=True, random_state=state)
    fold_idx = 1
    for train_index, valid_index in cv.split(data, tr_value):
        X_train, X_valid = data.iloc[train_index], data.iloc[valid_index]
        Y_train, Y_valid = tr_value[train_index], tr_value[valid_index]

        model = MultiOutputRegressor(XGBRegressor(**params))

        model.fit(X_train, Y_train)
        joblib.dump(model, f'models/사과_model_xgb_{fold_idx}.pkl')

        model = MultiOutputRegressor(ExtraTreesRegressor(random_state=state))
        model.fit(X_train,Y_train)
        joblib.dump(model, f'models/사과_model_et_{fold_idx}.pkl')

        fold_idx += 1

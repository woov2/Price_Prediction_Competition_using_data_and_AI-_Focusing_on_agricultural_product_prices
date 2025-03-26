import pandas as pd
import pickle
import numpy as np

from preprocessing_2 import create_sliding_block_dataset, external_data

submit = pd.read_csv('./submission_최종/inference1_submission.csv')
기상 = external_data('기상')
강수 = external_data('강수')

################################# test1 ####################################

test = pd.DataFrame()
for no in range(52):
   test = pd.concat([test, pd.read_csv(f'./data/test/TEST_{no:02}_1.csv')],axis=0)

test = test.rename(columns = {'품목(품종)명':'품목명'})

################################## 무 #######################################

test_무 = test[test.품목명 == '무'].reset_index(drop = True)
test_무['year'] = test_무.YYYYMMSOON.apply(lambda x:int(x[:4]))
test_무['month'] = test_무.YYYYMMSOON.apply(lambda x:int(x[4:6]))
test_무['soon'] = test_무.YYYYMMSOON.apply(lambda x:x[-2:])
test_무['month'] = np.cos(2 * np.pi * (test_무['month'] - 1) / 12)
test_무 = pd.merge(test_무,기상,on = ['YYYYMMSOON'],how = 'left')
test_무 = pd.merge(test_무,강수,on = ['YYYYMMSOON'],how = 'left')
test_무 = test_무.fillna(0)

df1 = test_무['평균가격(원)']

input_size = 9    
output_size = 0   
block_size = 9  

X_test, _ = create_sliding_block_dataset(df1, input_size, output_size, block_size)

X_test = pd.DataFrame([x.flatten() for x in X_test])  
X_test = pd.concat([test_무.iloc[8::9][['거래단위', '강수량','year', 'month', 'soon']].reset_index(drop=True),X_test.reset_index(drop=True)],axis=1)

with open('encoding/label_encoders_무.pkl', 'rb') as f:
    le_dict = pickle.load(f)

for col in le_dict.keys():
    X_test[col] = le_dict[col].transform(X_test[col])

X_test.columns = [str(i) for i in X_test.columns]

for i in range(8):
    X_test[f'{i}_{i+1}_diff'] = X_test[f'{i+1}'] - X_test[f'{i}']

with open('models/xgb_model_무.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scalers/scaler_무.pkl', 'rb') as f:
    scaler = pickle.load(f)

        # 예측 및 MAE 계산
pred = model.predict(X_test)

무_pred = scaler.inverse_transform(pred)
submit.loc[:,'무'] = 무_pred.reshape(156,)

################################# 대파 #######################################

test_대파 = test.query('품목명 == "대파(일반)"').reset_index(drop = True)
test_대파['year'] = test_대파.YYYYMMSOON.apply(lambda x:int(x[:4]))
test_대파['month'] = test_대파.YYYYMMSOON.apply(lambda x:int(x[4:6]))
test_대파['soon'] = test_대파.YYYYMMSOON.apply(lambda x:x[-2:])
test_대파['month'] = np.cos(2 * np.pi * (test_대파['month'] - 1) / 12)
test_대파 = pd.merge(test_대파,기상,on = ['YYYYMMSOON'],how = 'left')
test_대파 = pd.merge(test_대파,강수,on = ['YYYYMMSOON'],how = 'left')

df = test_대파['평균가격(원)']

input_size = 9    
output_size = 0   
block_size = 9  

X_test, _ = create_sliding_block_dataset(df, input_size, output_size, block_size)

X_test = pd.DataFrame([x.flatten() for x in X_test])  
X_test = pd.concat([test_대파.iloc[8::9][['강수량', '전순 평균가격(원) PreVious SOON', '0.5M 평균 습도(%)',
       '1.5M 평균 습도(%)', '4.0M 평균 습도(%)', '1.5M 평균 풍속(m/s)','4.0M 평균 풍속(m/s)', '10CM 일 토양수분(%)','20CM 일 토양수분(%)',
       '30CM 일 토양수분(%)', '50CM 일 토양수분(%)', 'soon']].reset_index(drop=True),X_test.reset_index(drop=True)],axis=1)

with open('encoding/label_encoders_대파.pkl', 'rb') as f:
    le_dict = pickle.load(f)

for col in ['soon']:
    X_test[col] = le_dict[col].transform(X_test[col])

X_test.columns = [str(i) for i in X_test.columns]
for i in range(8):
    X_test[f'{i}_{i+1}_diff'] = X_test[f'{i+1}'] - X_test[f'{i}']

with open('models/xgb_model_대파.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scalers/scaler_대파.pkl', 'rb') as f:
    scaler = pickle.load(f)

        # 예측 및 MAE 계산
pred = model.predict(X_test)

대파_pred = scaler.inverse_transform(pred)
submit.loc[:,'대파(일반)'] = 대파_pred.reshape(156,)

################################# 감자 #######################################

test_감자 = test.query('품목명 == "감자 수미"').reset_index(drop = True)
test_감자['year'] = test_감자.YYYYMMSOON.apply(lambda x:int(x[:4]))
test_감자['month'] = test_감자.YYYYMMSOON.apply(lambda x:int(x[4:6]))
test_감자['soon'] = test_감자.YYYYMMSOON.apply(lambda x:x[-2:])
test_감자['month'] = np.cos(2 * np.pi * (test_감자['month'] - 1) / 12)
test_감자 = pd.merge(test_감자,기상,on = ['YYYYMMSOON'],how = 'left')
test_감자 = pd.merge(test_감자,강수,on = ['YYYYMMSOON'],how = 'left')

df = test_감자['평균가격(원)']

input_size = 9    
output_size = 0   
block_size = 9  

X_test, _ = create_sliding_block_dataset(df, input_size, output_size, block_size)

X_test = pd.DataFrame([x.flatten() for x in X_test])  
X_test = pd.concat([test_감자.iloc[8::9][['거래단위','강수량','품목명',
       '전순 평균가격(원) PreVious SOON','전년 평균가격(원) PreVious YeaR',
       '평년 평균가격(원) Common Year SOON', '전달 평균가격(원) PreVious MMonth', 'year', 'month', 'soon','0.5M 평균 습도(%)', '1.5M 평균 습도(%)', '4.0M 평균 습도(%)', '10CM 일 토양수분(%)', '20CM 일 토양수분(%)', '30CM 일 토양수분(%)', '50CM 일 토양수분(%)','0.5M 평균 기온(°C)', '1.5M 평균 기온(°C)',
'4.0M 평균 기온(°C)', '1.5M 평균 풍속(m/s)', '4.0M 평균 풍속(m/s)', '평균 지면온도(°C)',
'최저 초상온도(°C)', '5CM 평균 지중온도(°C)', '10CM 평균 지중온도(°C)',
'20CM 평균 지중온도(°C)', '30CM 평균 지중온도(°C)', '0.5M 일 지중온도(°C)',
'1.0M 일 지중온도(°C)']].reset_index(drop=True),X_test.reset_index(drop=True)],axis=1)

with open('encoding/label_encoders_감자.pkl', 'rb') as f:
    le_dict = pickle.load(f)

for col in ['거래단위','품목명', 'soon']:
    X_test[col] = le_dict[col].transform(X_test[col])

X_test.columns = [str(i) for i in X_test.columns]

for i in range(8):
    X_test[f'{i}_{i+1}_diff'] = X_test[f'{i+1}'] - X_test[f'{i}']

with open('models/xgb_model_감자.pkl', 'rb') as f:
    model1 = pickle.load(f)

with open('models/et_model_감자.pkl', 'rb') as f:
    model2 = pickle.load(f)

with open('scalers/scaler_감자.pkl', 'rb') as f:
    scaler = pickle.load(f)

감자_pred1 = model1.predict(X_test)
감자_pred1 = scaler.inverse_transform(감자_pred1)

감자_pred2 = model2.predict(X_test)
감자_pred2 = scaler.inverse_transform(감자_pred2)

submit['감자 수미'] = 감자_pred1.reshape(156,)*0.5+감자_pred2.reshape(156,)*0.5

################################# test2 ####################################

test = pd.DataFrame()
for no in range(52):
   test = pd.concat([test, pd.read_csv(f'./data/test/TEST_{no:02}_2.csv')],axis=0)

################################# 상추 ######################################

test_상추 = test.query('품목명 == "상추"').reset_index(drop = True)
test_상추['year'] = test_상추.YYYYMMSOON.apply(lambda x:int(x[:4]))
test_상추['month'] = test_상추.YYYYMMSOON.apply(lambda x:int(x[4:6]))
test_상추['soon'] = test_상추.YYYYMMSOON.apply(lambda x:x[-2:])
test_상추['month'] = np.cos(2 * np.pi * (test_상추['month'] - 1) / 12)
test_상추 = pd.merge(test_상추,기상,on = ['YYYYMMSOON'],how = 'left')
test_상추 = pd.merge(test_상추,강수,on = ['YYYYMMSOON'],how = 'left')

df = test_상추['평균가격(원)']

input_size = 9    
output_size = 0   
block_size = 9  

X_test, _ = create_sliding_block_dataset(df, input_size, output_size, block_size)

X_test = pd.DataFrame([x.flatten() for x in X_test])  
X_test = pd.concat([test_상추.iloc[8::9][['강수량','유통단계별 무게 ','유통단계별 단위 ',
        '전순 평균가격(원) PreVious SOON','전년 평균가격(원) PreVious YeaR',
       '평년 평균가격(원) Common Year SOON', '전달 평균가격(원) PreVious MMonth', 'year', 'month', 'soon','0.5M 평균 습도(%)', '1.5M 평균 습도(%)', '4.0M 평균 습도(%)', '10CM 일 토양수분(%)', '20CM 일 토양수분(%)', '30CM 일 토양수분(%)', '50CM 일 토양수분(%)','0.5M 평균 기온(°C)', '1.5M 평균 기온(°C)',
'4.0M 평균 기온(°C)', '1.5M 평균 풍속(m/s)', '4.0M 평균 풍속(m/s)', '평균 지면온도(°C)',
'최저 초상온도(°C)', '5CM 평균 지중온도(°C)', '10CM 평균 지중온도(°C)',
'20CM 평균 지중온도(°C)', '30CM 평균 지중온도(°C)', '0.5M 일 지중온도(°C)',
'1.0M 일 지중온도(°C)']].reset_index(drop=True),X_test.reset_index(drop=True)],axis=1)

with open('encoding/label_encoders_상추.pkl', 'rb') as f:
    le_dict = pickle.load(f)

for col in ['soon','유통단계별 무게 ']:
    X_test[col] = le_dict[col].transform(X_test[col])

X_test.columns = [str(i) for i in X_test.columns]

for i in range(8):
    X_test[f'{i}_{i+1}_diff'] = X_test[f'{i+1}'] - X_test[f'{i}']

with open('models/xgb_model_상추.pkl', 'rb') as f:
    model1 = pickle.load(f)

with open('models/et_model_상추.pkl', 'rb') as f:
    model2 = pickle.load(f)

with open('scalers/scaler_상추.pkl', 'rb') as f:
    scaler = pickle.load(f)

상추_pred1 = model1.predict(X_test)
상추_pred1 = scaler.inverse_transform(상추_pred1)

상추_pred2 = model2.predict(X_test)
상추_pred2 = scaler.inverse_transform(상추_pred2)

submit['상추'] = 상추_pred1.reshape(156,)*0.5+상추_pred2.reshape(156,)*0.5

################################## 배 #######################################

test_배 = test.query('품목명 == "배"').reset_index(drop = True)
df = test_배['평균가격(원)']

input_size = 9    
output_size = 0   
block_size = 9  

X_test, _ = create_sliding_block_dataset(df, input_size, output_size, block_size)

X_test = pd.DataFrame([x.flatten() for x in X_test])  

X_test.columns = [str(i) for i in X_test.columns]

for i in range(8):
    X_test[f'{i}_{i+1}_diff'] = X_test[f'{i+1}'] - X_test[f'{i}']

with open('models/ransac_model_배.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scalers/scaler_배.pkl', 'rb') as f:
    scaler = pickle.load(f)

배_pred = model.predict(X_test)
배_pred = scaler.inverse_transform(배_pred)

submit.loc[:,"배"] = 배_pred.reshape(156,)

################################ 건고추 #####################################

건고추_test = test.query('품목명 == "건고추"').reset_index(drop = True)
건고추_test = pd.merge(건고추_test,강수,on = ['YYYYMMSOON'],how = 'left')
# 건고추_test = test.fillna(0)

df = 건고추_test['평균가격(원)']

input_size = 9    
output_size = 0   
block_size = 9  

X_test, _ = create_sliding_block_dataset(df, input_size, output_size, block_size)

X_test = pd.DataFrame([x.flatten() for x in X_test])  
X_test = pd.concat([건고추_test.iloc[8::9][['강수량']].reset_index(drop=True),X_test.reset_index(drop=True)],axis=1)
X_test.columns = [str(i) for i in X_test.columns]
for i in range(8):
    X_test[f'{i}_{i+1}_diff'] = X_test[f'{i+1}'] - X_test[f'{i}']

with open('models/ransac_model_건고추.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scalers/scaler_건고추.pkl', 'rb') as f:
    scaler = pickle.load(f)

건고추_pred = model.predict(X_test)
건고추_pred = scaler.inverse_transform(건고추_pred)

submit.loc[:,"건고추"] = 건고추_pred.reshape(156,)

################################# 제출 ######################################

submit.to_csv('./submission_최종/최종_submission.csv',index=False)

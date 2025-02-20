import pandas as pd
from glob import glob
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os
from sklearn.preprocessing import LabelEncoder

def create_sliding_block_dataset(df, input_size, output_size, block_size):
    X, y = [], []
    
    # 블록을 144개씩 나누어 처리
    for block_start in range(0, len(df), block_size):
        block_data = df.iloc[block_start:block_start + block_size] 
        
        if len(block_data) < input_size + output_size:
            break
        
        for i in range(len(block_data) - input_size - output_size + 1):
            X.append(block_data.iloc[i:i+input_size].values) 
            y.append(block_data.iloc[i+input_size:i+input_size+output_size].values) 

    return np.array(X), np.array(y)

def external_data(type):
    if type == '강수':
        강수 = pd.read_csv('data/외부/강수/강수일수.csv')
        강수 = 강수.iloc[3:,:-2].reset_index(drop=True)
        강수 = 강수.melt(id_vars=['연도'], var_name='월', value_name='강수량')

        강수['월'] = 강수['월'].str.replace('월', '').astype(int).astype(str).str.zfill(2)
        강수 = 강수.loc[강수.index.repeat(3)]
        강수['YYYYMMSOON'] = 강수.연도 + 강수.월
        강수['YYYYMMSOON'] =[f"{x}{'상순' if i % 3 == 0 else '중순' if i % 3 == 1 else '하순'}" for i, x in enumerate(강수['YYYYMMSOON'])]
        강수 = 강수[['강수량','YYYYMMSOON']]
        return 강수
    if type == '기상':
        기상 = pd.DataFrame()

        for i in glob('./data/외부/기상/*.csv'):
            file = i.replace('\\','/')
            기상 = pd.concat([기상,pd.read_csv(f"{file}",encoding='cp949')],axis = 0)
        기상.일시 = pd.to_datetime(기상.일시)
        def 상중하(날짜):
            day = 날짜.day
            if 1 <= day <= 10:
                return '상순'
            elif 11 <= day <= 20:
                return '중순'
            else:
                return '하순'
            
        기상['YYYYMMSOON'] = 기상['일시'].dt.strftime('%Y%m') + 기상['일시'].apply(상중하)
        기상 = 기상.groupby('YYYYMMSOON')[['0.5M 평균 습도(%)', '1.5M 평균 습도(%)', '4.0M 평균 습도(%)',
            '10CM 일 토양수분(%)', '20CM 일 토양수분(%)', '30CM 일 토양수분(%)', '50CM 일 토양수분(%)',
            '대형증발량(mm)', '소형증발량(mm)', '0.5M 평균 기온(°C)', '1.5M 평균 기온(°C)',
            '4.0M 평균 기온(°C)', '1.5M 평균 풍속(m/s)', '4.0M 평균 풍속(m/s)', '평균 지면온도(°C)',
            '최저 초상온도(°C)', '5CM 평균 지중온도(°C)', '10CM 평균 지중온도(°C)',
            '20CM 평균 지중온도(°C)', '30CM 평균 지중온도(°C)', '0.5M 일 지중온도(°C)',
            '1.0M 일 지중온도(°C)', '1.5M 일 지중온도(°C)', '3.0M 일 지중온도(°C)',
            '5.0M 일 지중온도(°C)', '일 순복사(MJ/m2)', '일 전천복사(MJ/m2)', '일 반사복사(MJ/m2)',
            '일 평균 조도(10lux)', '일 지하수위(cm)']].mean().reset_index()
        return 기상
    
기상 = external_data('기상')
강수 = external_data('강수')

os.makedirs('preprocessing_train', exist_ok=True)

################################################ 무 ##########################################################

train = pd.read_csv('./data/train/train_1.csv')
submit = pd.read_csv('./data/sample_submission.csv')
meta = pd.read_csv('data/train/meta/TRAIN_경락정보_가락도매_2018-2022.csv')
meta = meta.sort_values(['품목(품종)명', '거래단위','등급(특 5% 상 35% 중 40% 하 20%)'])
train = pd.concat([train,meta[(meta['품목(품종)명'] == '무') & (meta['거래단위'] == "20키로상자")]],axis=0)

meta = pd.read_csv('data/train/meta/TRAIN_경락정보_전국도매_2018-2022.csv')
meta = meta.query('품목명 == "무"')
temp_meta = meta.groupby(['품목명','시장명','품종명']).YYYYMMSOON.nunique()[meta.groupby(['품목명','시장명','품종명']).YYYYMMSOON.nunique() == 180].reset_index()

temp_df = pd.DataFrame()
for i in range(len(temp_meta)):
    temp_df = pd.concat([temp_df,meta[(meta['시장명'] == temp_meta.iloc[i]['시장명']) & (meta['품종명'] == temp_meta.iloc[i]['품종명'])]],axis=0)
meta = temp_df.reset_index(drop=True)
meta = meta.rename(columns = {'평균가(원/kg)':'평균가격(원)','품목명':'품목(품종)명','경매 건수':'등급(특 5% 상 35% 중 40% 하 20%)','품목코드':'가락시장 품목코드(5자리)'})
meta['거래단위'] = '1키로'

train = pd.concat([train, meta[train.columns]],axis=0).reset_index(drop=True)

train = train[(train['품목(품종)명'] == "무")].reset_index(drop=True)
train['year'] = train.YYYYMMSOON.apply(lambda x:int(x[:4]))
train['month'] = train.YYYYMMSOON.apply(lambda x:int(x[4:6]))
train['soon'] = train.YYYYMMSOON.apply(lambda x:x[-2:])

train['month'] = np.cos(2 * np.pi * (train['month'] - 1) / 12)

train = pd.merge(train,기상,on = ['YYYYMMSOON'],how = 'left')
train = pd.merge(train,강수,on = ['YYYYMMSOON'],how = 'left')

df = train['평균가격(원)']

input_size = 9   
output_size = 3  
block_size = 180 

X, y = create_sliding_block_dataset(df, input_size, output_size, block_size)

X = pd.DataFrame([x.flatten() for x in X])
temp_x = []
for i in range(int(len(train)/180)):
    temp_x.append(pd.concat([train[180*i:180*(i+1)][['거래단위', '강수량','year', 'month', 'soon']][8:177].reset_index(drop=True),X[169*i:169*(i+1)].reset_index(drop=True)],axis=1))
X = pd.concat(temp_x,axis = 0)

X.columns = [str(i) for i in X.columns]
for i in range(8):
    X[f'{i}_{i+1}_diff'] = X[f'{i+1}'] - X[f'{i}']
le_dict = {}
for col in ['거래단위', 'soon']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

    le_dict[col] = le

os.makedirs('encoding', exist_ok=True)

with open('encoding/label_encoders_무.pkl', 'wb') as f:
    pickle.dump(le_dict, f)

Y = pd.DataFrame(y)
Y.columns = ['y1','y2','y3']
X = X.reset_index(drop=True)
data = pd.concat([X,Y],axis = 1)
data.to_csv('preprocessing_train/preprocessing_무.csv',index=False)

############################################# 대파 ##########################################################

train1 = pd.read_csv('./data/train/train_1.csv')
meta1 = pd.read_csv('data/train/meta/TRAIN_경락정보_가락도매_2018-2022.csv')
meta1 = meta1.sort_values(['품목(품종)명', '거래단위','등급(특 5% 상 35% 중 40% 하 20%)'])
train1 = pd.concat([train1, meta1[(meta1['품목(품종)명'] == '대파(일반)') & (meta1['거래단위'] == "1키로단")]],axis=0)
meta = pd.read_csv('data/train/meta/TRAIN_경락정보_전국도매_2018-2022.csv')
meta = meta.query('품목명 == "대파"')
temp_meta = meta.groupby(['품목명','시장명','품종명']).YYYYMMSOON.nunique()[meta.groupby(['품목명','시장명','품종명']).YYYYMMSOON.nunique() == 180].reset_index().query('품종명 == "대파(일반)"')
temp_df = pd.DataFrame()
for i in range(len(temp_meta)):
    temp_df = pd.concat([temp_df,meta[(meta['시장명'] == temp_meta.iloc[i]['시장명']) & (meta['품종명'] == temp_meta.iloc[i]['품종명'])]],axis=0)
meta = temp_df.reset_index(drop=True)
meta = meta.rename(columns = {'평균가(원/kg)':'평균가격(원)','품종명':'품목(품종)명','경매 건수':'등급(특 5% 상 35% 중 40% 하 20%)','품목코드':'가락시장 품목코드(5자리)'})
meta['거래단위'] = '1키로'
train1 = pd.concat([train1, meta[train1.columns]],axis=0).reset_index(drop=True)

train1 = train1[(train1['품목(품종)명'] == "대파(일반)")]
train1['year'] = train1.YYYYMMSOON.apply(lambda x:int(x[:4]))
train1['month'] = train1.YYYYMMSOON.apply(lambda x:int(x[4:6]))
train1['soon'] = train1.YYYYMMSOON.apply(lambda x:x[-2:])
train1['month'] = np.cos(2 * np.pi * (train1['month'] - 1) / 12)

train1 = train1.rename(columns = {'품목(품종)명':'품목명'})
train1 = pd.merge(train1,기상,on = ['YYYYMMSOON'],how = 'left')
train1 = pd.merge(train1,강수,on = ['YYYYMMSOON'],how = 'left')

df1 = train1['평균가격(원)']

input_size = 9   
output_size = 3  
block_size = 180 

X1, y1 = create_sliding_block_dataset(df1, input_size, output_size, block_size)

X1 = pd.DataFrame([x.flatten() for x in X1])
temp_x = []
for i in range(int(len(train1)/180)):
    temp_x.append(pd.concat([train1[180*i:180*(i+1)][['강수량', '전순 평균가격(원) PreVious SOON', '0.5M 평균 습도(%)',
       '1.5M 평균 습도(%)', '4.0M 평균 습도(%)', '1.5M 평균 풍속(m/s)','4.0M 평균 풍속(m/s)', '10CM 일 토양수분(%)','20CM 일 토양수분(%)',
       '30CM 일 토양수분(%)', '50CM 일 토양수분(%)', 'soon']][8:177].reset_index(drop=True),X1[169*i:169*(i+1)].reset_index(drop=True)],axis=1))
X1 = pd.concat(temp_x,axis = 0)

le_dict = {}
for col in ['soon']:
    le = LabelEncoder()
    X1[col] = le.fit_transform(X1[col])

    le_dict[col] = le

with open('encoding/label_encoders_대파.pkl', 'wb') as f:
    pickle.dump(le_dict, f)

X1.columns = [str(i) for i in X1.columns]

for i in range(8):
    X1[f'{i}_{i+1}_diff'] = X1[f'{i+1}'] - X1[f'{i}']
Y = pd.DataFrame(y1)
Y.columns = ['y1','y2','y3']
X1 = X1.reset_index(drop=True)
data = pd.concat([X1,Y],axis = 1)
data.to_csv('preprocessing_train/preprocessing_대파.csv',index=False)

############################################# 감자 ##########################################################

train1 = pd.read_csv('./data/train/train_1.csv')
meta1 = pd.read_csv('data/train/meta/TRAIN_경락정보_가락도매_2018-2022.csv')
meta1 = meta1.sort_values(['품목(품종)명', '거래단위','등급(특 5% 상 35% 중 40% 하 20%)'])
train1 = pd.concat([train1,meta1[meta1['품목(품종)명'] == "감자 수미"]],axis=0)

meta = pd.read_csv('data/train/meta/TRAIN_경락정보_전국도매_2018-2022.csv')
meta = meta.query('품목명 == "감자"')
temp_meta = meta.groupby(['품목명','시장명','품종명']).YYYYMMSOON.nunique()[meta.groupby(['품목명','시장명','품종명']).YYYYMMSOON.nunique() == 180].reset_index()

temp_df = pd.DataFrame()
for i in range(len(temp_meta)):
    temp_df = pd.concat([temp_df,meta[(meta['시장명'] == temp_meta.iloc[i]['시장명']) & (meta['품종명'] == temp_meta.iloc[i]['품종명'])]],axis=0)
meta = temp_df.reset_index(drop=True)
meta = meta.rename(columns = {'평균가(원/kg)':'평균가격(원)','품목명':'품목(품종)명','경매 건수':'등급(특 5% 상 35% 중 40% 하 20%)','품목코드':'가락시장 품목코드(5자리)'})
meta['거래단위'] = '1키로'
train1 = pd.concat([train1, meta[train1.columns]],axis=0).reset_index(drop=True)

train1 = train1[(train1['품목(품종)명'] == "감자 수미")]
train1['year'] = train1.YYYYMMSOON.apply(lambda x:int(x[:4]))
train1['month'] = train1.YYYYMMSOON.apply(lambda x:int(x[4:6]))
train1['soon'] = train1.YYYYMMSOON.apply(lambda x:x[-2:])
train1['month'] = np.cos(2 * np.pi * (train1['month'] - 1) / 12)

train1 = train1.rename(columns = {'품목(품종)명':'품목명'})
train1 = pd.merge(train1,기상,on = ['YYYYMMSOON'],how = 'left')
train1 = pd.merge(train1,강수,on = ['YYYYMMSOON'],how = 'left')

df1 = train1['평균가격(원)']

input_size = 9   
output_size = 3  
block_size = 180 

X1, y1 = create_sliding_block_dataset(df1, input_size, output_size, block_size)

X1 = pd.DataFrame([x.flatten() for x in X1])

temp_x = []
for i in range(int(len(train1)/180)):
    temp_x.append(pd.concat([train1[180*i:180*(i+1)][['거래단위','강수량','품목명',
        '전순 평균가격(원) PreVious SOON','전년 평균가격(원) PreVious YeaR',
       '평년 평균가격(원) Common Year SOON', '전달 평균가격(원) PreVious MMonth', 'year', 'month', 'soon','0.5M 평균 습도(%)', '1.5M 평균 습도(%)', '4.0M 평균 습도(%)', '10CM 일 토양수분(%)', '20CM 일 토양수분(%)', '30CM 일 토양수분(%)', '50CM 일 토양수분(%)','0.5M 평균 기온(°C)', '1.5M 평균 기온(°C)',
'4.0M 평균 기온(°C)', '1.5M 평균 풍속(m/s)', '4.0M 평균 풍속(m/s)', '평균 지면온도(°C)',
'최저 초상온도(°C)', '5CM 평균 지중온도(°C)', '10CM 평균 지중온도(°C)',
'20CM 평균 지중온도(°C)', '30CM 평균 지중온도(°C)', '0.5M 일 지중온도(°C)',
'1.0M 일 지중온도(°C)']][8:177].reset_index(drop=True),X1[169*i:169*(i+1)].reset_index(drop=True)],axis=1))
X1 = pd.concat(temp_x,axis = 0)

le_dict = {}
for col in ['거래단위','품목명', 'soon']:
    le = LabelEncoder()
    X1[col] = le.fit_transform(X1[col])

    le_dict[col] = le

with open('encoding/label_encoders_감자.pkl', 'wb') as f:
    pickle.dump(le_dict, f)

X1.columns = [str(i) for i in X1.columns]

for i in range(8):
    X1[f'{i}_{i+1}_diff'] = X1[f'{i+1}'] - X1[f'{i}']
Y = pd.DataFrame(y1)
Y.columns = ['y1','y2','y3']
X1 = X1.reset_index(drop=True)
data = pd.concat([X1,Y],axis = 1)
data.to_csv('preprocessing_train/preprocessing_감자.csv',index=False)

############################################# 상추 ##########################################################

train2 = pd.read_csv('./data/train/train_2.csv')
meta = pd.read_csv('data/train/meta/TRAIN_소매_2018-2022.csv')

# 날짜 리스트를 생성하는 코드
years = range(2018, 2023)
months = range(1, 13)
periods = ['상순', '중순', '하순']

date_list = []

for year in years:
    for month in months:
        for period in periods:
            date_list.append(f"{year}{str(month).zfill(2)}{period}")

df = pd.DataFrame(date_list, columns=["YYYYMMSOON"])
meta = meta.sort_values(['품목명', '품종코드 ', '품종명', '등급명', '유통단계별 무게 ',
       '유통단계별 단위 '])
상추 = meta[(meta['품목명'] == '상추') & (meta['품종명'] == "청")]
train2 = pd.concat([train2,상추],axis=0)
meta = pd.read_csv('data/train/meta/TRAIN_경락정보_전국도매_2018-2022.csv')
meta = meta.query('품목명 == "상추"')
temp_meta = meta.groupby(['품목명','시장명','품종명']).YYYYMMSOON.nunique()[meta.groupby(['품목명','시장명','품종명']).YYYYMMSOON.nunique() == 180].reset_index().query('품종명 == "청상추"')
temp_df = pd.DataFrame()
for i in range(len(temp_meta)):
    temp_df = pd.concat([temp_df,meta[(meta['시장명'] == temp_meta.iloc[i]['시장명']) & (meta['품종명'] == temp_meta.iloc[i]['품종명'])]],axis=0)
meta = temp_df.reset_index(drop=True)
meta = meta.rename(columns = {'평균가(원/kg)':'평균가격(원)','경매 건수':'등급명','품목코드':'품목코드 ','품종코드':'품종코드 '})
meta['유통단계별 무게 '] = 'kg'
meta['유통단계별 단위 '] = 1
train2 = pd.concat([train2, meta[train2.columns]],axis=0).reset_index(drop=True)

train2 = train2[(train2['품목명'] == "상추")]
train2['year'] = train2.YYYYMMSOON.apply(lambda x:int(x[:4]))
train2['month'] = train2.YYYYMMSOON.apply(lambda x:int(x[4:6]))
train2['soon'] = train2.YYYYMMSOON.apply(lambda x:x[-2:])
train2['month'] = np.cos(2 * np.pi * (train2['month'] - 1) / 12)

train2 = pd.merge(train2,기상,on = ['YYYYMMSOON'],how = 'left')
train2 = pd.merge(train2,강수,on = ['YYYYMMSOON'],how = 'left')

df = train2['평균가격(원)']

input_size = 9   
output_size = 3  
block_size = 180 

X1, y1 = create_sliding_block_dataset(df, input_size, output_size, block_size)

X1 = pd.DataFrame([x.flatten() for x in X1])
temp_x = []
for i in range(int(len(train2)/180)):
    temp_x.append(pd.concat([train2[180*i:180*(i+1)][['강수량','유통단계별 무게 ','유통단계별 단위 ',
        '전순 평균가격(원) PreVious SOON','전년 평균가격(원) PreVious YeaR',
       '평년 평균가격(원) Common Year SOON', '전달 평균가격(원) PreVious MMonth', 'year', 'month', 'soon','0.5M 평균 습도(%)', '1.5M 평균 습도(%)', '4.0M 평균 습도(%)', '10CM 일 토양수분(%)', '20CM 일 토양수분(%)', '30CM 일 토양수분(%)', '50CM 일 토양수분(%)','0.5M 평균 기온(°C)', '1.5M 평균 기온(°C)',
'4.0M 평균 기온(°C)', '1.5M 평균 풍속(m/s)', '4.0M 평균 풍속(m/s)', '평균 지면온도(°C)',
'최저 초상온도(°C)', '5CM 평균 지중온도(°C)', '10CM 평균 지중온도(°C)',
'20CM 평균 지중온도(°C)', '30CM 평균 지중온도(°C)', '0.5M 일 지중온도(°C)',
'1.0M 일 지중온도(°C)']][8:177].reset_index(drop=True),X1[169*i:169*(i+1)].reset_index(drop=True)],axis=1))
X1 = pd.concat(temp_x,axis = 0)
le_dict = {}
for col in ['soon','유통단계별 무게 ']:
    le = LabelEncoder()
    X1[col] = le.fit_transform(X1[col])

    le_dict[col] = le

with open('encoding/label_encoders_상추.pkl', 'wb') as f:
    pickle.dump(le_dict, f)

X1.columns = [str(i) for i in X1.columns]

for i in range(8):
    X1[f'{i}_{i+1}_diff'] = X1[f'{i+1}'] - X1[f'{i}']
Y = pd.DataFrame(y1)
Y.columns = ['y1','y2','y3']
X1 = X1.reset_index(drop=True)
data = pd.concat([X1,Y],axis = 1)
data.to_csv('preprocessing_train/preprocessing_상추.csv',index=False)

############################################# 배 ##########################################################

train2 = pd.read_csv('./data/train/train_2.csv')
meta = pd.read_csv('data/train/meta/TRAIN_소매_2018-2022.csv')

years = range(2018, 2023)
months = range(1, 13)
periods = ['상순', '중순', '하순']

date_list = []

for year in years:
    for month in months:
        for period in periods:
            date_list.append(f"{year}{str(month).zfill(2)}{period}")

df = pd.DataFrame(date_list, columns=["YYYYMMSOON"])
배 = pd.merge(df,meta[(meta['품목명'] == '배') & (meta['품종명'] == "신고")],how='left')
배 = 배.ffill()

train2 = pd.concat([train2,배],axis=0)

meta = pd.read_csv('data/train/meta/TRAIN_경락정보_전국도매_2018-2022.csv')
meta = meta.query('품목명 == "배"')
temp_meta = meta.groupby(['품목명','시장명','품종명']).YYYYMMSOON.nunique()[meta.groupby(['품목명','시장명','품종명']).YYYYMMSOON.nunique() == 180].reset_index()
temp_df = pd.DataFrame()
for i in range(len(temp_meta)):
    temp_df = pd.concat([temp_df,meta[(meta['시장명'] == temp_meta.iloc[i]['시장명']) & (meta['품종명'] == temp_meta.iloc[i]['품종명'])]],axis=0)
meta = temp_df.reset_index(drop=True)
meta = meta.rename(columns = {'평균가(원/kg)':'평균가격(원)','경매 건수':'등급명','품목코드':'품목코드 ','품종코드':'품종코드 '})
meta['유통단계별 무게 '] = 'kg'
meta['유통단계별 단위 '] = 1
train2 = pd.concat([train2, meta[train2.columns]],axis=0).reset_index(drop=True)
train2 = train2[(train2['품목명'] == "배")]
train2['year'] = train2.YYYYMMSOON.apply(lambda x:int(x[:4]))
train2['month'] = train2.YYYYMMSOON.apply(lambda x:int(x[4:6]))
train2['soon'] = train2.YYYYMMSOON.apply(lambda x:x[-2:])
train2['month'] = np.cos(2 * np.pi * (train2['month'] - 1) / 12)

df1 = train2['평균가격(원)']

input_size = 9   
output_size = 3  
block_size = 180 

X1, y1 = create_sliding_block_dataset(df1, input_size, output_size, block_size)

X1 = pd.DataFrame([x.flatten() for x in X1])
X1.columns = [str(i) for i in X1.columns]
for i in range(8):
    X1[f'{i}_{i+1}_diff'] = X1[f'{i+1}'] - X1[f'{i}']
Y = pd.DataFrame(y1)
Y.columns = ['y1','y2','y3']
X1 = X1.reset_index(drop=True)
data = pd.concat([X1,Y],axis = 1)
data.to_csv('preprocessing_train/preprocessing_배.csv',index=False)

############################################ 건고추 #########################################################

train2 = pd.read_csv('./data/train/train_2.csv')
meta2 = pd.read_csv('data/train/meta/TRAIN_중도매_2018-2022.csv')
# 날짜 리스트를 생성하는 코드
import pandas as pd

years = range(2018, 2023)
months = range(1, 13)
periods = ['상순', '중순', '하순']

date_list = []

for year in years:
    for month in months:
        for period in periods:
            date_list.append(f"{year}{str(month).zfill(2)}{period}")

df = pd.DataFrame(date_list, columns=["YYYYMMSOON"])
meta2 = meta2.sort_values(['품목명', '품종코드 ', '품종명', '등급명', '유통단계별 무게 ',
       '유통단계별 단위 '])
건고추 = pd.merge(df,meta2[(meta2['품목명'] == '건고추') & (meta2['품종명'] == "화건")],how='left')
건고추 = 건고추.ffill()

train2 = pd.concat([train2,건고추],axis=0)

train2 = train2[(train2['품목명'] == "건고추")]
train2 = pd.merge(train2,강수,on = ['YYYYMMSOON'],how = 'left')

df1 = train2['평균가격(원)']

input_size = 9   
output_size = 3  
block_size = 180 

X1, y1 = create_sliding_block_dataset(df1, input_size, output_size, block_size)

X1 = pd.DataFrame([x.flatten() for x in X1])
temp_x = []
for i in range(int(len(train2)/180)):
    temp_x.append(pd.concat([train2[180*i:180*(i+1)][['강수량']][8:177].reset_index(drop=True),X1[169*i:169*(i+1)].reset_index(drop=True)],axis=1))

X1 = pd.concat(temp_x,axis = 0)

X1.columns = [str(i) for i in X1.columns]

for i in range(8):
    X1[f'{i}_{i+1}_diff'] = X1[f'{i+1}'] - X1[f'{i}']
Y = pd.DataFrame(y1)
Y.columns = ['y1','y2','y3']
X1 = X1.reset_index(drop=True)
data = pd.concat([X1,Y],axis = 1)
data.to_csv('preprocessing_train/preprocessing_건고추.csv',index=False)
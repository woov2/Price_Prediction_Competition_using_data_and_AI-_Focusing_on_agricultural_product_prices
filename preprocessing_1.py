import pandas as pd
from glob import glob
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder

def process_garak_domae(df, 품종명_list):
    # 컬럼명 리네임
    df = df.rename(columns={
        '품목(품종)명':'품종명',
        'YYYYMMSOON':'시점',
        '등급(특 5% 상 35% 중 40% 하 20%)':'등급',
        '평균가격(원)':'가락_평균가격',
        '전순 평균가격(원) PreVious SOON':'가락_전순가격',
        '전달 평균가격(원) PreVious MMonth':'가락_전달가격',
        '전년 평균가격(원) PreVious YeaR':'가락_전년가격',
        '평년 평균가격(원) Common Year SOON':'가락_평년가격'
    })
    
    # 필요 없는 컬럼 삭제
    df = df.drop(columns=['가락시장 품목코드(5자리)'])

    # '등급'이 '중'인 데이터만 필터링
    df = df[df['등급'] == '중']

    # 값 변경
    df = df.replace({
        '배 신고':'신고',
        '청상추':'청',
        '홍고추':'화건',
        '사과 홍로':'홍로',
        '햇마늘 한지':'깐마늘(국산)'
    })

    # 품종명이 주어진 리스트에 포함된 데이터만 필터링
    df = df[df['품종명'].isin(품종명_list)]

    # 최빈값 계산 함수 정의
    def mode_func(x):
        return x.mode()[0]  # 최빈값이 여러 개일 경우 첫 번째 값 반환

    # 품종명별로 거래단위의 최빈값 추출
    mode_df = df.groupby('품종명')['거래단위'].agg(mode_func).reset_index()

    # 최빈값을 원본 데이터프레임에 병합하여 필터링
    filtered_df = pd.merge(df, mode_df, on=['품종명', '거래단위'], how='inner')

    # 불필요한 컬럼 삭제
    filtered_df = filtered_df.drop(columns=['거래단위', '등급'])

    return filtered_df

def process_national_domae(df, 품종명_list):
    # 컬럼명 리네임
    df = df.rename(columns={
        'YYYYMMSOON':'시점',
        '총반입량(kg)':'전국_총반입량',
        '총거래금액(원)':'전국_총거래금액',
        '평균가(원/kg)':'전국_평균가',
        '고가(20%) 평균가':'전국_고가',
        '중가(60%) 평균가':'전국_중가',
        '저가(20%) 평균가':'전국_저가',
        '중간가(원/kg)':'전국_중간가',
        '최저가(원/kg)':'전국_최저가',
        '최고가(원/kg)':'전국_최고가',
        '경매 건수':'전국_경매건수',
        '전순 평균가격(원) PreVious SOON':'전국_전순가격',
        '전달 평균가격(원) PreVious MMonth':'전국_전달가격',
        '전년 평균가격(원) PreVious YeaR':'전국_전년가격',
        '평년 평균가격(원) Common Year SOON':'전국_평년가격'
    })

    # 전국도매시장 데이터만 필터링
    df = df[df['시장명'] == '*전국도매시장']

    # 필요 없는 컬럼 삭제
    df = df.drop(columns=['시장코드', '시장명', '품목코드', '품종코드', '연도', '품목명'])

    # 값 변경
    df = df.replace({
        '홍감자': '감자 수미',
        '깐마늘 한지': '깐마늘(국산)',
        '청상추': '청'
    })

    # 품종명이 주어진 리스트에 포함된 데이터만 필터링
    df = df[df['품종명'].isin(품종명_list)].reset_index(drop=True)

    return df

def process_sanji_gongpanjang(df, 품종명_list):
    # 전국농협공판장 데이터만 필터링
    df = df[df['공판장명'] == '*전국농협공판장']

    # 필요 없는 컬럼 삭제
    df = df.drop(columns=['공판장코드', '공판장명', '품목코드', '품목명', '품종코드', '등급코드'])

    # 컬럼명 리네임
    df = df.rename(columns={
        'YYYYMMSOON':'시점',
        '총반입량(kg)':'산지_총반입량',
        '총거래금액(원)':'산지_총거래금액',
        '평균가(원/kg)':'산지_평균가',
        '중간가(원/kg)':'산지_중간가',
        '최저가(원/kg)':'산지_최저가',
        '최고가(원/kg)':'산지_최고가',
        '경매 건수':'산지_경매건수',
        '전순 평균가격(원) PreVious SOON':'산지_전순가격',
        '전달 평균가격(원) PreVious MMonth':'산지_전달가격',
        '전년 평균가격(원) PreVious YeaR':'산지_전년가격',
        '평년 평균가격(원) Common Year SOON':'산지_평년가격'
    })

    # '등급명'이 '특'인 데이터만 필터링
    df = df[df['등급명'] == '특']

    # 값 변경
    df = df.replace({
        '쌈배추': '배추',
        '저장무': '무',
        '양파(일반)': '양파',
        '홍감자': '감자 수미',
        '깐마늘': '깐마늘(국산)',
        '청상추': '청'
    })

    # 품종명이 주어진 리스트에 포함된 데이터만 필터링
    df = df[df['품종명'].isin(품종명_list)].reset_index(drop=True)

    # 불필요한 컬럼 삭제
    df = df.drop(columns=['등급명', '연도'])

    return df

def create_sliding_block_dataset_version2(df, input_size, output_size, block_size):
    X, y = [], []

    # 블록을 180개씩 나누어 처리
    for block_start in range(0, len(df), block_size):
        block_data = df.iloc[block_start:block_start + block_size]

        if len(block_data) < input_size + output_size:
            break

        for i in range(len(block_data) - input_size - output_size + 1):
            # input은 모든 열을 사용, output은 평균가격(원)만 사용
            X.append(block_data.iloc[i:i+input_size].values)
            y.append(block_data.iloc[i+input_size:i+input_size+output_size]['평균가격(원)'].values)

    return np.array(X), np.array(y)

def process_jungdomae(df, 품종명_list):
    # 컬럼명 리네임
    df = df.rename(columns={
        'YYYYMMSOON': '시점',
        '평균가격(원)': '중도매_평균가격(원)',
        '전순 평균가격(원) PreVious SOON': '중도매_전순가격',
        '전달 평균가격(원) PreVious MMonth': '중도매_전달가격',
        '전년 평균가격(원) PreVious YeaR': '중도매_전년가격',
        '평년 평균가격(원) Common Year SOON': '중도매_평년가격'
    })

    # 필요한 품종명과 등급명 필터링
    감자 = df[(df['품종명'] == '수미') & (df['등급명'] == '상품')] 
    마늘 = df[(df['품종명'] == '깐마늘(대서)')]  # & (df['등급명'] == '상품')] 
    대파 = df[(df['품종명'] == '대파') & (df['등급명'] == '상품')] 
    무 = df[(df['품목명'] == '무') & (df['품종명'] == '월동') & (df['등급명'] == '상품')]
    배추 = df[(df['품목명'] == '배추') & (df['품종명'] == '월동') & (df['등급명'] == '상품')]
    신고 = df[(df['품종명'] == '신고') & (df['등급명'] == '상품')] 
    양파 = df[(df['품종명'] == '양파') & (df['등급명'] == '상품')] 
    청 = df[(df['품종명'] == '청') & (df['등급명'] == '상품')] 
    화건 = df[(df['품종명'] == '화건')]  # & (df['등급명'] == '상품')] 
    후지 = df[(df['품종명'] == '후지') & (df['등급명'] == '상품')] 
    홍로 = df[(df['품종명'] == '홍로') & (df['등급명'] == '상품')] 

    # 필터링된 데이터 결합
    data = pd.concat([감자, 마늘, 대파, 무, 배추, 신고, 양파, 청, 화건, 후지, 홍로])

    # 품목명과 품종명을 결합하여 새로운 품종명 생성
    data['품종명'] = data['품목명'] + data['품종명']

    # 값 변경
    data = data.replace({
        '감자수미': '감자 수미',
        '깐마늘(국산)깐마늘(대서)': '깐마늘(국산)',
        '파대파': '대파(일반)',
        '무월동': '무',
        '배추월동': '배추',
        '배신고': '신고',
        '양파양파': '양파',
        '상추청': '청',
        '건고추화건': '화건',
        '사과후지': '후지',
        '사과홍로': '홍로'
    })

    # 품종명이 주어진 리스트에 포함된 데이터만 필터링
    data = data[data['품종명'].isin(품종명_list)].reset_index(drop=True)

    # 필요 없는 컬럼 삭제
    data = data.drop(columns=['품목코드 ', '품목명', '품종코드 ', '등급명', '유통단계별 무게 ', '유통단계별 단위 '])

    return data

# 가락도매
def process_retail(df, 품종명_list):
    # 컬럼명 리네임
    df = df.rename(columns={
        'YYYYMMSOON': '시점',
        '평균가격(원)': '소매_평균가격(원)',
        '전순 평균가격(원) PreVious SOON': '소매_전순가격',
        '전달 평균가격(원) PreVious MMonth': '소매_전달가격',
        '전년 평균가격(원) PreVious YeaR': '소매_전년가격',
        '평년 평균가격(원) Common Year SOON': '소매_평년가격'
    })

    # 필요한 품종명과 등급명 필터링
    감자 = df[(df['품종명'] == '수미') & (df['등급명'] == '상품')] 
    깐마늘 = df[(df['품종명'] == '깐마늘(국산)') & (df['등급명'] == '상품')]
    대파 = df[(df['품종명'] == '대파') & (df['등급명'] == '상품')] 
    무 = df[(df['품목명'] == '무') & (df['품종명'] == '월동') & (df['등급명'] == '상품')]
    배추 = df[(df['품목명'] == '배추') & (df['품종명'] == '월동') & (df['등급명'] == '상품')]
    신고 = df[(df['품종명'] == '신고')]  # 등급 필터링 없음
    양파 = df[(df['품종명'] == '양파') & (df['등급명'] == '상품')] 
    청 = df[(df['품종명'] == '청')]  # 등급 필터링 없음
    화건 = df[(df['품종명'] == '화건') & (df['등급명'] == '상품')] 
    후지 = df[(df['품종명'] == '후지') & (df['등급명'] == '중품')] 
    홍로 = df[(df['품종명'] == '홍로') & (df['등급명'] == '중품')] 

    # 필터링된 데이터 결합
    data = pd.concat([감자, 깐마늘, 대파, 무, 배추, 신고, 양파, 청, 화건, 후지, 홍로])

    # 품목명과 품종명을 결합하여 새로운 품종명 생성
    data['품종명'] = data['품목명'] + data['품종명']

    # 값 변경
    data = data.replace({
        '감자수미': '감자 수미',
        '깐마늘(국산)깐마늘(국산)': '깐마늘(국산)',
        '파대파': '대파(일반)',
        '무월동': '무',
        '배추월동': '배추',
        '배신고': '신고',
        '양파양파': '양파',
        '상추청': '청',
        '건고추화건': '화건',
        '사과후지': '후지',
        '사과홍로': '홍로'
    })

    # 품종명이 주어진 리스트에 포함된 데이터만 필터링
    data = data[data['품종명'].isin(품종명_list)].reset_index(drop=True)

    # 필요 없는 컬럼 삭제
    data = data.drop(columns=['품목코드 ', '품목명', '품종코드 ', '등급명', '유통단계별 무게 ', '유통단계별 단위 '])

    return data

def month_sequence_to_sin_cos(month, sequence):
    # 월과 순에 따른 인덱스를 계산: (월 * 3 + 순 - 1)
    index = (month * 3 + sequence - 1)
    # 1년을 총 36개 구간으로 나눈다 (12개월 * 3순)
    total_divisions = 12 * 3
    # 인덱스를 기준으로 각도를 계산 (라디안 값)
    angle = 2 * np.pi * index / total_divisions
    # sin, cos 값을 계산
    sin_value = np.sin(angle)
    cos_value = np.cos(angle)
    return sin_value, cos_value

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

os.makedirs('encoding', exist_ok=True)
os.makedirs('preprocessing_train', exist_ok=True)

####################### Train Data Concat ###############################

train1 = pd.read_csv('./data/train/train_1.csv').drop(columns=['가락시장 품목코드(5자리)'])
train1['품종명'] = train1['품목(품종)명']
train1.loc[train1['품목(품종)명']=="감자 수미","품목(품종)명"] = "감자"
train1.loc[train1['품목(품종)명']=="대파(일반)","품목(품종)명"] = "대파"
train1.rename(columns={'YYYYMMSOON':'시점','품목(품종)명':'품목명','등급(특 5% 상 35% 중 40% 하 20%)':'등급','전순 평균가격(원) PreVious SOON':'전순 평균가격(원)','전달 평균가격(원) PreVious MMonth':'전달 평균가격(원)','전년 평균가격(원) PreVious YeaR':'전년 평균가격(원)','평년 평균가격(원) Common Year SOON':'평년 평균가격(원)'}, inplace=True)
train1 = train1[['시점','품목명','품종명','거래단위','등급','전순 평균가격(원)','전달 평균가격(원)','전년 평균가격(원)','평년 평균가격(원)','평균가격(원)']]

train2 = pd.read_csv('./data/train/train_2.csv').drop(columns=['품목코드 ','품종코드 '])
train2['거래단위'] = train2['유통단계별 단위 '].apply(lambda x : str(int(x))) + " " + train2['유통단계별 무게 '].apply(lambda x : str(x))
train2.drop(columns=['유통단계별 단위 ','유통단계별 무게 '], inplace=True)
train2.rename(columns={'YYYYMMSOON':'시점','등급명':'등급','전순 평균가격(원) PreVious SOON':'전순 평균가격(원)','전달 평균가격(원) PreVious MMonth':'전달 평균가격(원)','전년 평균가격(원) PreVious YeaR':'전년 평균가격(원)','평년 평균가격(원) Common Year SOON':'평년 평균가격(원)'}, inplace=True)
train2 = train2[['시점','품목명','품종명','거래단위','등급','전순 평균가격(원)','전달 평균가격(원)','전년 평균가격(원)','평년 평균가격(원)','평균가격(원)']]

train = pd.concat([train1,train2], axis=0).reset_index(drop=True)
train.to_csv('./data/train/train.csv', index=False)

###################### Feature Engineering ##############################

# 기상정보
기상 = pd.read_csv('./data/train/meta/TRAIN_기상_2018-2022.csv')
train = pd.read_csv('./data/train/train.csv')

기상['주산지 품목명'] = 기상['주산지 품목명'].apply(lambda x : x.replace("마늘","깐마늘(국산)"))
기상 = 기상.groupby(['YYYYMMSOON','주산지 품목명'])[['순 평균상대습도', '순 평균기온',
       '순 평균풍속', '순 최고기온', '순 최저상대습도', '순 최저기온', '순 강수량', '순 누적 일조시간', '특보 발효 순통계 개수']].mean().reset_index()
기상_train = 기상.rename(columns = {'주산지 품목명':'품목명','YYYYMMSOON':'시점'})

train = pd.merge(train,기상_train,on = ['시점','품목명'],how = 'left')

train['year'] = train.시점.apply(lambda x:int(x[:4]))
train['month'] = train.시점.apply(lambda x:int(x[4:6]))
train['soon'] = train.시점.apply(lambda x:x[-2:])
train['sin_month'] = np.sin(2 * np.pi * (train['month'] - 1) / 12)
train['cos_month'] = np.cos(2 * np.pi * (train['month'] - 1) / 12)
train['is_peak'] = train['month'].apply(lambda x : 1 if x in [11,12,1] else 0)
train['is_bottom'] = train['month'].apply(lambda x : 1 if x in [2,3,6,7] else 0)
train['sin_date'], train['cos_date'] = month_sequence_to_sin_cos(train['month'], train['soon'].map({'상순':1,'중순':2,"하순":3}))

가락도매 = pd.read_csv('./data/train/meta/TRAIN_경락정보_가락도매_2018-2022.csv')
전국도매 = pd.read_csv('./data/train/meta/TRAIN_경락정보_전국도매_2018-2022.csv')
산지공판장 = pd.read_csv('./data/train/meta/TRAIN_경락정보_산지공판장_2018-2022.csv')
중도매 = pd.read_csv('./data/train/meta/TRAIN_중도매_2018-2022.csv')
소매 = pd.read_csv('./data/train/meta/TRAIN_소매_2018-2022.csv')

lst_tr = train['품종명'].unique().tolist()

filtered_df = process_garak_domae(가락도매, lst_tr)
train = train.merge(filtered_df,on=['품종명','시점'],how='left')

national_filtered_df = process_national_domae(전국도매, lst_tr)
train = train.merge(national_filtered_df,on=['품종명','시점'],how='left')

sanji = process_sanji_gongpanjang(산지공판장, lst_tr)
train = train.merge(sanji,on=['품종명','시점'],how='left')

filtered_data = process_jungdomae(중도매, lst_tr)
train = train.merge(filtered_data,on=['품종명','시점'],how='left')

filtered_retail_data = process_retail(소매, lst_tr)
train = train.merge(filtered_retail_data,on=['품종명','시점'],how='left')

category_list = list(train['품목명'].unique())

for name in category_list:
    if name != "양파":
        train.loc[train['품목명']==name, "전순 평균가격(원)"] = train.loc[train['품목명']==name]['평균가격(원)'].shift(1)
        train.loc[train['품목명']==name, "전달 평균가격(원)"] = train.loc[train['품목명']==name]['평균가격(원)'].shift(3)
train['전순 평균가격(원)'].fillna(train['평균가격(원)'], inplace=True)
train['전달 평균가격(원)'].fillna(train['평균가격(원)'], inplace=True)
train.loc[train['전년 평균가격(원)']==0 , '전년 평균가격(원)'] =  train['평균가격(원)']
train.loc[train['평년 평균가격(원)']==0 , '평년 평균가격(원)'] =  train['평균가격(원)']
train.fillna(0, inplace=True)

train['tr_val_1'] = train['평균가격(원)'] - train['평년 평균가격(원)']
train['tr_val_2'] = train['평균가격(원)'] - train['전년 평균가격(원)']
train['tr_val_3'] = train['평균가격(원)'] - train['전달 평균가격(원)']
train['tr_val_4'] = train['평균가격(원)'] - train['전순 평균가격(원)']

train['tr_val_5'] = train['평년 평균가격(원)'] - train['전년 평균가격(원)']
train['tr_val_6'] = train['평년 평균가격(원)'] - train['전달 평균가격(원)']
train['tr_val_7'] = train['평년 평균가격(원)'] - train['전순 평균가격(원)']

train['tr_val_8'] = train['전년 평균가격(원)'] - train['전달 평균가격(원)']
train['tr_val_9'] = train['전년 평균가격(원)'] - train['전순 평균가격(원)']

train['tr_val_10'] = train['전달 평균가격(원)'] - train['전순 평균가격(원)']

#################################### 양파 #################################### 

양파_col = ['시점', '품목명', '품종명', '거래단위', '등급', '전순 평균가격(원)', '전달 평균가격(원)',
       '전년 평균가격(원)', '평년 평균가격(원)', '평균가격(원)', 'year', 'month', 'soon',
       'sin_month', 'cos_month', 'is_peak', 'is_bottom']
train_양파 = train[양파_col]
train_양파.to_csv("preprocessing_train/train_양파.csv", index=False)

################################### 깐마늘 ####################################

깐마늘_col = ['시점', '품목명', '품종명', '거래단위', '등급', '전순 평균가격(원)', '전달 평균가격(원)',
       '전년 평균가격(원)', '평년 평균가격(원)', '평균가격(원)', '순 평균상대습도', '순 평균기온', '순 평균풍속',
       '순 최고기온', '순 최저상대습도', '순 최저기온', '순 강수량', '순 누적 일조시간', '특보 발효 순통계 개수',
       'year', 'month', 'soon', 'sin_month', 'cos_month', 'sin_date',
       'cos_date']
train_깐마늘 = train[깐마늘_col]
train_깐마늘.to_csv("preprocessing_train/train_깐마늘.csv", index=False)

#################################### 배추 ####################################

배추_col = ['시점', '품목명', '품종명', '거래단위', '등급', '전순 평균가격(원)', '전달 평균가격(원)',
       '전년 평균가격(원)', '평년 평균가격(원)', '평균가격(원)', '순 평균상대습도', '순 평균기온', '순 평균풍속',
       '순 최고기온', '순 최저상대습도', '순 최저기온', '순 강수량', '순 누적 일조시간', '특보 발효 순통계 개수',
       'year', 'month', 'soon', 'sin_month', 'cos_month','가락_평균가격', '가락_전순가격', '가락_전달가격',
       '가락_전년가격', '가락_평년가격', '전국_총반입량', '전국_총거래금액', '전국_평균가', '전국_고가',
       '중가(60%) 평균가 ', '전국_저가', '전국_중간가', '전국_최저가', '전국_최고가', '전국_경매건수',
       '전국_전순가격', '전국_전달가격', '전국_전년가격', '전국_평년가격', '산지_총반입량', '산지_총거래금액',
       '산지_평균가','산지_중간가', '산지_최저가', '산지_최고가', '산지_경매건수', '산지_전순가격', '산지_전달가격',
       '산지_전년가격', '산지_평년가격', '중도매_평균가격(원)', '중도매_전순가격', '중도매_전달가격', '중도매_전년가격',
       '중도매_평년가격', '소매_평균가격(원)', '소매_전순가격', '소매_전달가격', '소매_전년가격', '소매_평년가격',
       'tr_val_1', 'tr_val_2', 'tr_val_3', 'tr_val_4', 'tr_val_5', 'tr_val_6',
       'tr_val_7', 'tr_val_8', 'tr_val_9', 'tr_val_10']
train_배추 = train[배추_col]
train_배추.to_csv("preprocessing_train/train_배추.csv", index=False)

#################################### 사과 ####################################

사과_col = ['시점', '품목명', '품종명', '거래단위', '등급', '전순 평균가격(원)', '전달 평균가격(원)',
       '전년 평균가격(원)', '평년 평균가격(원)', '평균가격(원)', 
          '순 평균상대습도', '순 평균기온', '순 평균풍속',
       '순 최고기온', '순 최저상대습도', '순 최저기온', '순 강수량', '순 누적 일조시간', '특보 발효 순통계 개수',
       'year', 'month', 'soon', 'sin_month', 'cos_month','전국_총반입량', '전국_총거래금액', '전국_평균가', '전국_고가',
       '중가(60%) 평균가 ', '전국_저가', '전국_중간가', '전국_최저가', '전국_최고가', '전국_경매건수',
       '전국_전순가격', '전국_전달가격', '전국_전년가격', '전국_평년가격']
train_사과 = train[사과_col]
train_사과.to_csv("preprocessing_train/train_사과.csv", index=False)

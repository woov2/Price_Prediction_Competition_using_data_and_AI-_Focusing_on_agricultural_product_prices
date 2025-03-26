import pandas as pd
import pickle
import numpy as np
import joblib
import os

from preprocessing_1 import create_sliding_block_dataset, external_data
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

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

os.makedirs('submission_최종', exist_ok=True)

# Initialize the combined DataFrame
test_total = pd.DataFrame()

for i in range(52):
    # Add padding to the filename if i is less than 10
    padding = "0"
    if i < 10:
        i = padding + str(i)
    
    # Load the first test dataset
    test1 = pd.read_csv(f'./data/test/TEST_{i}_1.csv').drop(columns=['가락시장 품목코드(5자리)'])
    test1['품종명'] = test1['품목(품종)명']
    test1.loc[test1['품목(품종)명'] == "감자 수미", "품목(품종)명"] = "감자"
    test1.loc[test1['품목(품종)명'] == "대파(일반)", "품목(품종)명"] = "대파"
    test1.rename(columns={
        'YYYYMMSOON': '시점',
        '품목(품종)명': '품목명',
        '등급(특 5% 상 35% 중 40% 하 20%)': '등급',
        '전순 평균가격(원) PreVious SOON': '전순 평균가격(원)',
        '전달 평균가격(원) PreVious MMonth': '전달 평균가격(원)',
        '전년 평균가격(원) PreVious YeaR': '전년 평균가격(원)',
        '평년 평균가격(원) Common Year SOON': '평년 평균가격(원)'
    }, inplace=True)
    test1 = test1[['시점', '품목명', '품종명', '거래단위', '등급', '전순 평균가격(원)', '전달 평균가격(원)', '전년 평균가격(원)', '평년 평균가격(원)', '평균가격(원)']]

    # Load the second test dataset
    test2 = pd.read_csv(f'./data/test/TEST_{i}_2.csv').drop(columns=['품목코드 ', '품종코드 '])
    test2['거래단위'] = test2['유통단계별 단위 '].apply(lambda x: str(int(x))) + " " + test2['유통단계별 무게 '].apply(lambda x: str(x))
    test2.drop(columns=['유통단계별 단위 ', '유통단계별 무게 '], inplace=True)
    test2.rename(columns={
        'YYYYMMSOON': '시점',
        '등급명': '등급',
        '전순 평균가격(원) PreVious SOON': '전순 평균가격(원)',
        '전달 평균가격(원) PreVious MMonth': '전달 평균가격(원)',
        '전년 평균가격(원) PreVious YeaR': '전년 평균가격(원)',
        '평년 평균가격(원) Common Year SOON': '평년 평균가격(원)'
    }, inplace=True)
    test2 = test2[['시점', '품목명', '품종명', '거래단위', '등급', '전순 평균가격(원)', '전달 평균가격(원)', '전년 평균가격(원)', '평년 평균가격(원)', '평균가격(원)']]

    # Combine test1 and test2
    test = pd.concat([test1, test2], axis=0).reset_index(drop=True)

    # Load and preprocess 기상 dataset
    기상 = pd.read_csv(f'./data/test/meta/TEST_기상_{i}.csv')
    기상['주산지 품목명'] = 기상['주산지 품목명'].apply(lambda x: x.replace("마늘", "깐마늘(국산)"))
    기상 = 기상.groupby(['YYYYMMSOON', '주산지 품목명'])[['순 평균상대습도', '순 평균기온', '순 평균풍속', '순 최고기온', '순 최저상대습도', '순 최저기온', '순 강수량', '순 누적 일조시간', '특보 발효 순통계 개수']].mean().reset_index()
    기상.rename(columns={'주산지 품목명': '품목명', 'YYYYMMSOON': '시점'}, inplace=True)

    # Merge test data with 기상
    test = pd.merge(test, 기상, on=['시점', '품목명'], how='left')

    # Extract year, month, and soon, and add seasonal indicators
    test['year'] = test['시점'].apply(lambda x: int(x[:4]))
    test['month'] = test['시점'].apply(lambda x: int(x[4:6]))
    test['soon'] = test['시점'].apply(lambda x: x[-2:])
    test['sin_month'] = np.sin(2 * np.pi * (test['month'] - 1) / 12)
    test['cos_month'] = np.cos(2 * np.pi * (test['month'] - 1) / 12)
    test['is_peak'] = test['month'].apply(lambda x: 1 if x in [11, 12, 1] else 0)
    test['is_bottom'] = test['month'].apply(lambda x: 1 if x in [2, 3, 6, 7] else 0)
    # Use a custom function to create sin_date and cos_date
    test['sin_date'], test['cos_date'] = month_sequence_to_sin_cos(test['month'], test['soon'].map({'상순': 1, '중순': 2, "하순": 3}))
    test.fillna(0, inplace=True)

    # Load and merge additional datasets
    가락도매 = pd.read_csv(f'./data/test/meta/TEST_경락정보_가락도매_{i}.csv')
    전국도매 = pd.read_csv(f'./data/test/meta/TEST_경락정보_전국도매_{i}.csv')
    산지공판장 = pd.read_csv(f'./data/test/meta/TEST_경락정보_산지공판장_{i}.csv')
    중도매 = pd.read_csv(f'./data/test/meta/TEST_중도매_{i}.csv')
    소매 = pd.read_csv(f'./data/test/meta/TEST_소매_{i}.csv')
    
    lst_tr = test['품종명'].unique().tolist()
    
    test = test.merge(process_garak_domae(가락도매, lst_tr), on=['품종명', '시점'], how='left')
    test = test.merge(process_national_domae(전국도매, lst_tr), on=['품종명', '시점'], how='left')
    test = test.merge(process_sanji_gongpanjang(산지공판장, lst_tr), on=['품종명', '시점'], how='left')
    test = test.merge(process_jungdomae(중도매, lst_tr), on=['품종명', '시점'], how='left')
    test = test.merge(process_retail(소매, lst_tr), on=['품종명', '시점'], how='left')

    # Calculate various transaction values
    test['tr_val_1'] = test['평균가격(원)'] - test['평년 평균가격(원)']
    test['tr_val_2'] = test['평균가격(원)'] - test['전년 평균가격(원)']
    test['tr_val_3'] = test['평균가격(원)'] - test['전달 평균가격(원)']
    test['tr_val_4'] = test['평균가격(원)'] - test['전순 평균가격(원)']
    test['tr_val_5'] = test['평년 평균가격(원)'] - test['전년 평균가격(원)']
    test['tr_val_6'] = test['평년 평균가격(원)'] - test['전달 평균가격(원)']
    test['tr_val_7'] = test['평년 평균가격(원)'] - test['전순 평균가격(원)']
    test['tr_val_8'] = test['전년 평균가격(원)'] - test['전달 평균가격(원)']
    test['tr_val_9'] = test['전년 평균가격(원)'] - test['전순 평균가격(원)']
    test['tr_val_10'] = test['전달 평균가격(원)'] - test['전순 평균가격(원)']

    # Fill NaN values with 0
    test.fillna(0, inplace=True)
    
    # Concatenate the current test data into the final DataFrame
    test_total = pd.concat([test_total, test], ignore_index=True)

# Reset index for the final DataFrame
test_total = test_total.reset_index(drop=True)
test = test_total

# Initialize the combined DataFrame
test_total = pd.DataFrame()

for i in range(52):
    # Add padding to the filename if i is less than 10
    padding = "0"
    if i < 10:
        i = padding + str(i)
    
    # Load the first test dataset
    test1 = pd.read_csv(f'./data/test/TEST_{i}_1.csv').drop(columns=['가락시장 품목코드(5자리)'])
    test1['품종명'] = test1['품목(품종)명']
    test1.loc[test1['품목(품종)명'] == "감자 수미", "품목(품종)명"] = "감자"
    test1.loc[test1['품목(품종)명'] == "대파(일반)", "품목(품종)명"] = "대파"
    test1.rename(columns={
        'YYYYMMSOON': '시점',
        '품목(품종)명': '품목명',
        '등급(특 5% 상 35% 중 40% 하 20%)': '등급',
        '전순 평균가격(원) PreVious SOON': '전순 평균가격(원)',
        '전달 평균가격(원) PreVious MMonth': '전달 평균가격(원)',
        '전년 평균가격(원) PreVious YeaR': '전년 평균가격(원)',
        '평년 평균가격(원) Common Year SOON': '평년 평균가격(원)'
    }, inplace=True)
    test1 = test1[['시점', '품목명', '품종명', '거래단위', '등급', '전순 평균가격(원)', '전달 평균가격(원)', '전년 평균가격(원)', '평년 평균가격(원)', '평균가격(원)']]

    # Load the second test dataset
    test2 = pd.read_csv(f'./data/test/TEST_{i}_2.csv').drop(columns=['품목코드 ', '품종코드 '])
    test2['거래단위'] = test2['유통단계별 단위 '].apply(lambda x: str(int(x))) + " " + test2['유통단계별 무게 '].apply(lambda x: str(x))
    test2.drop(columns=['유통단계별 단위 ', '유통단계별 무게 '], inplace=True)
    test2.rename(columns={
        'YYYYMMSOON': '시점',
        '등급명': '등급',
        '전순 평균가격(원) PreVious SOON': '전순 평균가격(원)',
        '전달 평균가격(원) PreVious MMonth': '전달 평균가격(원)',
        '전년 평균가격(원) PreVious YeaR': '전년 평균가격(원)',
        '평년 평균가격(원) Common Year SOON': '평년 평균가격(원)'
    }, inplace=True)
    test2 = test2[['시점', '품목명', '품종명', '거래단위', '등급', '전순 평균가격(원)', '전달 평균가격(원)', '전년 평균가격(원)', '평년 평균가격(원)', '평균가격(원)']]

    # Combine test1 and test2
    test = pd.concat([test1, test2], axis=0).reset_index(drop=True)

    # Load and preprocess 기상 dataset
    기상 = pd.read_csv(f'./data/test/meta/TEST_기상_{i}.csv')
    기상['주산지 품목명'] = 기상['주산지 품목명'].apply(lambda x: x.replace("마늘", "깐마늘(국산)"))
    기상 = 기상.groupby(['YYYYMMSOON', '주산지 품목명'])[['순 평균상대습도', '순 평균기온', '순 평균풍속', '순 최고기온', '순 최저상대습도', '순 최저기온', '순 강수량', '순 누적 일조시간', '특보 발효 순통계 개수']].mean().reset_index()
    기상.rename(columns={'주산지 품목명': '품목명', 'YYYYMMSOON': '시점'}, inplace=True)

    # Merge test data with 기상
    test = pd.merge(test, 기상, on=['시점', '품목명'], how='left')

    # Extract year, month, and soon, and add seasonal indicators
    test['year'] = test['시점'].apply(lambda x: int(x[:4]))
    test['month'] = test['시점'].apply(lambda x: int(x[4:6]))
    test['soon'] = test['시점'].apply(lambda x: x[-2:])
    test['sin_month'] = np.sin(2 * np.pi * (test['month'] - 1) / 12)
    test['cos_month'] = np.cos(2 * np.pi * (test['month'] - 1) / 12)
    test['is_peak'] = test['month'].apply(lambda x: 1 if x in [11, 12, 1] else 0)
    test['is_bottom'] = test['month'].apply(lambda x: 1 if x in [2, 3, 6, 7] else 0)
    # Use a custom function to create sin_date and cos_date
    test['sin_date'], test['cos_date'] = month_sequence_to_sin_cos(test['month'], test['soon'].map({'상순': 1, '중순': 2, "하순": 3}))
    test.fillna(0, inplace=True)

    # Load and merge additional datasets
    가락도매 = pd.read_csv(f'./data/test/meta/TEST_경락정보_가락도매_{i}.csv')
    전국도매 = pd.read_csv(f'./data/test/meta/TEST_경락정보_전국도매_{i}.csv')
    산지공판장 = pd.read_csv(f'./data/test/meta/TEST_경락정보_산지공판장_{i}.csv')
    중도매 = pd.read_csv(f'./data/test/meta/TEST_중도매_{i}.csv')
    소매 = pd.read_csv(f'./data/test/meta/TEST_소매_{i}.csv')
    
    lst_tr = test['품종명'].unique().tolist()
    
    test = test.merge(process_garak_domae(가락도매, lst_tr), on=['품종명', '시점'], how='left')
    test = test.merge(process_national_domae(전국도매, lst_tr), on=['품종명', '시점'], how='left')
    test = test.merge(process_sanji_gongpanjang(산지공판장, lst_tr), on=['품종명', '시점'], how='left')
    test = test.merge(process_jungdomae(중도매, lst_tr), on=['품종명', '시점'], how='left')
    test = test.merge(process_retail(소매, lst_tr), on=['품종명', '시점'], how='left')

    # Calculate various transaction values
    test['tr_val_1'] = test['평균가격(원)'] - test['평년 평균가격(원)']
    test['tr_val_2'] = test['평균가격(원)'] - test['전년 평균가격(원)']
    test['tr_val_3'] = test['평균가격(원)'] - test['전달 평균가격(원)']
    test['tr_val_4'] = test['평균가격(원)'] - test['전순 평균가격(원)']
    test['tr_val_5'] = test['평년 평균가격(원)'] - test['전년 평균가격(원)']
    test['tr_val_6'] = test['평년 평균가격(원)'] - test['전달 평균가격(원)']
    test['tr_val_7'] = test['평년 평균가격(원)'] - test['전순 평균가격(원)']
    test['tr_val_8'] = test['전년 평균가격(원)'] - test['전달 평균가격(원)']
    test['tr_val_9'] = test['전년 평균가격(원)'] - test['전순 평균가격(원)']
    test['tr_val_10'] = test['전달 평균가격(원)'] - test['전순 평균가격(원)']

    # Fill NaN values with 0
    test.fillna(0, inplace=True)
    
    # Concatenate the current test data into the final DataFrame
    test_total = pd.concat([test_total, test], ignore_index=True)

# Reset index for the final DataFrame
test_total = test_total.reset_index(drop=True)
test = test_total
## Test Data Split
## 양파
양파_col = ['시점', '품목명', '품종명', '거래단위', '등급', '전순 평균가격(원)', '전달 평균가격(원)',
       '전년 평균가격(원)', '평년 평균가격(원)', '평균가격(원)', 'year', 'month', 'soon',
       'sin_month', 'cos_month', 'is_peak', 'is_bottom']
test_양파 = test[양파_col]

## 깐마늘
깐마늘_col = ['시점', '품목명', '품종명', '거래단위', '등급', '전순 평균가격(원)', '전달 평균가격(원)',
       '전년 평균가격(원)', '평년 평균가격(원)', '평균가격(원)', '순 평균상대습도', '순 평균기온', '순 평균풍속',
       '순 최고기온', '순 최저상대습도', '순 최저기온', '순 강수량', '순 누적 일조시간', '특보 발효 순통계 개수',
       'year', 'month', 'soon', 'sin_month', 'cos_month', 'sin_date',
       'cos_date']
test_깐마늘 = test[깐마늘_col]

## 배추
배추_col = ['시점', '품목명', '품종명', '거래단위', '등급', '전순 평균가격(원)', '전달 평균가격(원)',
       '전년 평균가격(원)', '평년 평균가격(원)', '평균가격(원)', 
       'year', 'month', 'soon', 'sin_month', 'cos_month','가락_평균가격', '가락_전순가격', '가락_전달가격',
       '가락_전년가격', '가락_평년가격', '전국_총반입량', '전국_총거래금액', '전국_평균가', '전국_고가',
       '중가(60%) 평균가 ', '전국_저가', '전국_중간가', '전국_최저가', '전국_최고가', '전국_경매건수',
       '전국_전순가격', '전국_전달가격', '전국_전년가격', '전국_평년가격', '산지_총반입량', '산지_총거래금액',
       '산지_평균가','산지_중간가', '산지_최저가', '산지_최고가', '산지_경매건수', '산지_전순가격', '산지_전달가격',
       '산지_전년가격', '산지_평년가격', '중도매_평균가격(원)', '중도매_전순가격', '중도매_전달가격', '중도매_전년가격',
       '중도매_평년가격', '소매_평균가격(원)', '소매_전순가격', '소매_전달가격', '소매_전년가격', '소매_평년가격',
       'tr_val_1', 'tr_val_2', 'tr_val_3', 'tr_val_4', 'tr_val_5', 'tr_val_6',
       'tr_val_7', 'tr_val_8', 'tr_val_9', 'tr_val_10']
test_배추 = test[배추_col]

## 사과
사과_col = ['시점', '품목명', '품종명', '거래단위', '등급', '전순 평균가격(원)', '전달 평균가격(원)',
       '전년 평균가격(원)', '평년 평균가격(원)', '평균가격(원)', '순 평균상대습도', '순 평균기온', '순 평균풍속',
       '순 최고기온', '순 최저상대습도', '순 최저기온', '순 강수량', '순 누적 일조시간', '특보 발효 순통계 개수',
       'year', 'month', 'soon', 'sin_month', 'cos_month','전국_총반입량', '전국_총거래금액', '전국_평균가', '전국_고가',
       '중가(60%) 평균가 ', '전국_저가', '전국_중간가', '전국_최저가', '전국_최고가', '전국_경매건수',
       '전국_전순가격', '전국_전달가격', '전국_전년가격', '전국_평년가격']
test_사과 = test[사과_col]
## Inference

submit = pd.read_csv('./data/sample_submission.csv')

############################### 양파 ##################################
result_series_dict = {}

test_total = pd.DataFrame()
lst = []

k, v = "양파", 2

df = test_양파[['전달 평균가격(원)', '전년 평균가격(원)', 
        '평년 평균가격(원)', '평균가격(원)']]

input_size = 9
output_size = 0
block_size = 9

X_test, _ = create_sliding_block_dataset_version2(df, input_size, output_size, block_size)

X_test = pd.DataFrame([x.flatten() for x in X_test])

X_test = pd.concat([test_양파.iloc[::9][[
                                        '품목명', '품종명', '거래단위', '등급', 
                                        'soon',
                                        'year', 'month', 
                                        'sin_month', 'cos_month', 
                                        'is_peak', 'is_bottom']].reset_index(drop=True), 
                  X_test.reset_index(drop=True)], axis=1)

for i in range(52):
    tt = pd.DataFrame(X_test.iloc[v+10*i]).T
    test_total = pd.concat([test_total,tt])

for i in range(0,36):
    test_total[i] = test_total[i].astype(float)

test_total['year'] = test_total['year'].astype(float)
test_total['month'] = test_total['month'].astype(float)
test_total['sin_month'] = test_total['sin_month'].astype(float)
test_total['cos_month'] = test_total['cos_month'].astype(float)
test_total['is_peak'] = test_total['is_peak'].astype(float)
test_total['is_bottom'] = test_total['is_bottom'].astype(float)

test_total.drop(columns = ['품목명','품종명','거래단위','등급'], inplace=True)

with open('encoding/양파_labelencoder.pkl', 'rb') as f:
    le_dict = pickle.load(f)

for col in ['soon']:
    test_total[col] = le_dict[col].transform(test_total[col])

test_total.columns = [str(i) for i in test_total.columns]

model = joblib.load('models/양파_model.pkl')
양파_pred = model.predict(test_total).reshape(156,)

submit['양파'] = 양파_pred

############################### 깐마늘 ##################################

result_series_dict = {}

test_total = pd.DataFrame()
lst = []

k, v = "깐마늘(국산)", 6

df = test_깐마늘[['평균가격(원)']]

input_size = 9
output_size = 0
block_size = 9

X_test, _ = create_sliding_block_dataset_version2(df, input_size, output_size, block_size)

X_test = pd.DataFrame([x.flatten() for x in X_test])

X_test = pd.concat([test_깐마늘.iloc[::9][['품목명', '품종명', '거래단위', '등급', 'soon', '전순 평균가격(원)', '전달 평균가격(원)', '전년 평균가격(원)',
                                                   '순 평균상대습도', '순 평균기온', '순 평균풍속', '순 최고기온', '순 최저상대습도', 
            '순 최저기온', '순 강수량', '순 누적 일조시간', '특보 발효 순통계 개수',
                                                        'year', 'month', 
                                                        'sin_month', 'cos_month', 
                                                        'sin_date', 'cos_date'
                                                      ]].reset_index(drop=True), 
                  X_test.reset_index(drop=True)], axis=1)

for i in range(52):
    tt = pd.DataFrame(X_test.iloc[v+10*i]).T
    test_total = pd.concat([test_total,tt])

for i in range(0,9):
    test_total[i] = test_total[i].astype(float)

test_total['전순 평균가격(원)'] = test_total['전순 평균가격(원)'].astype(float)
test_total['전달 평균가격(원)'] = test_total['전달 평균가격(원)'].astype(float)
test_total['전년 평균가격(원)'] = test_total['전년 평균가격(원)'].astype(float)
test_total['순 평균상대습도'] = test_total['순 평균상대습도'].astype(float)
test_total['순 평균기온'] = test_total['순 평균기온'].astype(float)
test_total['순 평균풍속'] = test_total['순 평균풍속'].astype(float)
test_total['순 최고기온'] = test_total['순 최고기온'].astype(float)
test_total['순 최저상대습도'] = test_total['순 최저상대습도'].astype(float)
test_total['순 최저기온'] = test_total['순 최저기온'].astype(float)
test_total['순 강수량'] = test_total['순 강수량'].astype(float)
test_total['순 누적 일조시간'] = test_total['순 누적 일조시간'].astype(float)
test_total['특보 발효 순통계 개수'] = test_total['특보 발효 순통계 개수'].astype(float)
test_total['year'] = test_total['year'].astype(float)
test_total['month'] = test_total['month'].astype(float)
test_total['sin_month'] = test_total['sin_month'].astype(float)
test_total['cos_month'] = test_total['cos_month'].astype(float)
test_total['sin_date'] = test_total['sin_date'].astype(float)
test_total['cos_date'] = test_total['cos_date'].astype(float)

for i in range(9):  
    for j in range(i+1, 9):  
        test_total[f'Diff_{i}_{j}'] = test_total[i] - test_total[j]

with open('encoding/깐마늘_labelencoder.pkl', 'rb') as f:
    le_dict = pickle.load(f)
    
for col in ['품목명','품종명','거래단위','등급', 'soon']:
    test_total[col] = le_dict[col].transform(test_total[col])

test_total.columns = [str(i) for i in test_total.columns]

with open('scalers/깐마늘_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

test_preds = []

model = joblib.load('models/깐마늘_model.pkl')
pred = model.predict(test_total)
깐마늘_pred = scaler.inverse_transform(pred).reshape(156,)

submit['깐마늘(국산)'] = 깐마늘_pred

############################### 배추 ##################################

test_total = pd.DataFrame()
lst = []

k, v = "배추", 0

df = test_배추['평균가격(원)']

input_size = 9
output_size = 0
block_size = 9

X_test, _ = create_sliding_block_dataset(df, input_size, output_size, block_size)

X_test = pd.DataFrame([x.flatten() for x in X_test])

X_test = pd.concat([test_배추.iloc[::9][['품목명','품종명','거래단위','등급','전순 평균가격(원)','전달 평균가격(원)','전년 평균가격(원)','평년 평균가격(원)',
                                               'year', 'month', 'soon','sin_month','cos_month', #
                                                    '가락_평균가격', '가락_전순가격', '가락_전달가격',
  '가락_전년가격', '가락_평년가격', '전국_총반입량', '전국_총거래금액', '전국_평균가', '전국_고가',
    '중가(60%) 평균가 ', '전국_저가', '전국_중간가', '전국_최저가', '전국_최고가', '전국_경매건수',
    '전국_전순가격', '전국_전달가격', '전국_전년가격', '전국_평년가격', 
                                                    '산지_총반입량', '산지_총거래금액',
    '산지_평균가', '산지_중간가', '산지_최저가', '산지_최고가', '산지_경매건수', '산지_전순가격', '산지_전달가격','산지_전년가격', '산지_평년가격',
 '중도매_평균가격(원)', '중도매_전순가격', '중도매_전달가격', '중도매_전년가격','중도매_평년가격',
   '소매_평균가격(원)', '소매_전순가격', '소매_전달가격', '소매_전년가격', '소매_평년가격',
       'tr_val_1','tr_val_2','tr_val_3','tr_val_4','tr_val_5','tr_val_6','tr_val_7','tr_val_8','tr_val_9','tr_val_10'
]].reset_index(drop=True), 
                  X_test.reset_index(drop=True)], axis=1)

columns_to_convert = ['가락_평균가격', '가락_전순가격', '가락_전달가격',
                              '가락_전년가격', '가락_평년가격', '전국_총반입량', '전국_총거래금액', '전국_평균가', '전국_고가',
                  '중가(60%) 평균가 ', '전국_저가', '전국_중간가', '전국_최저가', '전국_최고가', '전국_경매건수',
                  '전국_전순가격', '전국_전달가격', '전국_전년가격', '전국_평년가격',
        '산지_총반입량', '산지_총거래금액',
                    '산지_평균가', '산지_중간가', '산지_최저가', '산지_최고가', '산지_경매건수', '산지_전순가격', '산지_전달가격','산지_전년가격', '산지_평년가격',
                     '중도매_평균가격(원)', '중도매_전순가격', '중도매_전달가격', '중도매_전년가격','중도매_평년가격',
         '소매_평균가격(원)', '소매_전순가격', '소매_전달가격', '소매_전년가격', '소매_평년가격',
 'tr_val_1','tr_val_2','tr_val_3','tr_val_4','tr_val_5','tr_val_6','tr_val_7','tr_val_8','tr_val_9','tr_val_10']

te = X_test

for i in range(0,52):
    X_test = pd.DataFrame(te.iloc[v+10*i]).T
    test_total = pd.concat([test_total,X_test],axis=0)
    
for i in range(0,9):
    test_total[i] = test_total[i].astype(float)

test_total['평년 평균가격(원)'] = test_total['평년 평균가격(원)'].astype(float)
test_total['전순 평균가격(원)'] = test_total['전순 평균가격(원)'].astype(float)
test_total['전달 평균가격(원)'] = test_total['전달 평균가격(원)'].astype(float)
test_total['전년 평균가격(원)'] = test_total['전년 평균가격(원)'].astype(float)
test_total['year'] = test_total['year'].astype(float)
test_total['month'] = test_total['month'].astype(float)
test_total['sin_month'] = test_total['sin_month'].astype(float)
test_total['cos_month'] = test_total['cos_month'].astype(float)
test_total[columns_to_convert] = test_total[columns_to_convert].astype(float)

for i in range(9):  
    for j in range(i+1, 9):  
        test_total[f'Diff_{i}_{j}'] = test_total[i] - test_total[j]

# 가격 변화율 (Price Change Rate)
test_total['가락_전순가격_변화율'] = (test_total['가락_평균가격'] - test_total['가락_전순가격']) / test_total['가락_전순가격']
test_total['가락_전달가격_변화율'] = (test_total['가락_평균가격'] - test_total['가락_전달가격']) / test_total['가락_전달가격']
test_total['가락_전년가격_변화율'] = (test_total['가락_평균가격'] - test_total['가락_전년가격']) / test_total['가락_전년가격']
test_total['가락_평년가격_변화율'] = (test_total['가락_평균가격'] - test_total['가락_평년가격']) / test_total['가락_평년가격']

# 전국 평균 가격 대비 가락 평균 가격 비율
test_total['전국대비_가락_평균가격_비율'] = test_total['가락_평균가격'] / test_total['전국_평균가']

# 전국 총 반입량 대비 산지 총 반입량 비율
test_total['전국대비_산지_반입량_비율'] = test_total['산지_총반입량'] / test_total['전국_총반입량']

# 전국 총 거래금액 대비 산지 총 거래금액 비율
test_total['전국대비_산지_거래금액_비율'] = test_total['산지_총거래금액'] / test_total['전국_총거래금액']

# 중도매 및 소매 가격 비교
test_total['중도매_소매_평균가격_비율'] = test_total['중도매_평균가격(원)'] / test_total['소매_평균가격(원)']

# 산지와 전국 최저가 비교
test_total['산지_전국_최저가_비율'] = test_total['산지_최저가'] / test_total['전국_최저가']

# 전국 경매건수 대비 산지 경매건수 비율
test_total['전국대비_산지_경매건수_비율'] = test_total['산지_경매건수'] / test_total['전국_경매건수']

# 전국 대비 중도매 전순가격 비율
test_total['전국대비_중도매_전순가격_비율'] = test_total['중도매_전순가격'] / test_total['전국_전순가격']

# 소매와 중도매 전달가격 비교
test_total['소매_중도매_전달가격_비율'] = test_total['소매_전달가격'] / test_total['중도매_전달가격']

# 산지 중간가 대비 전국 중간가 비율
test_total['산지_전국_중간가_비율'] = test_total['산지_중간가'] / test_total['전국_중간가']

test_total = test_total.replace([float('inf'), float('-inf')], 0)
test_total.fillna(0, inplace=True)

with open('encoding/배추_labelencoder.pkl', 'rb') as f:
    le_dict = pickle.load(f)
    
for col in ['품목명','품종명','거래단위','등급', 'soon']:
    test_total[col] = le_dict[col].transform(test_total[col])

test_total.columns = [str(i) for i in test_total.columns]

with open('scalers/배추_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

test_preds = []

model_xgb = joblib.load('models/배추_model_xgb.pkl')
pred1 = model_xgb.predict(test_total)
pred1 = scaler.inverse_transform(pred1)
model_et = joblib.load('models/배추_model_et.pkl')
pred2 = model_et.predict(test_total)
pred2 = scaler.inverse_transform(pred2)
pred = (pred1 + pred2)/2
배추_pred = pred.reshape(156,)

submit['배추'] = 배추_pred

############################### 사과 ##################################

test_total = pd.DataFrame()
lst = []

k, v = "사과", 8

df = test_사과[['평균가격(원)']]

input_size = 9
output_size = 0
block_size = 9

X_test, _ = create_sliding_block_dataset(df, input_size, output_size, block_size)

X_test = pd.DataFrame([x.flatten() for x in X_test])

X_test = pd.concat([test_사과.iloc[::9][['품목명','품종명','거래단위','등급','전순 평균가격(원)','전달 평균가격(원)','전년 평균가격(원)','평년 평균가격(원)',
                                                   '순 평균상대습도', '순 평균기온', '순 평균풍속', '순 최고기온', '순 최저상대습도', '순 최저기온', '순 강수량',
                                              '순 누적 일조시간', '특보 발효 순통계 개수', 
                                                'year', 'month', 'soon','sin_month','cos_month', #
                                                '전국_총반입량', '전국_총거래금액', '전국_평균가', '전국_고가',
    '중가(60%) 평균가 ', '전국_저가', '전국_중간가', '전국_최저가', '전국_최고가', '전국_경매건수',
    '전국_전순가격', '전국_전달가격', '전국_전년가격', '전국_평년가격', 


   ]].reset_index(drop=True), 
                  X_test.reset_index(drop=True)], axis=1)

columns_to_convert = [
        '전국_총반입량', '전국_총거래금액', '전국_평균가', '전국_고가',
                  '중가(60%) 평균가 ', '전국_저가', '전국_중간가', '전국_최저가', '전국_최고가', '전국_경매건수',
                  '전국_전순가격', '전국_전달가격', '전국_전년가격', '전국_평년가격',
                    ]

te = X_test

for i in range(0,52):
    X_test = pd.DataFrame(te.iloc[v+10*i]).T
#     print(X_test)
    test_total = pd.concat([test_total,X_test],axis=0)
    
for i in range(0,9):
    test_total[i] = test_total[i].astype(float)


test_total['평년 평균가격(원)'] = test_total['평년 평균가격(원)'].astype(float)
test_total['전순 평균가격(원)'] = test_total['전순 평균가격(원)'].astype(float)
test_total['전달 평균가격(원)'] = test_total['전달 평균가격(원)'].astype(float)
test_total['전년 평균가격(원)'] = test_total['전년 평균가격(원)'].astype(float)
test_total['순 평균상대습도'] = test_total['순 평균상대습도'].astype(float)
test_total['순 평균기온'] = test_total['순 평균기온'].astype(float)
test_total['순 평균풍속'] = test_total['순 평균풍속'].astype(float)
test_total['순 최고기온'] = test_total['순 최고기온'].astype(float)
test_total['순 최저상대습도'] = test_total['순 최저상대습도'].astype(float)
test_total['순 최저기온'] = test_total['순 최저기온'].astype(float)
test_total['순 강수량'] = test_total['순 강수량'].astype(float)
test_total['순 누적 일조시간'] = test_total['순 누적 일조시간'].astype(float)
test_total['특보 발효 순통계 개수'] = test_total['특보 발효 순통계 개수'].astype(float)
test_total['year'] = test_total['year'].astype(float)
test_total['month'] = test_total['month'].astype(float)
test_total['sin_month'] = test_total['sin_month'].astype(float)
test_total['cos_month'] = test_total['cos_month'].astype(float)
test_total[columns_to_convert] = test_total[columns_to_convert].astype(float)

with open('encoding/사과_labelencoder.pkl', 'rb') as f:
    le_dict = pickle.load(f)
    
for col in ['품목명','품종명','거래단위','등급', 'soon']:
    test_total[col] = le_dict[col].transform(test_total[col])

test_total.columns = [str(i) for i in test_total.columns]



with open('scalers/사과_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

test_preds = []

for i in range(1, 6):
    
    # 모델을 불러온 후 바로 예측을 실행
    model_xgb = joblib.load(f'models/사과_model_xgb_{i}.pkl')
    pred1 = model_xgb.predict(test_total)
    pred1 = scaler.inverse_transform(pred1)
    
    model_et = joblib.load(f'models/사과_model_et_{i}.pkl')
    pred2 = model_et.predict(test_total)
    pred2 = scaler.inverse_transform(pred2)

    # 예측값의 평균을 계산
    pred = (pred1 + pred2) / 2
    test_preds.append(pred)
# test_preds 리스트의 평균값 계산
test_preds = np.mean(test_preds, axis=0)
사과_pred = test_preds.reshape(156,)

submit['사과'] = 사과_pred

################################# 제출 ######################################

submit.to_csv('submission_최종/inference1_submission.csv',index=False)

requirement.txt 파일에 기재된 version_1 / version_2에 해당하는 os 및 라이브러리 버전으로 순서대로 실행시켜주시면 됩니다.
크게 두 가지 환경으로 구성하여 품목 별 예측을 진행하였습니다.

[ Preprocessing - Train - Inference 시 ]

Version_1 환경 <양파, 깐마늘, 배추, 사과>
1. preprocessing_1.py
2. train_1.py
3. inference_1.py

Version_2 환경 <배, 무, 건고추, 감자, 대파, 상추>
4. preprocessing_2.py
5. train_2.py
6. inference_2.py

위와같은 순서로 실행하게 되면 함께 제출한 전처리 데이터, pkl, 최종 submission을 생성할 수 있습니다.




[ Inference만 진행 시 ]

Version_1 환경 <양파, 깐마늘, 배추, 사과>
1. inference_1.py

Version_2 환경 <배, 무, 건고추, 감자, 대파, 상추>
2. inference_2.py

위의 순서로 실행하면 최종 submission을 생성할 수 있습니다.

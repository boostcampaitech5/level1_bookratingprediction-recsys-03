![header](https://capsule-render.vercel.app/api?type=wave&color=0:EEFF00,100:a82da8&height=300&section=header&text=Book%20Rating%20Prediction%20Project&fontSize=50)

:computer: RecSys 3조 
:blush: 김현정
:laughing: 민현지
:smirk: 황찬웅
:smiley: 이준원

## Project Configuration

<img width="367" alt="image" src="https://user-images.githubusercontent.com/69078499/234792671-445f9767-14b8-4f31-9bba-3539f5fac1a7.png">



<br><br>

## 프로젝트 개요
### 1. 프로젝트 주제 및 목표

&ensp;이번 대회는 책과 관련된 정보, 소비자의 정보, 평점 데이터를 활용하여 각 사용자가 책에 대해 얼마나 평점을 줄지 예측하는 개인화된 상품 추천 대회입니다. 대회에서는 정형 데이터부터 이미지, 텍스트 데이터까지 다양한 데이터를 활용하여 추천 시스템의 스킬을 적용할 수 있습니다. 최종 결과물은 .csv파일로 제출되며, 평가 데이터는 60%가 Public 점수 계산에 사용되고, 나머지 40%는 Private 결과 계산에 사용됩니다. RMSE를 사용하여 최종 리더보드의 성능을 수치화하였습니다.   

<br>

### 2. 프로젝트 개발 환경
v100 서버, python   
Notion, Github, Zoom, Slack, 카카오톡

<br><br>

## 프로젝트 팀 구성 및 역할
<img width="600" alt="image" src="https://user-images.githubusercontent.com/69078499/235096277-d3cc05ba-cc02-4143-a9d4-c1440b166f74.png">  

<br><br>

## 프로젝트 수행 내용 및 결과
### 1. EDA 및 데이터 전처리
#### 데이터
- users.csv
    - 고객의 정보를 담고 있는 메타데이터   
    - age의 결측치가 40% 이상   
    - city 결측치에 비해 country, state의 결측치가 많음

- books.csv
    - 책의 정보를 담고 있는 메타데이터
    - language와 category의 결측치가 40% 이상
    - 비정형 데이터인 summary와 img_url, img_path는 사용하지 않음

#### users 결측치 처리: age
- 실험 1 : mean 값으로 결측치 채우기 - 2.4605
- 실험 2 : median 값으로 결측치 채우기 - 2.4605
- 실험 3 : 분포를 고려하여 random sampling으로 채우기 - 2.4611

- 결과: Age map을 이용하여 indexing을 진행해주기 때문에 각 방법 별로 rmse 값에 큰 차이 존재하지 않아서 가장 구현이 용이한 mean 값을 이용하여 결측치를 채우기로 결정

#### users 결측치 처리: location
- 아이디어
    - city 정보를 이용해서 state, country  결측치 채우

- 전처리 과정
    - city, state, country로 location을 분해하여 feature를 재구성
    - 가장 작은 단위인 city가 null이 아닌 경우에 비어 있는 state와 country의 값을 city 정보를 이용해 채움
    - users 데이터셋에서 city에 해당하는 state와 country 정보가 이미 있는 경우에는 해당하는 정보 중 최빈값으로 채움
    - 해당 방법으로 채울 수 없는 경우, 2개 이상 존재하는 city에 한해서 실제 지리적인 배경 지식을 이용하여 하드 코딩

#### books 결측치 처리: language
- 아이디어
    - ISBN의 1 ~ 5자리 prefix 에 따라서 country, region, language area가 정해짐
    - range별로 prefix의 길이가 정해짐
    - ex) 0, 1, 2, 3, 4, 7로 시작 -> 한 자리 prerfix, 8로 시작 -> 두 자리 prefix

- 전처리 과정 
    - 주어진 book data를 활용해 해당 prefix를 가지는 책 중 가장 많은 빈도로 나타나는 language로 결측치를 채움
    - 1로 채우지 못한 결측치의 경우 최빈값인 ‘en’으로 채

- 결과: RMSE 2.4605에서 2.4496으로 약간의 성능 향상


#### books 결측치 처리: category
- 아이디어 
    - book_title을 이용해 책의 category 결측치를 채우자

- 전처리 과정 
    - category 값의 존재 유무로 train data와 test data로 split 
    - book_title을 pretrained BERT 모델을 이용해 embedding
    - train data의 embedding vector를 이용해 category를 예측하는 KNN을 학습
    - 학습된 KNN 모델을 이용해 test data의 category를 예측
    - 예측값으로 category 결측치를 채움

- 결과: 2.1124에서 2.1113로 약간의 성능 향상


<br>

### 2. 사용한 모델
#### 사용한 모델 종류
#### Catboost 모델
#### FFM + DCN 모델

<br>

### 3. 앙상블
가중치를 곱하여 평균을 취하는 stacking 방식으로 앙상블을 진행하였습니다.

<img width="600" alt="image" src="https://user-images.githubusercontent.com/69078499/235096581-9df1f17b-e489-4b85-bf42-3059abfc3352.png">
개별 모델 성능 (민트), 앙상블 성능 (파랑)
<br>
SVD 모델을 이용한 앙상블 시에 성능이 떨어지는 것을 확인하였습니다. SVD의 경우 일부 feature가 가지는 특징을 drop하게 되어 고려하게 되는 feature가 적어지는데, 이 때문에 다른 모델들과 stacking 방식으로 앙상블 진행했을 때 성능이 떨어지는 것으로 보여집니다.
<br>
가중치를 바꾸어가며 실험을 진행하였고 최종적으로 CatBoost 모델과 FFMDCN 하이브리드 모델을 1:1로 앙상블한 모델을 최종 솔루션으로 채택하게 되었습니다.

<br>
### 4. 리더보드 순위 및 성능 평가
Public – 2등, RMSE 2.1099
<img width="599" alt="image" src="https://user-images.githubusercontent.com/69078499/235097134-4eccebdb-6385-4d5d-9110-f943b5810600.png">
<br>
Private – 2등, RMSE 2.1074
<img width="599" alt="image" src="https://user-images.githubusercontent.com/69078499/235097355-5a9af50c-d50f-4638-ad7a-696900660e2d.png">
<br>

## 결론 및 개선 방안

### 마스터 클래스 피드백
- 데이터:
    - missing value, outlier 부분도 data 관점에서 살펴보기
    - 결측치를 모델기반이나 유사도 기반으로 채워보기
    - 피처 엔지니어링을 통해 추가 변수도 활용해보기 - 딥러닝 성능 개선
- 모델:
    - 이번 프로젝트에서는 FFM 분해 과정에서 생기는 latent vector를 넣는 방법을 사용
    - concatenate skim gram, residual을 이용한 fusion 방법론도 좋다 (멀티 모달도 요즘 화두)
    - attenion layer도 적용해서 성능을 높이는 것도 하나의 방법
- CatBoost를 CPU를 이용해 돌릴 경우 속도는 느리지만 성능 개선이 크게 이루어질 수도 있다.
- Optuna 라이브러리 사용시 최소값이 아닌 극솟값으로 수렴하게 될 수 있다.
따라서, Optuna를 사용해서 얻은 하이퍼파라미터가 가장 좋다고 단언할 수는 없다.

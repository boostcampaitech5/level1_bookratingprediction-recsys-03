![header](https://capsule-render.vercel.app/api?type=wave&color=0:EEFF00,100:a82da8&height=300&section=header&text=Book%20Rating%20Prediction%20Project&fontSize=50)

:computer: RecSys 3조 
:blush: 김현정
:laughing: 민현지
:smirk: 황찬웅
:smiley: 이준원

## Create with respective directories 


## Project Configuration


## Final Code


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
김현정 - EDA, 전처리, 모델링(AutoRec, LightGBM, XGBoost), 팀 깃헙 관리   
민현지 - EDA, 전처리, 모델링(MF, LightFM), 발표   
황찬웅 - EDA, 모델링(Catboost, LightGBM, HybridModel (FFM+DCN))   
이준원 - EDA, 전처리, 모델링(SVD), 팀 노션 관리   

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

#### users 결측치 처리: location

#### books 결측치 처리: language

#### books 결측치 처리: category

<br>

### 2. 사용한 모델
#### 사용한 모델 종류
#### Catboost 모델
#### FFM + DCN 모델

<br>

### 3. 앙상블


<br>

### 4. 결과

<br><br>

## 결론 및 개선 방안

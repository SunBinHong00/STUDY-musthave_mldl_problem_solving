# Musthave_mldl_problem_solving Study

- 머신러닝 딥러닝 문제해결 전략 스터디

## 스터디 목적

- 캐글 입문

## 스터디 방식

1. 머신러닝 딥러닝 문제해결 전략 교재로 캐글 프로세스 이해와 노트북 필사
  - 총 651페이지
2. 반복되는 작업 ToolBox 레포에 정리

## 목차

※1∼5장은 캐글에 대한 간략한 소개, 머신러닝 모델, 시각화에 관한 내용이므로 다루지 않겠음

6장. 경진대회 자전거 대여 수요 예측 (p.167 ~ p.236)

    - 회귀문제, simple EDA, simple feature engineering, baseline model, LinearRegression
    - Ridge, Rasso, GridSearchCV, RandomForestRegressor, rmsle

7장. 경진대회 범주형 데이터 이진분류 (~p.296)

    - 분류문제, LogisticRegression, OrdinalEncoder, OnehotEncoder, GridSearchCV, scipy.sparse
    - matplotlib.gridspec, get_crosstab, pointplot, CategoricalDtype

8장. 경진대회 안전 운전자 이진분류 (~p.369)

    - 분류문제, missingno, 피처별 타겟 비율과 신뢰구간 시각화, 유효한 피처 선택법, OOF, LightGBM
    - feature_eengineering, bayesian_optimization, XGBoost, ensemble, 
    
9장. 경진대회 향후 판매량 예측 회귀 (~p.296)

    - 회귀문제, groupby, down_castiong, garbage_collection, LightGBM, feature_engineering(이번 장의 전부)
    - time lag feature, bayesian_optimization


- 2022.08.05 ~ 08.06
  - 6장 교재 읽기, 캐글 노트북 클론 코딩
    - 처음 보는 데이터셋에 대해 피처 요약 함수에 대한 필요성을 느끼고 함수 구현에 포커스를 두고 공부함
    - 타겟데이터의 비율에 따른 피처 중요도에 대한 통계학적 이해가 부족하다고 생각해 따로 통계공부를 함
  
- 2022.08.07 ~ 08.10
  - 7장 교재 읽기, 캐글 노트북 클론 코딩 
    - Model selection (K-fold, Stratified K-fold, auc_roc, GridSearchCV), 데이터 전처리
    - Model Selection에 집중하여 복습함
    - 타겟과 피처와의 관계를 시각화 하는 함수를 구현하는대에 집중하여 공부함
- 2022.08.11 ~ 08.14
  - 8장 교재 읽기, 캐글 노트북 클론 코딩
    - LightGBM, XGBoost모델의 원리에 집중하여 공부함, 책에서는 각각의 파라미터에 대해 깊게 다루지않아 따로 공부함
    - OOF, bayesian_optimization의 원리는 알지만 코드를 짜는 구성에 익숙해지려 노력함
    - 전처리 과정중 인코딩시에 변수의 종류 각각에 대해서 직관적인 느낌만 알고 있어 책을 보지 않고 직접 인코딩하려니 어려움이 있었음
- 2022.08.15 ~ 08.19
 - 9장 교재 읽기, 캐글 노트북 클론 코딩, 캐글 notebook과 discussion탭을 활용하여 다양한 시도
    - 창의적인 피처엔지니어링이 이번 대회의 성적에 결정적인 기술이었음 따라서 discussion탭을 통해 다양한 의견들을 수용해서 선택지를 늘렸음
    - 데이터셋의 크기가 커지면서 메모리관리에 대한 필요성이 점점 커지고있음
    - 데이터셋의 피처가 많아질 수록 시각화가 더 많은 insight를 도출함을 느낌
- 2022.08.19 ~
 - 캐글대회 Tabular Playground Series - Aug 2022참가 https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/overview
    - 10장부터는 딥러닝대회에 관한 내용이라 지금까지 공부한 내용을 실전에서 다루고 익숙해지고 넘어가고싶어 이주남은 대회를 통해 내 것으로 만들고 넘어가기로함
    

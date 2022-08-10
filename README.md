# Musthave_mldl_problem_solving Study

- 머신러닝 딥러닝 문제해결 전략 스터디

## 스터디 목적

- 캐글 입문

## 스터디 방식

1. 머신러닝 딥러닝 문제해결 전략 교재로 캐글 프로세스 이해와 노트북 필사
  - 총 651페이지
2. 반복되는 작업 ToolBox 레포에 정리

## 목차

※1∼5장은 캐글에 대한 간략한 소개이므로 따로 다루지 않을 예정

6장. 경진대회 자전거 대여 수요 예측 (p.167 ~ p.236)

    - 회귀문제, simple EDA, simple feature engineering, baseline model, LinearRegression, Ridge, Rasso, GridSearchCV, RandomForestRegressor, rmsle

7장. 경진대회 범주형 데이터 이진분류 (~p.296)

    - 분류문제, LogisticRegression, OrdinalEncoder, OnehotEncoder, GridSearchCV, scipy.sparse,  matplotlib.gridspec, get_crosstab, pointplot, CategoricalDtype




- 2022.08.05 ~ 08.06
  - 6장 교재 읽기, 캐글 노트북 클론 코딩
    - 처음 보는 데이터셋에 대해 피처 요약 함수에 대한 필요성을 느끼고 함수 구연에 포커스를 두고 공부함
    - 타겟데이터의 비율에 따른 피처 중요도에 대한 통계학적 설명이 부족하다고 생각해 따로 통계공부를 함
  
- 2022.08.07 ~ 08.10
  - 7장 교재 읽기, 캐글 노트북 클론 코딩 
    - Model selection (K-fold, Stratified K-fold, auc_roc, GridSearchCV), 데이터 전처리
    - Model Selection에 집중하여 복습함
    - 타겟과 피처와의 관계를 시각화 하는 함수를 구현하는대에 집중하여 공부함
  
- 2019.11.11 ~ 11.12
  - 3장  평가 강의 완강, 교재 읽기, 코드 리뷰

    - Accuracy, Confusion Matrix, Precison and Recall, F1 Score, ROC/AUC


- 2019.11.13 ~ 11.16

  - 4장  분류 강의 완강, 교재 읽기, 코드 리뷰

    - Decision Tree, Ensemble(voting, bagging, boosting), GBM, XGBoost, LightBoost, Over/Under Sampling(SMOTE), Stacking
  - 모델 학습 코드에 집중하여 복습함
- 2019.11.17 ~ 11.21

  - 5장  회귀 강의 완강, 교재 읽기, 코드 리뷰
  
    - Gradient Descent, Stochastic Gradient Descent, Linear Regression, Polynomial Regression, Regularized Linear Models (Ridge, Lasso, ElasticNet), Logistic Regression, Tree Regression, Preprocessing(Scaling, Log Transformation, Feature Encoding), Mixed Model Prediction
  - 각 휘귀 모델 별 차이점 숙지
  - 스케일링, 인코딩, 아웃라이어 제거, 하이퍼 파라미터 튜닝에 따라 예측 성능이 향상되는 흐름 복습
- 2019.11.22 ~ 11.24
  - 6장  차원 축소
    - 차원 축소 (피쳐 선택, 피쳐 추출), PCA(Principal Component Analysis), LDA(Linear Discriminant Analysis), SVD(Singular Value Decomposition), Truncated SVD, NMF(Non-Negative Matrix Fatorization)
  - 각 차원 축소 기법 별 선형 대수적 의미를 최대한 이해하며 학습
- 2019.11.25 ~ 11.28
  - 7장  군집화
    - K-means, Cluster Evaluation(실루엣 계수), Mean Shift, GMM, DBSCAN

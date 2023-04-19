import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor, plot_importance

# 데이터 불러오기
data = pd.read_csv('your_data.csv')
X = data.drop(columns=['target_column'])
y = data['target_column']


# XGBoost 회귀 모델 정의
xgb_model = XGBRegressor()

# Cross-validation 수행 및 평가 지표 출력
scores = cross_val_score(xgb_model, X, y, cv=5, scoring='neg_mean_squared_error')
rmse_scores = [(-score)**0.5 for score in scores]
print("RMSE: %0.2f (+/- %0.2f)" % (rmse_scores.mean(), rmse_scores.std() * 2))

# XGBoost 회귀 모델 학습
xgb_model.fit(X, y)

# 피처별 중요도 출력
plot_importance(xgb_model)

# 예측 수행 및 평가
y_pred = xgb_model.predict(X)
rmse = mean_squared_error(y, y_pred, squared=False)
print("RMSE: %0.2f" % rmse)